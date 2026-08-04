#!/usr/bin/env python3
"""Run zero-temperature complete-basis DM-NRG spin-response scans.

The runner changes only beta0, U, the projected SOC overlap ratio, z shift,
and numerical profile.  It does not insert an EP minimum or a peak model.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from common import find_julia, load_toml, run_command


def project_root() -> Path:
    return SCRIPT_DIR.parents[2]


def fmt(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, int):
        return str(value)
    x = float(value)
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "-inf" if x < 0 else "inf"
    return f"{x:.16g}"


def write_section(lines: list[str], name: str, values: dict[str, Any], order: list[str]) -> None:
    lines.append(f"[{name}]")
    for key in order:
        value = values[key]
        if isinstance(value, list):
            encoded = ", ".join(fmt(x) for x in value)
            lines.append(f"{key} = [{encoded}]")
        else:
            lines.append(f"{key} = {fmt(value)}")
    lines.append("")


def write_config(path: Path, base: dict[str, Any], scan_cfg: dict[str, Any],
                 beta: float, U: float, soc_ratio: float, z: float,
                 profile: str, output: Path) -> None:
    model = {
        "U": 2.0, "eps_d": -1.0, "delta0": 0.075, "c_delta": 0.05,
        "g_gamma": 0.2, "beta0": beta, "delta_coh": 0.0,
        "gamma_common": 0.12, "frame": "relative", "bandwidth": math.pi/4,
        "bath_exponent": 1.0, "gamma_edge": 0.12, "hybridization_V": math.nan,
        "bath_split": 0.0, "reciprocal_hybridization": True,
        "soc_mode": "none", "soc_lambda": 0.0, "soc_kmax": math.pi/4,
    }
    model.update(base.get("model", {}))
    model.update({"U": U, "beta0": beta,
                  "delta_coh": float(scan_cfg["scan"].get("delta_coh", 0.0)),
                  "frame": "relative"})
    kmax = float(model["soc_kmax"])
    model["soc_lambda"] = soc_ratio * kmax
    model["soc_mode"] = "none" if abs(soc_ratio) < 1e-15 else "overlap"

    nrg = {
        "Lambda": 3.0, "z_shift": z, "iterations": 16, "nkeep": 96,
        "min_keep_per_charge": 2, "sort_type": "LowRe", "star_intervals": 100,
        "overlap_floor": 1e-10, "degeneracy_tolerance": 1e-10,
        "residual_tolerance": 1e-9, "save_levels": 24,
    }
    nrg.update(base.get("nrg", {}))
    profiles = scan_cfg["profiles"]
    nrg["z_shift"] = z
    nrg["iterations"] = int(profiles[f"{profile}_iterations"])
    nrg["nkeep"] = int(profiles[f"{profile}_nkeep"])

    leh = {"enabled": False, "omega_min": -1.0, "omega_max": 1.0,
           "omega_points": 401, "eta": 0.01}
    thermo = {"enabled": False, "temperature_min": 1e-7,
              "temperature_max": 1.0, "temperature_points": 121,
              "imag_tolerance": 1e-8, "entropy_target": 0.5*math.log(2),
              "low_entropy_max": 0.30*math.log(2),
              "moment_entropy_min": 0.65*math.log(2)}
    response = dict(scan_cfg["dmnrg_response"])

    lines: list[str] = []
    write_section(lines, "model", model,
                  ["U","eps_d","delta0","c_delta","g_gamma","beta0",
                   "delta_coh","gamma_common","frame","bandwidth","bath_exponent",
                   "gamma_edge","hybridization_V","bath_split",
                   "reciprocal_hybridization","soc_mode","soc_lambda","soc_kmax"])
    write_section(lines, "nrg", nrg,
                  ["Lambda","z_shift","iterations","nkeep","min_keep_per_charge",
                   "sort_type","star_intervals","overlap_floor",
                   "degeneracy_tolerance","residual_tolerance","save_levels"])
    write_section(lines, "lehmann", leh,
                  ["enabled","omega_min","omega_max","omega_points","eta"])
    write_section(lines, "thermodynamics", thermo,
                  ["enabled","temperature_min","temperature_max",
                   "temperature_points","imag_tolerance","entropy_target",
                   "low_entropy_max","moment_entropy_min"])
    write_section(lines, "dmnrg_response", response,
                  ["omega_min","omega_max","omega_points","eta","imag_tolerance",
                   "components","weight_fraction","weight_floor",
                   "max_transitions_per_shell","time_max","time_points",
                   "exploratory_complex"])
    lines += ["[output]", f'directory = "{output.as_posix()}"', ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_csv_filter(value: str | None) -> list[float] | None:
    if value is None:
        return None
    return [float(x) for x in value.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", type=Path, default=project_root()/"config"/"example.toml")
    ap.add_argument("--scan-config", type=Path,
                    default=SCRIPT_DIR.parent/"config"/"dmnrg_response_scan.toml")
    ap.add_argument("--out", type=Path, default=project_root()/"output"/"dmnrg_response_scan")
    ap.add_argument("--profile", choices=("smoke","pilot","matrix"), default="smoke")
    ap.add_argument("--only-beta", default=None,
                    help="comma-separated beta0 values overriding the profile")
    ap.add_argument("--only-u", default=None,
                    help="comma-separated U values overriding the profile")
    ap.add_argument("--only-soc-ratio", default=None,
                    help="comma-separated lambda/kmax values overriding the profile")
    ap.add_argument("--julia", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-analyze", action="store_true")
    args = ap.parse_args()

    root = project_root()
    base = load_toml(args.base_config.resolve())
    scan_cfg = load_toml(args.scan_config.resolve())
    scan = scan_cfg["scan"]
    profile = args.profile
    betas = parse_csv_filter(args.only_beta) or [float(x) for x in scan[f"beta_values_{profile}"]]
    Uvals = parse_csv_filter(args.only_u) or [float(x) for x in scan[f"U_values_{profile}"]]
    socvals = parse_csv_filter(args.only_soc_ratio) or [float(x) for x in scan[f"soc_ratio_values_{profile}"]]
    zvals = [float(x) for x in scan[f"z_values_{profile}"]]
    beta_ep = float(scan.get("beta_EP", 0.5))
    betas = [b for b in betas if abs(b-beta_ep) > 1e-12]

    out = args.out.resolve()
    generated = out/"generated_configs"
    runs = out/"runs"
    logs = out/"logs"
    out.mkdir(parents=True, exist_ok=True)
    julia = None if args.dry_run else find_julia(args.julia)
    manifest: dict[str, Any] = {"profile": profile, "beta_EP": beta_ep, "runs": []}

    jobs = [(b, U, s, z) for U in Uvals for s in socvals for b in betas for z in zvals]
    for index, (beta, U, soc, z) in enumerate(jobs, start=1):
        name = f"U_{U:.6f}/soc_{soc:.6f}/beta_{beta:.6f}/z_{z:.3f}"
        run_dir = runs/name
        cfg = generated/f"U_{U:.6f}_soc_{soc:.6f}_beta_{beta:.6f}_z_{z:.3f}.toml"
        write_config(cfg, base, scan_cfg, beta, U, soc, z, profile, run_dir)
        manifest["runs"].append({"beta0": beta, "U": U, "soc_ratio": soc,
                                 "z_shift": z, "config": str(cfg),
                                 "output": str(run_dir)})
        if args.dry_run:
            print(f"[{index}/{len(jobs)}] dry-run beta={beta:g} U={U:g} soc={soc:g} z={z:g}")
            continue
        if args.resume and (run_dir/"dmnrg_response_summary.toml").exists():
            print(f"[{index}/{len(jobs)}] resume beta={beta:g} U={U:g} soc={soc:g} z={z:g}")
            continue
        print(f"[{index}/{len(jobs)}] beta={beta:g} U={U:g} soc={soc:g} z={z:g}")
        cmd = [julia, "--project=.", str(SCRIPT_DIR/"run_dmnrg_response_one.jl"), str(cfg)]
        safe = f"U_{U:.6f}_soc_{soc:.6f}_beta_{beta:.6f}_z_{z:.3f}.log"
        run_command(cmd, root, logs/safe)

    manifest_path = out/"dmnrg_response_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    if args.dry_run or args.no_analyze:
        print(f"Manifest: {manifest_path}")
        return
    subprocess.run([sys.executable, str(SCRIPT_DIR/"extract_dmnrg_response.py"),
                    str(out), "--out", str(out/"analysis")], cwd=root, check=True)


if __name__ == "__main__":
    main()
