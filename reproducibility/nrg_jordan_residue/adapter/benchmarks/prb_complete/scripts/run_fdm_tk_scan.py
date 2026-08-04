#!/usr/bin/env python3
"""Run equilibrium complete-basis/FDM thermodynamics versus beta0.

The runner deliberately sets delta_coh=0, frame=relative, and disables the
kept-space Lehmann calculation.  Exact beta_EP is skipped because the
biorthogonal eigensystem is defective there.  Broken-PT points may be included
as adverse controls; the Julia thermodynamic gate will return no T_K when the
complete-basis spectrum is complex.
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
    return f"{x:.16g}"


def write_config(path: Path, raw: dict[str, Any], beta: float, z_shift: float,
                 output_dir: Path, profile: str, scan_cfg: dict[str, Any]) -> None:
    model = dict(raw.get("model", {}))
    nrg = dict(raw.get("nrg", {}))
    leh = dict(raw.get("lehmann", {}))
    thermo = dict(scan_cfg.get("thermodynamics", {}))
    profiles = scan_cfg["profiles"]
    scan = scan_cfg["scan"]

    model["beta0"] = beta
    model["delta_coh"] = float(scan.get("delta_coh", 0.0))
    model["frame"] = "relative"
    nrg["z_shift"] = z_shift
    if profile == "smoke":
        nrg["iterations"] = int(profiles["smoke_iterations"])
        nrg["nkeep"] = int(profiles["smoke_nkeep"])
    elif profile == "pilot":
        nrg["iterations"] = int(profiles["pilot_iterations"])
        nrg["nkeep"] = int(profiles["pilot_nkeep"])
    else:
        nrg["iterations"] = int(profiles["production_iterations"])
        nrg["nkeep"] = int(profiles["production_nkeep"])
    nrg["min_keep_per_charge"] = max(1, int(nrg.get("min_keep_per_charge", 1)))
    leh["enabled"] = False
    thermo["enabled"] = True

    model_defaults = {
        "U": 2.0, "eps_d": -1.0, "delta0": 0.075, "c_delta": 0.05,
        "g_gamma": 0.2, "beta0": beta, "delta_coh": 0.0,
        "gamma_common": 0.12, "frame": "relative", "bandwidth": math.pi/4,
        "bath_exponent": 1.0, "gamma_edge": 0.12, "hybridization_V": math.nan,
        "bath_split": 0.0, "reciprocal_hybridization": True,
        "soc_mode": "none", "soc_lambda": 0.0, "soc_kmax": math.pi/4,
    }
    nrg_defaults = {
        "Lambda": 3.0, "z_shift": z_shift, "iterations": 30, "nkeep": 300,
        "min_keep_per_charge": 1, "sort_type": "LowRe", "star_intervals": 100,
        "overlap_floor": 1e-10, "degeneracy_tolerance": 1e-10,
        "residual_tolerance": 1e-9, "save_levels": 24,
    }
    leh_defaults = {"enabled": False, "omega_min": -1.0, "omega_max": 1.0,
                    "omega_points": 401, "eta": 0.01}
    thermo_defaults = {
        "enabled": True, "temperature_min": 1e-7, "temperature_max": 1.0,
        "temperature_points": 281, "imag_tolerance": 1e-8,
        "entropy_target": 0.5*math.log(2.0),
        "low_entropy_max": 0.30*math.log(2.0),
        "moment_entropy_min": 0.65*math.log(2.0),
    }
    model_defaults.update(model)
    nrg_defaults.update(nrg)
    leh_defaults.update(leh)
    thermo_defaults.update(thermo)

    sections = [
        ("model", model_defaults, ["U","eps_d","delta0","c_delta","g_gamma","beta0",
         "delta_coh","gamma_common","frame","bandwidth","bath_exponent","gamma_edge",
         "hybridization_V","bath_split","reciprocal_hybridization","soc_mode",
         "soc_lambda","soc_kmax"]),
        ("nrg", nrg_defaults, ["Lambda","z_shift","iterations","nkeep",
         "min_keep_per_charge","sort_type","star_intervals","overlap_floor",
         "degeneracy_tolerance","residual_tolerance","save_levels"]),
        ("lehmann", leh_defaults, ["enabled","omega_min","omega_max","omega_points","eta"]),
        ("thermodynamics", thermo_defaults, ["enabled","temperature_min","temperature_max",
         "temperature_points","imag_tolerance","entropy_target","low_entropy_max",
         "moment_entropy_min"]),
    ]
    lines: list[str] = []
    int_keys = {"iterations","nkeep","min_keep_per_charge","star_intervals","save_levels",
                "omega_points","temperature_points"}
    for name, values, order in sections:
        lines.append(f"[{name}]")
        for key in order:
            value = int(values[key]) if key in int_keys else values[key]
            lines.append(f"{key} = {fmt(value)}")
        lines.append("")
    lines += ["[output]", f'directory = "{output_dir.as_posix()}"', ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", type=Path, default=project_root()/"config"/"example.toml")
    ap.add_argument("--scan-config", type=Path,
                    default=SCRIPT_DIR.parent/"config"/"fdm_tk_scan.toml")
    ap.add_argument("--out", type=Path, default=project_root()/"output"/"fdm_tk_scan")
    ap.add_argument("--profile", choices=("smoke","pilot","production"), default="pilot")
    ap.add_argument("--julia", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-analyze", action="store_true")
    args = ap.parse_args()

    root = project_root()
    base = load_toml(args.base_config.resolve())
    scan_cfg = load_toml(args.scan_config.resolve())
    scan = scan_cfg["scan"]
    betas = [float(x) for x in scan[f"beta_values_{args.profile}"]]
    zvals = [float(x) for x in scan[f"z_values_{args.profile}"]]
    beta_ep = float(scan.get("beta_EP", 0.5))
    betas = [x for x in betas if abs(x-beta_ep) > 1e-12]
    out = args.out.resolve()
    generated, runs, logs = out/"generated_configs", out/"runs", out/"logs"
    out.mkdir(parents=True, exist_ok=True)
    julia = None if args.dry_run else find_julia(args.julia)
    manifest: dict[str, Any] = {"profile": args.profile, "beta_EP": beta_ep, "runs": []}

    total = len(betas)*len(zvals)
    index = 0
    for beta in betas:
        for z in zvals:
            index += 1
            name = f"beta_{beta:.6f}/z_{z:.3f}"
            run_dir = runs/name
            cfg = generated/f"beta_{beta:.6f}_z_{z:.3f}.toml"
            write_config(cfg, base, beta, z, run_dir, args.profile, scan_cfg)
            manifest["runs"].append({"beta0": beta, "z_shift": z,
                                     "config": str(cfg), "output": str(run_dir)})
            if args.dry_run:
                print(f"[{index}/{total}] dry-run beta={beta:g} z={z:g}")
                continue
            if args.resume and (run_dir/"fdm_tk_summary.toml").exists():
                print(f"[{index}/{total}] resume beta={beta:g} z={z:g}")
                continue
            print(f"[{index}/{total}] beta={beta:g} z={z:g}")
            cmd = [julia, "--project=.", str(SCRIPT_DIR/"run_one.jl"), str(cfg)]
            run_command(cmd, root, logs/f"beta_{beta:.6f}_z_{z:.3f}.log")

    manifest_path = out/"fdm_tk_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    if args.dry_run or args.no_analyze:
        print(f"Manifest: {manifest_path}")
        return
    subprocess.run([sys.executable, str(SCRIPT_DIR/"extract_fdm_tk.py"), str(out),
                    "--out", str(out/"analysis")], cwd=root, check=True)


if __name__ == "__main__":
    main()
