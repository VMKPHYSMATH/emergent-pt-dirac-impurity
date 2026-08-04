#!/usr/bin/env python3
"""Run a beta0 scan for operational low-energy-scale extraction.

The runner copies the current adapter model/NRG/Lehmann settings from a base
TOML file, changes only beta0 and the explicitly configured profile overrides,
and writes one ordinary adapter run directory per beta0.  It does not call any
new Hamiltonian or change the NRG recursion.
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
    x = float(value)
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return f"{x:.16g}"


def write_config(path: Path, raw: dict[str, Any], beta: float,
                 delta_coh: float, output_dir: Path, profile: str,
                 scan_cfg: dict[str, Any]) -> None:
    model = dict(raw.get("model", {}))
    nrg = dict(raw.get("nrg", {}))
    leh = dict(raw.get("lehmann", {}))
    model["beta0"] = float(beta)
    model["delta_coh"] = float(delta_coh)

    profiles = scan_cfg.get("profiles", {})
    if profile == "smoke":
        nrg["iterations"] = min(int(nrg.get("iterations", 30)), int(profiles.get("smoke_iterations", 6)))
        nrg["nkeep"] = min(int(nrg.get("nkeep", 300)), int(profiles.get("smoke_nkeep", 96)))
        leh["enabled"] = bool(profiles.get("smoke_lehmann", False))
    elif profile == "pilot":
        nrg["iterations"] = min(int(nrg.get("iterations", 30)), int(profiles.get("pilot_iterations", 18)))
        nrg["nkeep"] = min(int(nrg.get("nkeep", 300)), int(profiles.get("pilot_nkeep", 240)))
        leh["enabled"] = bool(profiles.get("pilot_lehmann", True))
    else:
        leh["enabled"] = bool(profiles.get("production_lehmann", True))

    defaults_model = {
        "U": 2.0, "eps_d": -1.0, "delta0": 0.075, "c_delta": 0.05,
        "g_gamma": 0.2, "beta0": beta, "delta_coh": delta_coh,
        "gamma_common": 0.12, "frame": "relative", "bandwidth": math.pi / 4,
        "bath_exponent": 1.0, "gamma_edge": 0.12, "hybridization_V": math.nan,
        "bath_split": 0.0, "reciprocal_hybridization": True,
        "soc_mode": "none", "soc_lambda": 0.0, "soc_kmax": math.pi / 4,
    }
    defaults_nrg = {
        "Lambda": 3.0, "z_shift": 0.5, "iterations": 30, "nkeep": 300,
        "min_keep_per_charge": 1, "sort_type": "LowRe", "star_intervals": 100,
        "overlap_floor": 1e-10, "degeneracy_tolerance": 1e-10,
        "residual_tolerance": 1e-9, "save_levels": 64,
    }
    defaults_leh = {
        "enabled": True, "omega_min": -1.0, "omega_max": 1.0,
        "omega_points": 2001, "eta": 0.01,
    }
    defaults_model.update(model)
    defaults_nrg.update(nrg)
    defaults_leh.update(leh)

    lines: list[str] = ["[model]"]
    model_order = [
        "U", "eps_d", "delta0", "c_delta", "g_gamma", "beta0", "delta_coh",
        "gamma_common", "frame", "bandwidth", "bath_exponent", "gamma_edge",
        "hybridization_V", "bath_split", "reciprocal_hybridization", "soc_mode",
        "soc_lambda", "soc_kmax",
    ]
    for key in model_order:
        lines.append(f"{key} = {fmt(defaults_model[key])}")

    lines += ["", "[nrg]"]
    nrg_order = [
        "Lambda", "z_shift", "iterations", "nkeep", "min_keep_per_charge",
        "sort_type", "star_intervals", "overlap_floor", "degeneracy_tolerance",
        "residual_tolerance", "save_levels",
    ]
    int_keys = {"iterations", "nkeep", "min_keep_per_charge", "star_intervals", "save_levels"}
    for key in nrg_order:
        value = int(defaults_nrg[key]) if key in int_keys else defaults_nrg[key]
        lines.append(f"{key} = {fmt(value)}")

    lines += ["", "[lehmann]"]
    leh_order = ["enabled", "omega_min", "omega_max", "omega_points", "eta"]
    for key in leh_order:
        value = int(defaults_leh[key]) if key == "omega_points" else defaults_leh[key]
        lines.append(f"{key} = {fmt(value)}")

    lines += ["", "[output]", f'directory = "{output_dir.as_posix()}"', ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, default=project_root() / "config" / "example.toml")
    parser.add_argument("--scan-config", type=Path,
                        default=SCRIPT_DIR.parent / "config" / "beta_scale_scan.toml")
    parser.add_argument("--out", type=Path, default=project_root() / "output" / "beta_scale_scan")
    parser.add_argument("--profile", choices=("smoke", "pilot", "production"), default="pilot")
    parser.add_argument("--julia", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-analyze", action="store_true")
    args = parser.parse_args()

    root = project_root()
    base = load_toml(args.base_config.resolve())
    scan_cfg = load_toml(args.scan_config.resolve())
    scan = scan_cfg["scan"]
    betas = [float(x) for x in scan[f"beta_values_{args.profile}"]]
    delta = float(scan.get("delta_coh", base.get("model", {}).get("delta_coh", 1e-3)))
    out = args.out.resolve()
    generated = out / "generated_configs"
    runs = out / "runs"
    logs = out / "logs"
    out.mkdir(parents=True, exist_ok=True)

    julia = None if args.dry_run else find_julia(args.julia)
    manifest: dict[str, Any] = {
        "base_config": str(args.base_config.resolve()),
        "scan_config": str(args.scan_config.resolve()),
        "profile": args.profile,
        "delta_coh": delta,
        "runs": [],
    }
    for index, beta in enumerate(betas, start=1):
        name = f"beta_{beta:.6f}"
        run_dir = runs / name
        cfg_path = generated / f"{name}.toml"
        write_config(cfg_path, base, beta, delta, run_dir, args.profile, scan_cfg)
        manifest["runs"].append({"beta0": beta, "config": str(cfg_path), "output": str(run_dir)})
        if args.dry_run:
            print(f"[{index}/{len(betas)}] dry-run {name}: {cfg_path}")
            continue
        if args.resume and (run_dir / "RUN_SUMMARY.txt").exists():
            print(f"[{index}/{len(betas)}] resume {name}")
            continue
        command = [julia, "--project=.",
                   str(SCRIPT_DIR / "run_one.jl"), str(cfg_path)]
        print(f"[{index}/{len(betas)}] beta0={beta:.6g}")
        run_command(command, root, logs / f"{name}.log")

    (out / "beta_scale_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if args.dry_run or args.no_analyze:
        print(f"Manifest: {out / 'beta_scale_manifest.json'}")
        return

    command = [
        sys.executable, str(SCRIPT_DIR / "extract_beta_scales.py"),
        str(out), "--scan-config", str(args.scan_config.resolve()),
        "--out", str(out / "analysis"),
    ]
    subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
