#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from common import find_julia, load_toml, run_command, slug_float, write_run_config


def project_root() -> Path:
    # Installed location: PROJECT/benchmarks/prb_complete/scripts
    return SCRIPT_DIR.parents[2]


def benchmark_root() -> Path:
    return SCRIPT_DIR.parent


def create_job(name: str, model: dict[str, Any], nrg: dict[str, Any],
               config_root: Path, output_root: Path) -> dict[str, Any]:
    config_path = config_root / f"{name}.toml"
    run_output = output_root / "nhnrg" / name
    write_run_config(config_path, model=model, nrg=nrg, output_dir=run_output)
    return {
        "name": name,
        "config": config_path,
        "output": run_output,
        "model": model,
        "nrg": nrg,
    }


def build_jobs(cfg: dict[str, Any], profile: str, output_root: Path) -> list[dict[str, Any]]:
    physics = cfg["physics"]
    scan = cfg["scan"]
    nrg_cfg = cfg["nrg"]
    controls = cfg["controls"]
    config_root = output_root / "generated_configs"

    nkeep = int(
        nrg_cfg["nkeep_production"] if profile == "production"
        else nrg_cfg["nkeep_pilot"]
    )
    iterations = 8 if profile == "smoke" else int(nrg_cfg["iterations"])
    base_model = {
        "U": float(physics["U"]),
        "eps_d": float(physics["eps_d"]),
        "delta0": float(physics["delta0"]),
        "c_delta": float(physics["c_delta"]),
        "g_gamma": float(physics["g_gamma"]),
        "beta0": float(physics["beta_core"]),
        "delta_coh": 1.0e-3,
        "gamma_common": float(physics["gamma_common"]),
        "frame": str(physics["frame"]),
        "bandwidth": float(physics["bandwidth"]),
        "bath_exponent": float(physics["bath_exponent"]),
        "gamma_edge": float(physics["gamma_edge"]),
    }
    base_nrg = {
        "Lambda": float(nrg_cfg["Lambda"]),
        "z_shift": float(nrg_cfg["z_shift"]),
        "iterations": iterations,
        "nkeep": nkeep,
        "min_keep_per_charge": int(nrg_cfg["min_keep_per_charge"]),
        "sort_type": str(nrg_cfg["sort_type"]),
        "star_intervals": int(nrg_cfg["star_intervals"]),
        "overlap_floor": float(nrg_cfg["overlap_floor"]),
        "degeneracy_tolerance": float(nrg_cfg["degeneracy_tolerance"]),
        "residual_tolerance": float(nrg_cfg["residual_tolerance"]),
        "save_levels": int(nrg_cfg["save_levels"]),
        "lehmann_enabled": bool(nrg_cfg["lehmann_enabled"]),
    }

    jobs: list[dict[str, Any]] = []

    deltas = [1.0e-3] if profile == "smoke" else [float(x) for x in scan["delta_values"]]
    if profile == "pilot":
        deltas = [1.0e-2, 1.0e-3, 1.0e-4]
    for delta in deltas:
        model = dict(base_model, delta_coh=delta)
        jobs.append(create_job(
            f"delta_scan/delta_{slug_float(delta)}",
            model, dict(base_nrg), config_root, output_root
        ))

    if profile != "smoke":
        u_values = [0.0, 2.0, 4.0] if profile == "pilot" else [
            float(x) for x in scan["u_values"]
        ]
        u_deltas = [1.0e-3] if profile == "pilot" else [
            float(x) for x in scan["u_scan_delta_values"]
        ]
        for U in u_values:
            for delta in u_deltas:
                model = dict(base_model, U=U, eps_d=-0.5 * U, delta_coh=delta)
                jobs.append(create_job(
                    f"u_scan/U_{U:g}_delta_{slug_float(delta)}",
                    model, dict(base_nrg), config_root, output_root
                ))

        if bool(controls["run_side_points"]):
            for beta in [float(x) for x in scan["side_beta_values"]]:
                model = dict(base_model, beta0=beta, delta_coh=1.0e-3)
                jobs.append(create_job(
                    f"controls/side_beta_{beta:.3f}",
                    model, dict(base_nrg), config_root, output_root
                ))

        if bool(controls["run_metallic"]):
            model = dict(base_model, bath_exponent=0.0, delta_coh=1.0e-3)
            jobs.append(create_job(
                "controls/metallic_r0", model, dict(base_nrg),
                config_root, output_root
            ))

        if bool(controls["run_hermitian"]):
            model = dict(base_model, g_gamma=0.0, delta_coh=1.0e-3)
            jobs.append(create_job(
                "controls/hermitian_gammaPT0", model, dict(base_nrg),
                config_root, output_root
            ))

        if profile == "production" and bool(controls["run_convergence"]):
            seen: set[tuple[int, str, float]] = set()
            for nk in controls["convergence_nkeep"]:
                key = (int(nk), str(base_nrg["sort_type"]), float(base_nrg["Lambda"]))
                if key in seen:
                    continue
                seen.add(key)
                model = dict(base_model, delta_coh=1.0e-3)
                nrg = dict(base_nrg, nkeep=int(nk))
                jobs.append(create_job(
                    f"convergence/nkeep_{int(nk)}",
                    model, nrg, config_root, output_root
                ))
            for sort_type in controls["convergence_sort"]:
                key = (int(base_nrg["nkeep"]), str(sort_type), float(base_nrg["Lambda"]))
                if key in seen:
                    continue
                seen.add(key)
                model = dict(base_model, delta_coh=1.0e-3)
                nrg = dict(base_nrg, sort_type=str(sort_type))
                jobs.append(create_job(
                    f"convergence/sort_{sort_type}",
                    model, nrg, config_root, output_root
                ))
            for Lambda in controls["convergence_Lambda"]:
                key = (int(base_nrg["nkeep"]), str(base_nrg["sort_type"]), float(Lambda))
                if key in seen:
                    continue
                seen.add(key)
                model = dict(base_model, delta_coh=1.0e-3)
                nrg = dict(base_nrg, Lambda=float(Lambda))
                jobs.append(create_job(
                    f"convergence/Lambda_{float(Lambda):g}",
                    model, nrg, config_root, output_root
                ))

    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the complete PT-Dirac Bethe/saddle/RG/NH-NRG benchmark."
    )
    parser.add_argument("--config", type=Path,
                        default=benchmark_root() / "config" / "benchmark.toml")
    parser.add_argument("--profile", choices=("smoke", "pilot", "production"),
                        default="pilot")
    parser.add_argument("--julia", default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Skip NH-NRG jobs whose RUN_SUMMARY.txt exists.")
    parser.add_argument("--skip-nhnrg", action="store_true")
    parser.add_argument("--skip-saddle", action="store_true")
    parser.add_argument("--skip-rg", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    root = project_root()
    cfg = load_toml(args.config)
    output_root = root / str(cfg["paths"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "profile": args.profile,
        "project_root": str(root),
        "output_root": str(output_root),
        "jobs": [],
    }

    if not args.analyze_only and not args.skip_nhnrg:
        julia = find_julia(args.julia)
        jobs = build_jobs(cfg, args.profile, output_root)
        for index, job in enumerate(jobs, start=1):
            summary = job["output"] / "RUN_SUMMARY.txt"
            manifest["jobs"].append({
                "name": job["name"],
                "config": str(job["config"]),
                "output": str(job["output"]),
            })
            if args.resume and summary.exists():
                print(f"[{index}/{len(jobs)}] resume: {job['name']}")
                continue
            print(f"[{index}/{len(jobs)}] NH-NRG: {job['name']}")
            job["output"].parent.mkdir(parents=True, exist_ok=True)
            command = [
                julia, "--project=.",
                str(benchmark_root() / "scripts" / "run_one.jl"),
                str(job["config"]),
            ]
            run_command(
                command, root,
                output_root / "logs" / f"{job['name'].replace('/', '__')}.log"
            )

    auxiliary = cfg["auxiliary"]
    if (not args.analyze_only and not args.skip_saddle
            and bool(auxiliary["run_saddle_gate"])):
        saddle_out = output_root / "saddle"
        command = [
            sys.executable,
            str(benchmark_root() / "vendor" / "self_consistent_saddle_gate.py"),
            "--out", str(saddle_out),
        ]
        run_command(command, benchmark_root() / "vendor",
                    output_root / "logs" / "saddle.log")

    if (not args.analyze_only and not args.skip_rg
            and bool(auxiliary["run_causal_rg_gate"])):
        rg_out = output_root / "causal_rg"
        command = [
            sys.executable,
            str(benchmark_root() / "vendor" / "channel_resolved_rg_gate.py"),
            "--out", str(rg_out),
        ]
        run_command(command, benchmark_root() / "vendor",
                    output_root / "logs" / "causal_rg.log")

    (output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    analyze_command = [
        sys.executable,
        str(benchmark_root() / "scripts" / "analyze_benchmark.py"),
        "--config", str(args.config),
        "--output-root", str(output_root),
    ]
    run_command(analyze_command, root, output_root / "logs" / "analysis.log")


if __name__ == "__main__":
    main()
