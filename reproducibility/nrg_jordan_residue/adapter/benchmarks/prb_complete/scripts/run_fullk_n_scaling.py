#!/usr/bin/env python3
"""Run complete-basis full-k matrix-Wilson chain-length scaling.

This driver is intentionally separate from the projected-overlap Julia
DM-NRG response scan.  It orchestrates the exact finite-chain center solver
and the branch-tracked local-discriminant contour solver bundled under
``fullk_matrix_chain``.

For every requested chain length N the EP center is re-solved.  The optional
topology stage then evaluates every N on one common parameter contour centered
on a selected reference-N solution, allowing pointwise comparison of the
finite-N discriminants.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
FULLK_ROOT = SCRIPT_DIR.parent / "fullk_matrix_chain"
CORE = FULLK_ROOT / "core"


def center_path(run_root: Path, n_value: int) -> Path:
    return run_root / f"N_{n_value:02d}" / "center" / "fullk_nrg_ep_predictor_summary.json"


def contour_path(run_root: Path, n_value: int) -> Path:
    return run_root / f"N_{n_value:02d}" / "common_contour" / "fullk_nrg_contour_pilot_summary.json"


def load_summary(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def same_float(left: object, right: object) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=1.0e-12, abs_tol=1.0e-14)
    except (TypeError, ValueError):
        return False


def same_center(left: object, right: object) -> bool:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    left_z = left.get("z")
    right_z = right.get("z")
    if not isinstance(left_z, list) or not isinstance(right_z, list) or len(left_z) != 2 or len(right_z) != 2:
        return False
    return (
        same_float(left_z[0], right_z[0])
        and same_float(left_z[1], right_z[1])
        and same_float(left.get("epsilon_d"), right.get("epsilon_d"))
        and same_float(left.get("W"), right.get("W"))
    )


def matching_center_result(path: Path, *, n_value: int, interaction: float, z_count: int) -> bool:
    payload = load_summary(path)
    if not payload or payload.get("status") != "CENTER_PASS":
        return False
    parameters = payload.get("parameters")
    history = payload.get("history")
    if not isinstance(parameters, dict) or not isinstance(history, list) or not history:
        return False
    flow_last = history[-1].get("flow_last") if isinstance(history[-1], dict) else None
    charges = flow_last.get("z_ground_charges") if isinstance(flow_last, dict) else None
    return (
        parameters.get("sites") == n_value
        and same_float(parameters.get("U"), interaction)
        and isinstance(charges, list)
        and len(charges) == z_count
    )


def matching_contour_result(
    path: Path,
    *,
    n_value: int,
    interaction: float,
    samples: int,
    gate_halfwidth: float,
    hybridization_halfwidth: float,
    center: dict[str, object],
) -> bool:
    payload = load_summary(path)
    accepted = {"COMPLETE_BASIS_TOPOLOGY_PASS", "COMPLETE_BASIS_TOPOLOGY_FAIL"}
    if not payload or payload.get("status") not in accepted:
        return False
    parameters = payload.get("parameters")
    contour = payload.get("contour")
    if not isinstance(parameters, dict) or not isinstance(contour, list):
        return False
    return (
        parameters.get("sites") == n_value
        and parameters.get("samples") == samples
        and same_float(parameters.get("U"), interaction)
        and same_float(parameters.get("gate_halfwidth"), gate_halfwidth)
        and same_float(parameters.get("hybridization_halfwidth"), hybridization_halfwidth)
        and same_center(parameters.get("center"), center)
        and len(contour) == samples + 1
    )


def run_command(
    command: list[str],
    log_path: Path,
    *,
    dry_run: bool,
    env: dict[str, str],
) -> None:
    printable = shlex.join(command)
    print(printable, flush=True)
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
        raise RuntimeError(
            f"command failed with exit code {result.returncode}: {printable}\n"
            + "\n".join(tail)
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "output" / "fullk_n_scaling")
    parser.add_argument("--n-values", type=int, nargs="+", default=[10, 12])
    parser.add_argument("--z-count", type=int, choices=(4, 8), default=4)
    parser.add_argument("--interaction", type=float, default=0.0025)
    parser.add_argument("--center-iterations", type=int, default=7)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--gate-halfwidth", type=float, default=1.0e-4)
    parser.add_argument("--hybridization-halfwidth", type=float, default=1.0e-3)
    parser.add_argument("--derivative-step", type=float, default=2.0e-6)
    parser.add_argument("--reference-n", type=int)
    parser.add_argument("--mode", choices=("centers", "common-contours", "all"), default="all")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    n_values = sorted(set(args.n_values))
    if not n_values or any(value < 2 for value in n_values):
        raise SystemExit("--n-values must contain integers >= 2")
    reference_n = args.reference_n if args.reference_n is not None else max(n_values)
    if reference_n not in n_values:
        raise SystemExit("--reference-n must be one of --n-values")

    preflight_name = "preflight_lowbands_z4" if args.z_count == 4 else "preflight_lowbands_z8_l15"
    preflight = CORE / preflight_name
    manifest = preflight / "z_chain_manifest.json"
    mapping = preflight / "full_k_wilson_preflight_summary.json"
    certificate = FULLK_ROOT / "full_k_two_cut_ep_certificate.py"
    ep_summary = FULLK_ROOT / "full_k_two_cut_ep_summary.json"
    bundled_seed = FULLK_ROOT / "reference_center_U00025.json"
    predictor = CORE / "fullk_nrg_ep_predictor.py"
    contour = CORE / "fullk_nrg_contour_pilot.py"
    exact_zavg = CORE / "exact_zavg_block_chain_response.py"

    required = [manifest, mapping, certificate, ep_summary, bundled_seed, predictor, contour, exact_zavg]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing full-k payload files:\n" + "\n".join(missing))

    run_root = args.out.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    logs = run_root / "logs"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(CORE) + os.pathsep + env.get("PYTHONPATH", "")

    records: list[dict[str, object]] = []
    seed = bundled_seed
    if args.mode in {"centers", "all"}:
        for n_value in n_values:
            destination = run_root / f"N_{n_value:02d}" / "center"
            summary = center_path(run_root, n_value)
            command = [
                args.python,
                str(predictor),
                "--nrg-source", str(exact_zavg),
                "--certificate-source", str(certificate),
                "--chain", str(manifest),
                "--mapping-summary", str(mapping),
                "--ep-summary", str(ep_summary),
                "--initial-center-summary", str(seed),
                "--out", str(destination),
                "--interaction", str(args.interaction),
                "--sites", str(n_value),
                "--cluster-sites", "0",
                "--nkeep", "0",
                "--outer-iterations", str(args.center_iterations),
                "--derivative-step", str(args.derivative_step),
            ]
            skipped = bool(
                args.resume
                and matching_center_result(
                    summary,
                    n_value=n_value,
                    interaction=args.interaction,
                    z_count=args.z_count,
                )
            )
            if skipped:
                print(f"resume center N={n_value}: {summary}", flush=True)
            else:
                run_command(command, logs / f"N_{n_value:02d}_center.log", dry_run=args.dry_run, env=env)
            if summary.exists():
                payload = json.loads(summary.read_text(encoding="utf-8"))
                if payload.get("status") == "CENTER_PASS":
                    seed = summary
            records.append({
                "stage": "center",
                "N": n_value,
                "summary": str(summary),
                "skipped": skipped,
                "command": command,
            })

    if args.mode in {"common-contours", "all"}:
        common_center = center_path(run_root, reference_n)
        reference_center_matches = matching_center_result(
            common_center,
            n_value=reference_n,
            interaction=args.interaction,
            z_count=args.z_count,
        )
        if not args.dry_run and not reference_center_matches:
            raise SystemExit(
                f"reference center is unavailable, did not pass, or was produced with "
                f"different N/U/z settings: {common_center}. Run --mode centers first."
            )
        if args.dry_run and not reference_center_matches:
            common_center = bundled_seed
        common_center_summary = load_summary(common_center)
        if not common_center_summary:
            raise SystemExit(f"could not read reference center: {common_center}")
        common_center_payload = common_center_summary.get("predicted_center")
        if not isinstance(common_center_payload, dict):
            raise SystemExit(f"reference center lacks predicted_center: {common_center}")
        for n_value in n_values:
            destination = run_root / f"N_{n_value:02d}" / "common_contour"
            summary = contour_path(run_root, n_value)
            command = [
                args.python,
                str(contour),
                "--nrg-source", str(exact_zavg),
                "--certificate-source", str(certificate),
                "--chain", str(manifest),
                "--mapping-summary", str(mapping),
                "--center-summary", str(common_center),
                "--out", str(destination),
                "--interaction", str(args.interaction),
                "--samples", str(args.samples),
                "--gate-halfwidth", str(args.gate_halfwidth),
                "--hybridization-halfwidth", str(args.hybridization_halfwidth),
                "--sites", str(n_value),
                "--cluster-sites", "0",
                "--nkeep", "0",
                "--derivative-step", str(args.derivative_step),
            ]
            skipped = bool(
                args.resume
                and matching_contour_result(
                    summary,
                    n_value=n_value,
                    interaction=args.interaction,
                    samples=args.samples,
                    gate_halfwidth=args.gate_halfwidth,
                    hybridization_halfwidth=args.hybridization_halfwidth,
                    center=common_center_payload,
                )
            )
            if skipped:
                print(f"resume common contour N={n_value}: {summary}", flush=True)
            else:
                run_command(
                    command,
                    logs / f"N_{n_value:02d}_common_contour.log",
                    dry_run=args.dry_run,
                    env=env,
                )
            records.append({
                "stage": "common_contour",
                "N": n_value,
                "summary": str(summary),
                "skipped": skipped,
                "command": command,
            })

    payload = {
        "status": "DRY_RUN" if args.dry_run else "RUN_COMPLETE",
        "scope": (
            "Complete-basis finite matrix-Wilson chain-length scaling. "
            "This is not by itself a thermodynamic-limit FDM-NRG or Kondo-scale claim."
        ),
        "parameters": {
            "N_values": n_values,
            "reference_N": reference_n,
            "z_count": args.z_count,
            "U": args.interaction,
            "samples": args.samples,
            "gate_halfwidth": args.gate_halfwidth,
            "hybridization_halfwidth": args.hybridization_halfwidth,
        },
        "records": records,
    }
    manifest_path = run_root / "fullk_n_scaling_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
