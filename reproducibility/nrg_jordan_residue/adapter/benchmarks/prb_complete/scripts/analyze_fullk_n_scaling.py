#!/usr/bin/env python3
"""Analyze full-k matrix-chain N scaling and common-contour stability."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


COORDINATES = ("Re_z", "Im_z", "epsilon_d", "W")


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def center_summary(run_root: Path, n_value: int) -> Path:
    return run_root / f"N_{n_value:02d}" / "center" / "fullk_nrg_ep_predictor_summary.json"


def contour_summary(run_root: Path, n_value: int) -> Path:
    return run_root / f"N_{n_value:02d}" / "common_contour" / "fullk_nrg_contour_pilot_summary.json"


def discover_n_values(run_root: Path) -> list[int]:
    values = []
    for path in run_root.glob("N_[0-9][0-9]"):
        try:
            values.append(int(path.name.split("_", 1)[1]))
        except ValueError:
            continue
    return sorted(set(values))


def center_row(n_value: int, payload: dict[str, Any], bandwidth: float, wilson_lambda: float) -> dict[str, Any]:
    center = payload["predicted_center"]
    z = complex(*center["z"])
    history = payload.get("history", [])
    final = history[-1] if history else {}
    singular = final.get("kernel_singular_values", [math.nan, math.nan])
    flow_last = final.get("flow_last", {})
    ground_charges = flow_last.get("z_ground_charges", [])
    return {
        "N": n_value,
        "omega_N": bandwidth * wilson_lambda ** (-(n_value - 1) / 2.0),
        "center_status": payload.get("status", "UNKNOWN"),
        "Re_z": z.real,
        "Im_z": z.imag,
        "epsilon_d": float(center["epsilon_d"]),
        "W": float(center["W"]),
        "center_displacement": float(final.get("displacement", math.nan)),
        "F_modulus": float(final.get("F_modulus", math.nan)),
        "F_z_modulus": float(final.get("F_z_modulus", math.nan)),
        "F_zz_modulus": float(abs(complex(*final.get("F_zz", [math.nan, math.nan])))),
        "kernel_small_singular": float(singular[-1]) if singular else math.nan,
        "sumrule_error_center": float(final.get("sumrule_error", math.nan)),
        "z_ground_charges": json.dumps(ground_charges),
        "uniform_z_ground_charge": bool(ground_charges and len(set(ground_charges)) == 1),
    }


def contour_fields(payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {
            "contour_status": "MISSING",
            "winding": math.nan,
            "half_winding": math.nan,
            "permutation": "--",
            "minimum_boundary_modulus": math.nan,
            "maximum_phase_increment": math.nan,
            "minimum_F_zz_modulus": math.nan,
            "maximum_F_z_residual": math.nan,
            "maximum_assignment_ratio": math.nan,
            "maximum_sumrule_error": math.nan,
        }
    cert = payload.get("certificate", {})
    return {
        "contour_status": payload.get("status", "UNKNOWN"),
        "winding": float(cert.get("winding", math.nan)),
        "half_winding": float(cert.get("half_winding", math.nan)),
        "permutation": json.dumps(cert.get("endpoint_permutation", [])),
        "minimum_boundary_modulus": float(cert.get("minimum_boundary_modulus", math.nan)),
        "maximum_phase_increment": float(cert.get("maximum_phase_increment", math.nan)),
        "minimum_F_zz_modulus": float(cert.get("minimum_F_zz_modulus", math.nan)),
        "maximum_F_z_residual": float(cert.get("maximum_F_z_residual", math.nan)),
        "maximum_assignment_ratio": float(
            cert.get("maximum_assignment_to_half_separation_ratio", math.nan)
        ),
        "maximum_sumrule_error": float(cert.get("maximum_sumrule_error", math.nan)),
    }


def contour_array(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = payload.get("contour", [])
    epsilon = np.asarray([float(row["epsilon_d"]) for row in rows])
    width = np.asarray([float(row["W"]) for row in rows])
    discriminant = np.asarray([complex(*row["D"]) for row in rows])
    return epsilon, width, discriminant


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite(value: float) -> bool:
    return math.isfinite(float(value))


def make_plot(out: Path, rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not rows:
        return

    n_values = np.asarray([row["N"] for row in rows], dtype=float)
    omega = np.asarray([row["omega_N"] for row in rows], dtype=float)
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 7.2), constrained_layout=True)

    for key, label in (("Re_z", r"$\mathrm{Re}\,z_{\rm EP}$"),
                       ("Im_z", r"$\mathrm{Im}\,z_{\rm EP}$"),
                       ("epsilon_d", r"$\epsilon_d^{\rm EP}$"),
                       ("W", r"$W^{\rm EP}$")):
        axes[0, 0].plot(omega, [row[key] for row in rows], "o-", label=label)
    axes[0, 0].set_xlabel(r"$\omega_N=D\Lambda^{-(N-1)/2}$")
    axes[0, 0].set_ylabel("reoptimized center")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(n_values, [row["winding"] for row in rows], "o-", label="winding")
    axes[0, 1].plot(n_values, [row["half_winding"] for row in rows], "s-", label="half-winding")
    axes[0, 1].axhline(-1.0, color="0.5", lw=0.8, ls="--")
    axes[0, 1].axhline(-0.5, color="0.5", lw=0.8, ls=":")
    axes[0, 1].set_xlabel("chain length N")
    axes[0, 1].set_ylabel("topological index")
    axes[0, 1].legend(fontsize=8)

    boundary = np.asarray([row["minimum_boundary_modulus"] for row in rows], dtype=float)
    axes[1, 0].semilogy(n_values, boundary, "o-", label=r"$\min_C|\mathcal{D}_N|$")
    if pair_rows:
        pair_n = [row["N_low"] for row in pair_rows]
        axes[1, 0].semilogy(pair_n, [row["sup_discriminant_difference"] for row in pair_rows],
                           "s-", label=r"$\|\mathcal{D}_{N'}-\mathcal{D}_N\|_{\infty,C}$")
    axes[1, 0].set_xlabel("chain length N")
    axes[1, 0].set_ylabel("common-contour modulus")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].semilogy(n_values, [row["maximum_assignment_ratio"] for row in rows],
                       "o-", label="assignment ratio")
    axes[1, 1].semilogy(n_values, [row["maximum_sumrule_error"] for row in rows],
                       "s-", label="sum-rule error")
    axes[1, 1].axhline(0.5, color="tab:red", lw=0.8, ls="--", label="tracking gate")
    axes[1, 1].set_xlabel("chain length N")
    axes[1, 1].set_ylabel("gate value")
    axes[1, 1].legend(fontsize=8)

    for suffix in ("pdf", "png"):
        figure.savefig(out / f"fullk_n_scaling.{suffix}", dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--n-values", type=int, nargs="+")
    parser.add_argument("--wilson-lambda", type=float, default=1.5)
    parser.add_argument("--bandwidth", type=float, default=math.pi / 4.0)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    out = (args.out or run_root / "analysis").resolve()
    out.mkdir(parents=True, exist_ok=True)
    n_values = sorted(set(args.n_values or discover_n_values(run_root)))
    if not n_values:
        raise SystemExit(f"no N directories found under {run_root}")

    rows: list[dict[str, Any]] = []
    contours: dict[int, dict[str, Any]] = {}
    for n_value in n_values:
        center = load_json(center_summary(run_root, n_value))
        if center is None or "predicted_center" not in center:
            continue
        contour = load_json(contour_summary(run_root, n_value))
        row = center_row(n_value, center, args.bandwidth, args.wilson_lambda)
        row.update(contour_fields(contour))
        rows.append(row)
        if contour is not None and contour.get("contour"):
            contours[n_value] = contour

    if not rows:
        raise SystemExit("no readable center summaries found")

    center_pair_rows: list[dict[str, Any]] = []
    for low_row, high_row in zip(rows, rows[1:]):
        absolute = {
            key: abs(float(high_row[key]) - float(low_row[key])) for key in COORDINATES
        }
        relative = {
            key: absolute[key] / max(abs(float(high_row[key])), 1.0e-14)
            for key in COORDINATES
        }
        center_pair_rows.append({
            "N_low": low_row["N"],
            "N_high": high_row["N"],
            **{f"abs_{key}": value for key, value in absolute.items()},
            **{f"rel_{key}": value for key, value in relative.items()},
            "maximum_absolute_shift": max(absolute.values()),
            "maximum_relative_shift": max(relative.values()),
        })

    pair_rows: list[dict[str, Any]] = []
    contour_n = sorted(contours)
    for low, high in zip(contour_n, contour_n[1:]):
        epsilon_low, width_low, disc_low = contour_array(contours[low])
        epsilon_high, width_high, disc_high = contour_array(contours[high])
        same_grid = bool(
            len(disc_low) == len(disc_high)
            and np.allclose(epsilon_low, epsilon_high, rtol=0.0, atol=1.0e-13)
            and np.allclose(width_low, width_high, rtol=0.0, atol=1.0e-13)
        )
        if not same_grid:
            pair_rows.append({
                "N_low": low,
                "N_high": high,
                "same_contour_grid": False,
                "minimum_base_modulus": math.nan,
                "sup_discriminant_difference": math.nan,
                "conservative_ratio": math.nan,
                "maximum_pointwise_ratio": math.nan,
                "rouche_sufficient_pass": False,
            })
            continue
        difference = np.abs(disc_high - disc_low)
        base_modulus = np.abs(disc_low)
        minimum = float(np.min(base_modulus))
        supremum = float(np.max(difference))
        pointwise = float(np.max(difference / np.maximum(base_modulus, 1.0e-300)))
        pair_rows.append({
            "N_low": low,
            "N_high": high,
            "same_contour_grid": True,
            "minimum_base_modulus": minimum,
            "sup_discriminant_difference": supremum,
            "conservative_ratio": supremum / minimum if minimum > 0 else math.inf,
            "maximum_pointwise_ratio": pointwise,
            "rouche_sufficient_pass": bool(supremum < minimum),
        })

    extrapolations: dict[str, Any] = {}
    omega = np.asarray([row["omega_N"] for row in rows], dtype=float)
    if len(rows) >= 3:
        for coordinate in COORDINATES:
            values = np.asarray([row[coordinate] for row in rows], dtype=float)
            linear = np.polyfit(omega, values, deg=1)
            prediction = np.polyval(linear, omega)
            extrapolations[coordinate] = {
                "linear_limit_omega_to_zero": float(linear[-1]),
                "linear_slope": float(linear[0]),
                "linear_rms_residual": float(np.sqrt(np.mean((values - prediction) ** 2))),
            }

    center_pass = all(row["center_status"] == "CENTER_PASS" for row in rows)
    topology_rows = [row for row in rows if row["contour_status"] != "MISSING"]
    topology_pass = bool(topology_rows) and all(
        row["contour_status"] == "COMPLETE_BASIS_TOPOLOGY_PASS"
        and finite(row["winding"])
        and abs(abs(row["winding"]) - 1.0) < 0.05
        and finite(row["half_winding"])
        and abs(abs(row["half_winding"]) - 0.5) < 0.05
        and row["permutation"] == "[1, 0]"
        for row in topology_rows
    )
    rouche_all = bool(pair_rows) and all(row["rouche_sufficient_pass"] for row in pair_rows)
    if not topology_rows:
        status = "CENTER_SCALING_ONLY"
    elif center_pass and topology_pass and rouche_all:
        status = "FINITE_N_COMMON_CONTOUR_ROUCHE_PASS"
    elif center_pass and topology_pass:
        status = "FINITE_N_TOPOLOGY_PASS_ROUCHE_UNRESOLVED"
    else:
        status = "N_SCALING_GATE_NOT_PASSED"

    result = {
        "status": status,
        "scope": (
            "Finite-chain full-k matrix-Wilson N scaling. Pairwise Rouché gates, when passed, "
            "preserve the enclosed zero count between the reported finite N values. They do not "
            "alone prove a thermodynamic FDM-NRG or strong-coupling Kondo limit."
        ),
        "parameters": {
            "N_values": [row["N"] for row in rows],
            "wilson_lambda": args.wilson_lambda,
            "bandwidth": args.bandwidth,
        },
        "center_and_topology": rows,
        "center_pairwise": center_pair_rows,
        "common_contour_pairwise": pair_rows,
        "center_extrapolations": extrapolations,
        "gates": {
            "all_centers_pass": center_pass,
            "all_available_topology_gates_pass": topology_pass,
            "all_pairwise_conservative_rouche_gates_pass": rouche_all,
        },
    }
    (out / "fullk_n_scaling_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(out / "fullk_n_scaling_centers.csv", rows)
    write_csv(out / "fullk_n_scaling_center_pairs.csv", center_pair_rows)
    write_csv(out / "fullk_n_scaling_rouche_pairs.csv", pair_rows)
    make_plot(out, rows, pair_rows)

    lines = [
        "# Full-k matrix-chain N-scaling audit",
        "",
        f"- status: **{status}**",
        f"- chain lengths: **{', '.join(str(row['N']) for row in rows)}**",
        f"- Wilson Lambda: **{args.wilson_lambda:g}**",
        "",
        "The centers are re-solved independently at every N. Topology runs use one common",
        "parameter contour. A pairwise Rouché pass is the sufficient numerical inequality",
        "`sup_C |D_high-D_low| < min_C |D_low|` on that common contour.",
        "",
        "| N | center | z charges | contour | winding | half-winding | permutation | min boundary | assignment | sum rule |",
        "|--:|:--:|:--:|:--:|--:|--:|:--:|--:|--:|--:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['N']} | {row['center_status']} | {row['z_ground_charges']} | {row['contour_status']} | "
            f"{row['winding']:.8g} | {row['half_winding']:.8g} | {row['permutation']} | "
            f"{row['minimum_boundary_modulus']:.3e} | {row['maximum_assignment_ratio']:.3e} | "
            f"{row['maximum_sumrule_error']:.3e} |"
        )
    lines += [
        "",
        "## Reoptimized-center drift",
        "",
        "| N pair | max absolute shift | max relative shift |",
        "|:--:|--:|--:|",
    ]
    for row in center_pair_rows:
        lines.append(
            f"| {row['N_low']}->{row['N_high']} | {row['maximum_absolute_shift']:.3e} | "
            f"{row['maximum_relative_shift']:.3e} |"
        )
    lines += [
        "",
        "## Pairwise common-contour checks",
        "",
        "| N pair | sup difference | min base modulus | conservative ratio | Rouché pass |",
        "|:--:|--:|--:|--:|:--:|",
    ]
    for row in pair_rows:
        lines.append(
            f"| {row['N_low']}->{row['N_high']} | {row['sup_discriminant_difference']:.3e} | "
            f"{row['minimum_base_modulus']:.3e} | {row['conservative_ratio']:.3e} | "
            f"{'yes' if row['rouche_sufficient_pass'] else 'no'} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "A stable winding, pole exchange, and half-winding show that the EP topology persists",
        "over the computed finite chains. A pairwise Rouché pass additionally prevents the",
        "enclosed zero count from changing between that finite-N pair. Infinite-chain or",
        "thermodynamic claims require a separately controlled tail/chain-length limit.",
        "",
    ]
    (out / "FULLK_N_SCALING_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "analysis": str(out)}, indent=2))


if __name__ == "__main__":
    main()
