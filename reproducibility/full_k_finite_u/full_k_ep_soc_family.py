#!/usr/bin/env python3
"""Continue and certify a finite SOC family of full-k nonlinear pole EPs.

The calculation imports the exact two-cut retarded kernel from
``full_k_two_cut_ep_certificate.py``.  It continues the real solution of
F=F_z=0 in the SOC parameter while tuning the real impurity gate and the
off-shell auxiliary vertex.  Three representative points are then certified
independently by local-discriminant winding and explicit pole exchange.

This is a one-dimensional locus in the enlarged parameter space, not an EP
phase region.  The high-SOC endpoint is not classified here because the
continued pole approaches an open-channel threshold.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ptdirac_soc_family_matplotlib")

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root, root_scalar

from full_k_two_cut_ep_certificate import (
    I2,
    P,
    analytic_physical_self_energy,
    auxiliary_terms,
    local_certificate,
    reduced_data,
    reduced_residual,
    solve_ep,
)


def solve_at_soc(soc: float, initial: np.ndarray) -> tuple[object, np.ndarray]:
    """Solve the reduced double-zero equations at fixed SOC."""

    p = replace(P, soc=float(soc))
    result = root(
        lambda coordinates: reduced_residual(coordinates, p, p.derivative_step),
        initial,
        tol=1.0e-12,
    )
    residual = float(
        np.linalg.norm(reduced_residual(result.x, p, p.derivative_step))
    )
    if (not result.success and residual > 2.0e-10) or residual > 2.0e-9:
        raise RuntimeError(
            f"SOC continuation failed at lambda={soc:.8f}: "
            f"success={result.success}, residual={residual:.3e}"
        )
    return p, np.asarray(result.x, dtype=float)


def upper_threshold(auxiliary_vertex: float, p) -> float:
    """Upper k=0 threshold of the driven two-channel bath."""

    def equation(energy: float) -> float:
        even, odd = auxiliary_terms(energy, auxiliary_vertex, p)
        return float(energy - even.real - abs(odd.real))

    grid = np.linspace(-0.08, 0.08, 2001)
    values = np.asarray([equation(value) for value in grid])
    changes = np.flatnonzero(values[:-1] * values[1:] <= 0.0)
    positive_roots: list[float] = []
    for index in changes:
        result = root_scalar(
            equation,
            bracket=(float(grid[index]), float(grid[index + 1])),
            method="brentq",
            xtol=1.0e-14,
        )
        if result.root > 0.0:
            positive_roots.append(float(result.root))
    if not positive_roots:
        raise RuntimeError("No positive driven k=0 threshold was found")
    return min(positive_roots)


def row_from_solution(soc: float, coordinates: np.ndarray, p) -> dict[str, float]:
    data = reduced_data(coordinates, p, p.derivative_step)
    z = complex(data["z"])
    epsilon = complex(data["epsilon"])
    vertex = float(data["auxiliary_vertex"])
    upper = analytic_physical_self_energy(z.real + 1.0e-8j, vertex, p)
    rate = (upper.conj().T - upper) / (2.0j)
    rate_eigenvalues = np.linalg.eigvalsh(rate)
    mean_rate = float(np.mean(rate_eigenvalues))
    threshold = upper_threshold(vertex, p)
    return {
        "soc": float(soc),
        "z_re": float(z.real),
        "z_im": float(z.imag),
        "epsilon_d": float(epsilon.real),
        "auxiliary_vertex": vertex,
        "off_shell_ratio": vertex / (p.auxiliary_energy - p.drive_frequency),
        "rate_min": float(np.min(rate_eigenvalues)),
        "rate_max": float(np.max(rate_eigenvalues)),
        "mean_rate": mean_rate,
        "passivity_margin": float(np.min(rate_eigenvalues) / mean_rate),
        "upper_threshold": threshold,
        "threshold_gap": float(z.real - threshold),
        "threshold_gap_in_widths": float((z.real - threshold) / abs(z.imag)),
        "reduced_residual_norm": float(
            np.linalg.norm(reduced_residual(coordinates, p, p.derivative_step))
        ),
    }


def continuation() -> list[dict[str, float]]:
    """Continue from the already-certified lambda=0.10 solution."""

    reference = solve_ep(P)
    initial = np.asarray(reference["coordinates"], dtype=float)
    rows: dict[float, dict[str, float]] = {}

    downward = np.linspace(0.10, 0.02, 41)
    current = initial.copy()
    for soc in downward:
        p, current = solve_at_soc(float(soc), current)
        rows[round(float(soc), 10)] = row_from_solution(float(soc), current, p)

    upward = np.linspace(0.10, 0.1578, 59)
    current = initial.copy()
    for soc in upward:
        p, current = solve_at_soc(float(soc), current)
        rows[round(float(soc), 10)] = row_from_solution(float(soc), current, p)

    return [rows[key] for key in sorted(rows)]


def representative_certificates(rows: list[dict[str, float]]) -> list[dict[str, object]]:
    answer: list[dict[str, object]] = []
    for soc, contour_factor in ((0.05, 1.0), (0.10, 1.0), (0.15, 0.5)):
        nearest = min(rows, key=lambda row: abs(row["soc"] - soc))
        initial = np.asarray(
            [nearest["z_re"], nearest["z_im"], nearest["auxiliary_vertex"]],
            dtype=float,
        )
        p, solution = solve_at_soc(soc, initial)
        data = reduced_data(solution, p, p.derivative_step)
        ep = {
            "z": [float(complex(data["z"]).real), float(complex(data["z"]).imag)],
            "epsilon": [float(complex(data["epsilon"]).real), float(complex(data["epsilon"]).imag)],
            "auxiliary_vertex": float(data["auxiliary_vertex"]),
        }
        contour_parameters = replace(
            p,
            gate_halfwidth=contour_factor * p.gate_halfwidth,
            auxiliary_vertex_halfwidth=contour_factor * p.auxiliary_vertex_halfwidth,
        )
        topology = local_certificate(ep, contour_parameters)
        if not topology["certified"]:
            raise RuntimeError(f"Topology certificate failed at lambda={soc:.2f}")
        physical = row_from_solution(soc, solution, p)
        answer.append(
            {
                "soc": soc,
                "contour_factor": contour_factor,
                "z": ep["z"],
                "epsilon_d": ep["epsilon"][0],
                "auxiliary_vertex": ep["auxiliary_vertex"],
                "winding": topology["winding"],
                "endpoint_permutation": topology["endpoint_permutation"],
                "half_winding": topology["half_winding"],
                "minimum_boundary_modulus": topology["minimum_boundary_modulus"],
                "minimum_F_zz_modulus": topology["minimum_F_zz_modulus"],
                "minimum_pole_separation": topology["minimum_pole_separation"],
                "maximum_assignment_to_half_separation_ratio": topology[
                    "maximum_assignment_to_half_separation_ratio"
                ],
                "pole_failures": topology["pole_failures"],
                "passivity_margin": physical["passivity_margin"],
                "threshold_gap_in_widths": physical[
                    "threshold_gap_in_widths"
                ],
            }
        )
    return answer


def write_outputs(
    output: Path,
    rows: list[dict[str, float]],
    certificates: list[dict[str, object]],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with (output / "full_k_ep_soc_family.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "status": "PASS",
        "continued_interval": [rows[0]["soc"], rows[-1]["soc"]],
        "continued_point_count": len(rows),
        "representative_topological_certificates": certificates,
        "scope": (
            "A one-dimensional full-k nonlinear-EP locus in the enlarged "
            "parameter space. It is not an EP phase region. The high-SOC "
            "endpoint is not classified because the branch approaches an "
            "open-channel threshold."
        ),
    }
    (output / "full_k_ep_soc_family_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    soc = np.asarray([row["soc"] for row in rows])
    z_re = np.asarray([row["z_re"] for row in rows])
    epsilon = np.asarray([row["epsilon_d"] for row in rows])
    off_shell = np.asarray([row["off_shell_ratio"] for row in rows])
    passivity = np.asarray([row["passivity_margin"] for row in rows])
    threshold_distance = np.asarray(
        [row["threshold_gap_in_widths"] for row in rows]
    )
    certified_soc = np.asarray([item["soc"] for item in certificates])

    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.2), constrained_layout=True)
    axes[0, 0].plot(soc, z_re, lw=1.8, label=r"$\operatorname{Re}z_{\rm EP}$")
    axes[0, 0].plot(soc, epsilon, lw=1.8, label=r"$\epsilon_d$")
    axes[0, 0].set_ylabel("energy")
    axes[0, 0].legend(frameon=False, fontsize=8)

    axes[0, 1].plot(soc, off_shell, lw=1.8)
    axes[0, 1].set_ylabel(r"$V_O/(\epsilon_O-\Omega)$")

    axes[1, 0].plot(soc, passivity, lw=1.8)
    axes[1, 0].axhline(0.0, color="black", lw=0.8)
    axes[1, 0].set_ylabel(r"$\Gamma_{\min}/\bar\Gamma$")
    axes[1, 0].set_xlabel(r"SOC $\lambda$")

    axes[1, 1].plot(soc, threshold_distance, lw=1.8)
    axes[1, 1].axhline(0.0, color="black", lw=0.8)
    axes[1, 1].set_ylabel(
        r"$(\operatorname{Re}z_{\rm EP}-\omega_{\rm th}^{+})/|\operatorname{Im}z_{\rm EP}|$"
    )
    axes[1, 1].set_xlabel(r"SOC $\lambda$")

    plotted_series = (z_re, off_shell, passivity, threshold_distance)
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        for value in certified_soc:
            axis.axvline(value, color="black", lw=0.6, alpha=0.22)
    for axis, series in zip(axes.flat, plotted_series):
        indices = [int(np.argmin(abs(soc - value))) for value in certified_soc]
        axis.scatter(
            soc[indices],
            series[indices],
            s=24,
            facecolor="white",
            edgecolor="black",
            linewidth=0.9,
            zorder=4,
        )
    epsilon_indices = [
        int(np.argmin(abs(soc - value))) for value in certified_soc
    ]
    axes[0, 0].scatter(
        soc[epsilon_indices],
        epsilon[epsilon_indices],
        s=24,
        facecolor="white",
        edgecolor="black",
        linewidth=0.9,
        zorder=4,
    )
    axes[0, 0].set_title("Continued double-zero locus")
    axes[0, 1].set_title("Off-shell control")
    axes[1, 0].set_title("Passive physical sheet")
    axes[1, 1].set_title("Approach to upper threshold")

    fig.savefig(output / "SFig10_full_k_soc_family.pdf")
    fig.savefig(output / "SFig10_full_k_soc_family.png", dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("soc_family_output"))
    arguments = parser.parse_args()
    rows = continuation()
    certificates = representative_certificates(rows)
    write_outputs(arguments.out.resolve(), rows, certificates)
    print(
        json.dumps(
            {
                "status": "PASS",
                "continued_interval": [rows[0]["soc"], rows[-1]["soc"]],
                "continued_points": len(rows),
                "certified_soc_values": [item["soc"] for item in certificates],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
