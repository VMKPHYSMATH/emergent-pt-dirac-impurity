#!/usr/bin/env python3
"""Pointwise finite-U kept-space NRG contour around the causal EP predictor.

Every contour point is solved by a fresh block-chain NRG run, so the matrix
interaction self-energy is allowed to depend on both epsilon_d and W.  The
critical pole z_c is branch-tracked from F_z=0 and the winding of F(z_c) is
computed.  This is the correct topology test within the kept-space pilot, but
it remains non-publication evidence until complete-basis and discretization
convergence are supplied.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment, root


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nrg-source", type=Path, required=True)
    parser.add_argument("--certificate-source", type=Path, required=True)
    parser.add_argument("--chain", type=Path, required=True)
    parser.add_argument("--mapping-summary", type=Path, required=True)
    parser.add_argument("--center-summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interaction", type=float, required=True)
    parser.add_argument("--auxiliary-vertex", type=float, default=1.2134220096136235)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--gate-halfwidth", type=float, default=1.0e-4)
    parser.add_argument("--hybridization-halfwidth", type=float, default=1.0e-3)
    parser.add_argument("--sites", type=int, default=11)
    parser.add_argument("--cluster-sites", type=int, default=2)
    parser.add_argument("--nkeep", type=int, default=320)
    parser.add_argument("--derivative-step", type=float, default=2.0e-6)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    nrg = load(args.nrg_source, "python_fullk_nrg_contour_module")
    fullk = load(args.certificate_source, "fullk_certificate_contour_module")
    onsite, hopping, impurity_chain = nrg.read_chain(args.chain, args.mapping_summary)
    center_data = json.loads(args.center_summary.read_text())
    center = center_data["predicted_center"]
    z_center = complex(*center["z"])
    epsilon_center = float(center["epsilon_d"])
    width_center = float(center["W"])
    p0 = replace(fullk.P, soc=0.05)

    def derivative(function, value: complex) -> complex:
        h = args.derivative_step
        return complex(
            (
                function(value - 2.0 * h)
                - 8.0 * function(value - h)
                + 8.0 * function(value + h)
                - function(value + 2.0 * h)
            ) / (12.0 * h)
        )

    theta = np.linspace(0.0, 2.0 * math.pi, args.samples + 1)
    current = z_center
    rows = []
    discriminants = []
    tracked_poles: list[np.ndarray] = []
    all_success = True
    maximum_assignment_ratio = 0.0
    maximum_pole_residual = 0.0
    for index, angle in enumerate(theta):
        epsilon = epsilon_center + args.gate_halfwidth * math.cos(angle)
        width = width_center + args.hybridization_halfwidth * math.sin(angle)
        state, flow = nrg.run_nrg(
            onsite,
            hopping,
            impurity_chain,
            epsilon,
            width,
            0.10,
            args.interaction,
            args.sites,
            args.cluster_sites,
            args.nkeep,
        )

        def sigma(value: complex) -> np.ndarray:
            green, composite, _ = nrg.response(state, value)
            return args.interaction * composite @ np.linalg.inv(green)

        def determinant(value: complex) -> complex:
            p = replace(p0, impurity_hybridization=width)
            kernel = (
                (value - epsilon) * fullk.I2
                - fullk.resonance_self_energy(value, args.auxiliary_vertex, p)
                - sigma(value)
            )
            return complex(np.linalg.det(kernel))

        def first(value: complex) -> complex:
            return derivative(determinant, value)

        answer = root(
            lambda coordinates: [
                first(complex(*coordinates)).real,
                first(complex(*coordinates)).imag,
            ],
            [current.real, current.imag],
            tol=1.0e-10,
        )
        all_success = all_success and bool(answer.success)
        current = complex(*answer.x)
        value = determinant(current)
        curvature = derivative(first, current)
        _, _, sumrule = nrg.response(state, current)
        discriminants.append(value)

        split = np.sqrt(-2.0 * value / curvature)
        poles = []
        for guess in (current + split, current - split):
            pole_answer = root(
                lambda coordinates: [
                    determinant(complex(*coordinates)).real,
                    determinant(complex(*coordinates)).imag,
                ],
                [guess.real, guess.imag],
                tol=1.0e-10,
            )
            all_success = all_success and bool(pole_answer.success)
            pole = complex(*pole_answer.x)
            poles.append(pole)
            maximum_pole_residual = max(maximum_pole_residual, abs(determinant(pole)))
        poles_array = np.asarray(poles)
        if tracked_poles:
            cost = abs(tracked_poles[-1][:, None] - poles_array[None, :])
            assignment_rows, assignment_columns = linear_sum_assignment(cost)
            ordered = np.empty(2, dtype=complex)
            ordered[assignment_rows] = poles_array[assignment_columns]
            half_separation = 0.5 * min(
                abs(tracked_poles[-1][1] - tracked_poles[-1][0]),
                abs(ordered[1] - ordered[0]),
            )
            maximum_assignment_ratio = max(
                maximum_assignment_ratio,
                float(np.max(cost[assignment_rows, assignment_columns])) / half_separation,
            )
        else:
            ordered = poles_array
        tracked_poles.append(ordered)
        rows.append({
            "index": index,
            "theta": float(angle),
            "epsilon_d": epsilon,
            "W": width,
            "z_c": [current.real, current.imag],
            "D": [value.real, value.imag],
            "F_z_modulus": float(abs(first(current))),
            "F_zz_modulus": float(abs(curvature)),
            "sumrule_error": float(np.linalg.norm(sumrule - np.eye(2))),
            "ground_charge": int(flow[-1]["ground_charge"]),
            "poles": [[pole.real, pole.imag] for pole in ordered],
        })
        print(
            f"{index:03d}/{args.samples:03d} theta={angle:.6f} "
            f"|D|={abs(value):.4e} z={current.real:.7f}{current.imag:+.7f}i "
            f"sum={rows[-1]['sumrule_error']:.3e}",
            flush=True,
        )

    disc = np.asarray(discriminants)
    phase_steps = np.angle(disc[1:] / disc[:-1])
    winding = float(np.sum(phase_steps) / (2.0 * math.pi))
    tracked = np.asarray(tracked_poles)
    assignment_rows, assignment_columns = linear_sum_assignment(
        abs(tracked[-1][:, None] - tracked[0][None, :])
    )
    permutation = np.empty(2, dtype=int)
    permutation[assignment_rows] = assignment_columns
    separation = tracked[:, 1] - tracked[:, 0]
    half_winding = float(
        np.sum(np.angle(separation[1:] / separation[:-1])) / (2.0 * math.pi)
    )
    complete_basis = hasattr(nrg, "ExactState") or hasattr(nrg, "ZAverageState")
    maximum_sumrule_error = float(max(row["sumrule_error"] for row in rows))
    ground_charges = sorted(set(row["ground_charge"] for row in rows))
    pilot_pass = bool(
        all_success
        and abs(abs(winding) - 1.0) < 0.05
        and permutation.tolist() == [1, 0]
        and abs(abs(half_winding) - 0.5) < 0.05
        and np.min(abs(disc)) > 1.0e-12
        and np.max(abs(phase_steps)) < math.pi / 2.0
        and min(row["F_zz_modulus"] for row in rows) > 1.0e-4
        and max(row["F_z_modulus"] for row in rows) < 1.0e-7
        and maximum_pole_residual < 1.0e-7
        and maximum_assignment_ratio < 0.5
        and min(ground_charges) >= 0
        and (not complete_basis or maximum_sumrule_error < 1.0e-10)
    )
    result = {
        "status": (
            "COMPLETE_BASIS_TOPOLOGY_PASS" if pilot_pass else "COMPLETE_BASIS_TOPOLOGY_FAIL"
        ) if complete_basis else (
            "PILOT_TOPOLOGY_PASS" if pilot_pass else "PILOT_TOPOLOGY_FAIL"
        ),
        "scope": (
            "Pointwise complete-basis block-Wilson topology certificate. A "
            "manuscript claim additionally requires Wilson Lambda, z-average, "
            "chain-length, interaction-interval, and contour convergence."
            if complete_basis else
            "Pointwise kept-space NRG topology pilot. A publication-level "
            "certificate requires complete-basis response and convergence in "
            "nkeep, Wilson Lambda, z averaging, sites, and contour sampling."
        ),
        "parameters": {
            "U": args.interaction,
            "nkeep": args.nkeep,
            "sites": args.sites,
            "samples": args.samples,
            "center": center,
            "gate_halfwidth": args.gate_halfwidth,
            "hybridization_halfwidth": args.hybridization_halfwidth,
        },
        "certificate": {
            "winding": winding,
            "minimum_boundary_modulus": float(np.min(abs(disc))),
            "maximum_phase_increment": float(np.max(abs(phase_steps))),
            "minimum_F_zz_modulus": float(min(row["F_zz_modulus"] for row in rows)),
            "maximum_F_z_residual": float(max(row["F_z_modulus"] for row in rows)),
            "endpoint_permutation": permutation.tolist(),
            "half_winding": half_winding,
            "minimum_pole_separation": float(np.min(abs(separation))),
            "maximum_pole_residual": float(maximum_pole_residual),
            "maximum_assignment_to_half_separation_ratio": float(maximum_assignment_ratio),
            "maximum_sumrule_error": maximum_sumrule_error,
            "ground_charges": ground_charges,
        },
        "contour": rows,
    }
    (args.out / "fullk_nrg_contour_pilot_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps({"status": result["status"], "certificate": result["certificate"]}, indent=2))


if __name__ == "__main__":
    main()
