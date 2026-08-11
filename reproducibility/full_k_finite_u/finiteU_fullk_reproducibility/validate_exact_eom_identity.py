#!/usr/bin/env python3
"""Validate the exact finite-chain equation-of-motion self-energy identity.

For every z-discretized Hermitian block Wilson chain, compare

    Sigma_U = U F G^{-1}

against the independent Dyson construction

    Sigma_U = (G_0^imp)^{-1} - G^{-1}.

The comparison is pointwise in complex frequency and is performed before any
z averaging, where both objects are exact for the stated finite chain.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import exact_zavg_block_chain_response as zavg
import exact_block_chain_response as exact


def encoded(matrix: np.ndarray) -> list[list[list[float]]]:
    return [
        [[[float(value.real), float(value.imag)] for value in row] for row in matrix]
    ][0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mapping-summary", type=Path, required=True)
    parser.add_argument("--center-summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interaction", type=float, required=True)
    parser.add_argument("--sites", type=int, default=12)
    parser.add_argument("--reference-hybridization", type=float, default=0.10)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    center = json.loads(args.center_summary.read_text())["predicted_center"]
    epsilon = float(center["epsilon_d"])
    width = float(center["W"])
    z0 = complex(*center["z"])
    points = [z0, z0 + 2.0e-4, z0 - 2.0e-4j, z0 + 3.0e-4 + 1.0e-4j]

    chains, _, _ = zavg.read_chain(args.manifest, args.mapping_summary)
    state, _ = zavg.run_nrg(
        chains,
        None,
        None,
        epsilon,
        width,
        args.reference_hybridization,
        args.interaction,
        args.sites,
        0,
        0,
    )

    rows = []
    maximum_absolute = 0.0
    maximum_relative = 0.0
    maximum_sumrule = 0.0
    for z_index, member in enumerate(state.states):
        for point_index, point in enumerate(points):
            green, composite, sumrule = exact.response(member, point)
            sigma_eom = args.interaction * composite @ np.linalg.inv(green)
            green0 = exact.noninteracting_green(member, point)
            sigma_dyson = np.linalg.inv(green0) - np.linalg.inv(green)
            absolute = float(np.linalg.norm(sigma_eom - sigma_dyson))
            scale = max(float(np.linalg.norm(sigma_eom)), float(np.linalg.norm(sigma_dyson)), 1.0e-30)
            relative = absolute / scale
            sumrule_error = float(np.linalg.norm(sumrule - np.eye(2)))
            maximum_absolute = max(maximum_absolute, absolute)
            maximum_relative = max(maximum_relative, relative)
            maximum_sumrule = max(maximum_sumrule, sumrule_error)
            rows.append({
                "z_index": z_index,
                "point_index": point_index,
                "frequency": [float(point.real), float(point.imag)],
                "absolute_residual": absolute,
                "relative_residual": relative,
                "sumrule_error": sumrule_error,
                "Sigma_EOM": encoded(sigma_eom),
                "Sigma_Dyson": encoded(sigma_dyson),
            })

    passed = bool(maximum_absolute < 1.0e-11 and maximum_relative < 1.0e-7 and maximum_sumrule < 1.0e-11)
    result = {
        "status": "PASS" if passed else "FAIL",
        "scope": "Exact pointwise EOM-versus-Dyson identity on each finite z-discretized Hermitian block chain, before z averaging.",
        "parameters": {
            "U": args.interaction,
            "sites": args.sites,
            "epsilon_d": epsilon,
            "W": width,
            "z_count": len(state.states),
        },
        "maximum_absolute_residual": maximum_absolute,
        "maximum_relative_residual": maximum_relative,
        "maximum_sumrule_error": maximum_sumrule,
        "checks": rows,
    }
    output = args.out / "exact_eom_identity_summary.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "checks"}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
