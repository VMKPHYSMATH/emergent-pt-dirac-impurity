#!/usr/bin/env python3
"""Direct kept-space versus exact response calibration at a fixed EP center."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import exact_zavg_block_chain_response as exact_zavg
import kept_zavg_block_chain_response as kept_zavg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mapping-summary", type=Path, required=True)
    parser.add_argument("--center-summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interaction", type=float, required=True)
    parser.add_argument("--sites", type=int, default=12)
    parser.add_argument("--cluster-sites", type=int, default=2)
    parser.add_argument("--nkeep", type=int, nargs="+", default=[320, 640, 960])
    parser.add_argument("--reference-hybridization", type=float, default=0.10)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    center = json.loads(args.center_summary.read_text())["predicted_center"]
    epsilon = float(center["epsilon_d"])
    width = float(center["W"])
    z0 = complex(*center["z"])
    points = [z0, z0 + 2.0e-4, z0 - 2.0e-4j]

    exact_chains, _, _ = exact_zavg.read_chain(args.manifest, args.mapping_summary)
    exact_state, exact_flow = exact_zavg.run_nrg(
        exact_chains, None, None, epsilon, width, args.reference_hybridization,
        args.interaction, args.sites, 0, 0,
    )
    references = []
    for point in points:
        green, composite, sumrule = exact_zavg.response(exact_state, point)
        references.append((
            green,
            composite,
            args.interaction * composite @ np.linalg.inv(green),
            sumrule,
        ))

    kept_chains, _, _ = kept_zavg.read_chain(args.manifest, args.mapping_summary)
    reports = []
    for nkeep in args.nkeep:
        state, flow = kept_zavg.run_nrg(
            kept_chains, None, None, epsilon, width, args.reference_hybridization,
            args.interaction, args.sites, args.cluster_sites, nkeep,
        )
        rows = []
        maxima = {"G": 0.0, "F": 0.0, "Sigma": 0.0, "sumrule": 0.0}
        for point, reference in zip(points, references, strict=True):
            green, composite, sumrule = kept_zavg.response(state, point)
            sigma = args.interaction * composite @ np.linalg.inv(green)
            errors = {
                "G": float(np.linalg.norm(green - reference[0]) / np.linalg.norm(reference[0])),
                "F": float(np.linalg.norm(composite - reference[1]) / max(np.linalg.norm(reference[1]), 1.0e-30)),
                "Sigma": float(np.linalg.norm(sigma - reference[2]) / max(np.linalg.norm(reference[2]), 1.0e-30)),
                "sumrule": float(np.linalg.norm(sumrule - np.eye(2))),
            }
            for key, value in errors.items():
                maxima[key] = max(maxima[key], value)
            rows.append({"z": [point.real, point.imag], **errors})
        reports.append({
            "nkeep": nkeep,
            "maximum_errors": maxima,
            "z_ground_charges": flow[-1]["z_ground_charges"],
            "z_cutoffs": flow[-1]["z_cutoffs"],
            "points": rows,
        })

    result = {
        "status": "CALIBRATION_COMPLETE",
        "scope": "Direct kept-space truncation calibration against the exact complete-basis response on the same finite four-z block Wilson chains.",
        "parameters": {
            "U": args.interaction,
            "sites": args.sites,
            "epsilon_d": epsilon,
            "W": width,
            "z": [z0.real, z0.imag],
            "exact_z_ground_charges": exact_flow[-1]["z_ground_charges"],
        },
        "reports": reports,
    }
    path = args.out / "kept_against_exact_summary.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
