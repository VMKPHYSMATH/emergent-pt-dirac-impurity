#!/usr/bin/env python3
"""Self-consistent kept-space NRG predictor for the finite-U causal EP.

At each outer iteration an NRG run is performed at the current (epsilon_d,W).
Its equation-of-motion interaction self-energy is held fixed while the four
real equations F=F_z=0 are solved for a new (z,epsilon_d,W).  Repeating this
map supplies an economical center for a later point-by-point topological
contour.  The result remains a pilot until complete-basis and contour gates
pass.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import root


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def encode_matrix(matrix: np.ndarray):
    return [[[float(v.real), float(v.imag)] for v in row] for row in matrix]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nrg-source", type=Path, required=True)
    parser.add_argument("--certificate-source", type=Path, required=True)
    parser.add_argument("--chain", type=Path, required=True)
    parser.add_argument("--mapping-summary", type=Path, required=True)
    parser.add_argument("--ep-summary", type=Path, required=True)
    parser.add_argument("--initial-center-summary", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interaction", type=float, required=True)
    parser.add_argument("--sites", type=int, default=11)
    parser.add_argument("--cluster-sites", type=int, default=2)
    parser.add_argument("--nkeep", type=int, default=640)
    parser.add_argument("--outer-iterations", type=int, default=4)
    parser.add_argument("--derivative-step", type=float, default=2.0e-6)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    nrg = load(args.nrg_source, "python_fullk_nrg_pilot_module")
    fullk = load(args.certificate_source, "fullk_certificate_predictor_module")
    onsite, hopping, impurity_chain = nrg.read_chain(args.chain, args.mapping_summary)
    archived = json.loads(args.ep_summary.read_text())
    z = complex(*archived["ep"]["z"])
    epsilon = float(archived["ep"]["epsilon"][0])
    hybridization = 0.10
    if args.initial_center_summary is not None:
        initial = json.loads(args.initial_center_summary.read_text())["predicted_center"]
        z = complex(*initial["z"])
        epsilon = float(initial["epsilon_d"])
        hybridization = float(initial["W"])
    auxiliary_vertex = float(archived["ep"]["auxiliary_vertex"])
    p0 = replace(fullk.P, soc=0.05, impurity_hybridization=hybridization)

    def derivative(function, value: complex) -> complex:
        h = args.derivative_step
        return complex(
            (
                function(value - 2.0 * h)
                - 8.0 * function(value - h)
                + 8.0 * function(value + h)
                - function(value + 2.0 * h)
            )
            / (12.0 * h)
        )

    history = []
    final_state = None
    for outer in range(args.outer_iterations):
        state, flow = nrg.run_nrg(
            onsite,
            hopping,
            impurity_chain,
            epsilon,
            hybridization,
            0.10,
            args.interaction,
            args.sites,
            args.cluster_sites,
            args.nkeep,
        )
        final_state = state

        def sigma(value: complex) -> np.ndarray:
            green, composite, _ = nrg.response(state, value)
            return args.interaction * composite @ np.linalg.inv(green)

        def determinant(value: complex, gate: float, width: float) -> complex:
            p = replace(p0, impurity_hybridization=width)
            kernel = (
                (value - gate) * fullk.I2
                - fullk.resonance_self_energy(value, auxiliary_vertex, p)
                - sigma(value)
            )
            return complex(np.linalg.det(kernel))

        answer = root(
            lambda coordinates: [
                determinant(complex(coordinates[0], coordinates[1]), coordinates[2], coordinates[3]).real,
                determinant(complex(coordinates[0], coordinates[1]), coordinates[2], coordinates[3]).imag,
                derivative(
                    lambda value: determinant(value, coordinates[2], coordinates[3]),
                    complex(coordinates[0], coordinates[1]),
                ).real,
                derivative(
                    lambda value: determinant(value, coordinates[2], coordinates[3]),
                    complex(coordinates[0], coordinates[1]),
                ).imag,
            ],
            [z.real, z.imag, epsilon, hybridization],
            tol=1.0e-10,
        )
        new_z = complex(answer.x[0], answer.x[1])
        new_epsilon = float(answer.x[2])
        new_hybridization = float(answer.x[3])
        f_value = determinant(new_z, new_epsilon, new_hybridization)
        f_z = derivative(lambda value: determinant(value, new_epsilon, new_hybridization), new_z)
        f_zz = derivative(
            lambda value: derivative(
                lambda inner: determinant(inner, new_epsilon, new_hybridization), value
            ),
            new_z,
        )
        green, _, sumrule = nrg.response(state, new_z)
        sigma_value = sigma(new_z)
        p = replace(p0, impurity_hybridization=new_hybridization)
        kernel = (
            (new_z - new_epsilon) * fullk.I2
            - fullk.resonance_self_energy(new_z, auxiliary_vertex, p)
            - sigma_value
        )
        singular = np.linalg.svd(kernel, compute_uv=False)
        displacement = max(
            abs(new_z - z),
            abs(new_epsilon - epsilon),
            abs(new_hybridization - hybridization),
        )
        history.append({
            "outer_iteration": outer,
            "solver_success": bool(answer.success),
            "z": [new_z.real, new_z.imag],
            "epsilon_d": new_epsilon,
            "W": new_hybridization,
            "displacement": float(displacement),
            "F_modulus": float(abs(f_value)),
            "F_z_modulus": float(abs(f_z)),
            "F_zz": [f_zz.real, f_zz.imag],
            "kernel_singular_values": singular.tolist(),
            "sumrule_error": float(np.linalg.norm(sumrule - np.eye(2))),
            "Sigma_U": encode_matrix(sigma_value),
            "flow_last": flow[-1],
        })
        z, epsilon, hybridization = new_z, new_epsilon, new_hybridization

    converged = bool(
        history
        and history[-1]["solver_success"]
        and history[-1]["displacement"] < 5.0e-6
        and history[-1]["F_modulus"] < 1.0e-8
        and history[-1]["F_z_modulus"] < 1.0e-7
        and abs(complex(*history[-1]["F_zz"])) > 1.0e-4
        and history[-1]["kernel_singular_values"][1] < 1.0e-7
    )
    complete_basis = hasattr(nrg, "ExactState") or hasattr(nrg, "ZAverageState")
    result = {
        "status": "CENTER_PASS" if converged and complete_basis else (
            "PILOT_PASS" if converged else "CENTER_NOT_CONVERGED"
        ),
        "scope": (
            "Self-consistent complete-basis block-Wilson center predictor; a "
            "manuscript claim additionally requires pointwise topology and "
            "discretization/length convergence."
            if complete_basis else
            "Self-consistent kept-space NRG center predictor. This is not a "
            "topological certificate and is not manuscript evidence until the "
            "complete-basis, pointwise-contour, and discretization gates pass."
        ),
        "parameters": {
            "U": args.interaction,
            "sites": args.sites,
            "cluster_sites": args.cluster_sites,
            "nkeep": args.nkeep,
            "auxiliary_vertex": auxiliary_vertex,
            "soc": 0.05,
        },
        "history": history,
        "predicted_center": {
            "z": [z.real, z.imag],
            "epsilon_d": epsilon,
            "W": hybridization,
        },
    }
    (args.out / "fullk_nrg_ep_predictor_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
