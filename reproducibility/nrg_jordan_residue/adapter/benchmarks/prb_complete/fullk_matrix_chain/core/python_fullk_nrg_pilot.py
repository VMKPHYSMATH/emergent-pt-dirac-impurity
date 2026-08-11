#!/usr/bin/env python3
"""Hermitian block-chain NRG pilot for the full-k rotating-frame bath.

This solver is intentionally conservative.  It consumes the matrix Wilson
chain produced by ``full_k_wilson_preflight.py`` and performs a charge-only,
Hermitian iterative diagonalization of the finite-U two-flavour Anderson
impurity.  It propagates both the impurity fermion and the equation-of-motion
composite operator n_{bar a} d_a, allowing the matrix interaction self-energy

    Sigma_U(z) = U F(z) G(z)^{-1}

to be evaluated in the final kept space.

The kept-space response is a *pilot*, not a complete-basis DM-NRG spectrum.
It is used to establish numerical feasibility and initial finite-U shifts; a
publication claim requires the convergence and complete-basis gates described
in the generated summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


NFLAV = 2
LOCAL_DIM = 4


def annihilation_operator(mode: int) -> np.ndarray:
    result = np.zeros((LOCAL_DIM, LOCAL_DIM), dtype=complex)
    for ket in range(LOCAL_DIM):
        bit = 1 << mode
        if ket & bit:
            bra = ket ^ bit
            parity = (ket & (bit - 1)).bit_count()
            result[bra, ket] = 1.0 if parity % 2 == 0 else -1.0
    return result


C = (annihilation_operator(0), annihilation_operator(1))
CDAG = tuple(operator.conj().T for operator in C)
N = tuple(CDAG[index] @ C[index] for index in range(NFLAV))
ILOCAL = np.eye(LOCAL_DIM, dtype=complex)
QLOCAL = np.asarray([value.bit_count() for value in range(LOCAL_DIM)], dtype=int)


def parity(charges: np.ndarray) -> np.ndarray:
    return np.diag(np.where(charges % 2 == 0, 1.0, -1.0)).astype(complex)


def site_hamiltonian(matrix: np.ndarray) -> np.ndarray:
    answer = np.zeros((LOCAL_DIM, LOCAL_DIM), dtype=complex)
    for a in range(NFLAV):
        for b in range(NFLAV):
            answer += matrix[a, b] * CDAG[a] @ C[b]
    return 0.5 * (answer + answer.conj().T)


def impurity_hamiltonian(epsilon: float, interaction: float) -> np.ndarray:
    return epsilon * (N[0] + N[1]) + interaction * (N[0] @ N[1])


def coupling_term(
    old_ann: tuple[np.ndarray, np.ndarray],
    old_cre: tuple[np.ndarray, np.ndarray],
    old_charges: np.ndarray,
    coupling: np.ndarray,
) -> np.ndarray:
    pold = parity(old_charges)
    dimension = len(old_charges) * LOCAL_DIM
    answer = np.zeros((dimension, dimension), dtype=complex)
    for a in range(NFLAV):
        for b in range(NFLAV):
            value = coupling[a, b]
            if value != 0.0:
                answer += value * np.kron(old_cre[a] @ pold, C[b])
                answer += value.conjugate() * np.kron(pold @ old_ann[a], CDAG[b])
    return answer


@dataclass
class State:
    iteration: int
    energies: np.ndarray
    charges: np.ndarray
    last_ann: tuple[np.ndarray, np.ndarray]
    last_cre: tuple[np.ndarray, np.ndarray]
    impurity_ann: tuple[np.ndarray, np.ndarray]
    impurity_cre: tuple[np.ndarray, np.ndarray]
    composite_ann: tuple[np.ndarray, np.ndarray]
    composite_cre: tuple[np.ndarray, np.ndarray]
    truncation_cutoff: float


def diagonalize_by_charge(
    hamiltonian: np.ndarray,
    charges: np.ndarray,
    nkeep: int | None,
    minimum_per_charge: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    dimension = len(charges)
    records: list[tuple[float, int, np.ndarray]] = []
    for charge in np.unique(charges):
        indices = np.flatnonzero(charges == charge)
        block = hamiltonian[np.ix_(indices, indices)]
        values, vectors = np.linalg.eigh(0.5 * (block + block.conj().T))
        for column, value in enumerate(values):
            vector = np.zeros(dimension, dtype=complex)
            vector[indices] = vectors[:, column]
            records.append((float(value), int(charge), vector))
    records.sort(key=lambda item: item[0])
    if nkeep is None or nkeep >= len(records):
        selected = list(range(len(records)))
    else:
        mandatory: set[int] = set()
        for charge in np.unique(charges):
            local = [index for index, item in enumerate(records) if item[1] == charge]
            mandatory.update(local[:minimum_per_charge])
        selected = sorted(mandatory)
        for index in range(len(records)):
            if len(selected) >= nkeep:
                break
            if index not in mandatory:
                selected.append(index)
        selected.sort(key=lambda index: records[index][0])
    reference = records[selected[0]][0]
    values = np.asarray([records[index][0] - reference for index in selected])
    kept_charges = np.asarray([records[index][1] for index in selected], dtype=int)
    rotation = np.column_stack([records[index][2] for index in selected])
    cutoff = float(records[selected[-1]][0] - reference)
    return values, kept_charges, rotation, cutoff


def transform(rotation: np.ndarray, operator: np.ndarray) -> np.ndarray:
    return rotation.conj().T @ operator @ rotation


def initialize(
    onsite0: np.ndarray,
    impurity_chain: np.ndarray,
    epsilon: float,
    interaction: float,
) -> State:
    himp = impurity_hamiltonian(epsilon, interaction)
    hamiltonian = np.kron(himp, ILOCAL) + np.kron(ILOCAL, site_hamiltonian(onsite0))
    hamiltonian += coupling_term(C, CDAG, QLOCAL, impurity_chain)
    charges = np.asarray([qi + qs for qi in QLOCAL for qs in QLOCAL], dtype=int)
    values, kept_charges, rotation, cutoff = diagonalize_by_charge(
        hamiltonian, charges, None
    )
    pimp = parity(QLOCAL)
    last_ann_full = tuple(np.kron(pimp, C[a]) for a in range(NFLAV))
    last_cre_full = tuple(np.kron(pimp, CDAG[a]) for a in range(NFLAV))
    impurity_ann_full = tuple(np.kron(C[a], ILOCAL) for a in range(NFLAV))
    impurity_cre_full = tuple(np.kron(CDAG[a], ILOCAL) for a in range(NFLAV))
    composite_local = (N[1] @ C[0], N[0] @ C[1])
    composite_ann_full = tuple(np.kron(composite_local[a], ILOCAL) for a in range(NFLAV))
    composite_cre_full = tuple(operator.conj().T for operator in composite_ann_full)
    return State(
        iteration=0,
        energies=values,
        charges=kept_charges,
        last_ann=tuple(transform(rotation, operator) for operator in last_ann_full),
        last_cre=tuple(transform(rotation, operator) for operator in last_cre_full),
        impurity_ann=tuple(transform(rotation, operator) for operator in impurity_ann_full),
        impurity_cre=tuple(transform(rotation, operator) for operator in impurity_cre_full),
        composite_ann=tuple(transform(rotation, operator) for operator in composite_ann_full),
        composite_cre=tuple(transform(rotation, operator) for operator in composite_cre_full),
        truncation_cutoff=cutoff,
    )


def add_site(
    old: State,
    onsite: np.ndarray,
    hopping: np.ndarray,
    nkeep: int | None,
) -> State:
    nold = len(old.energies)
    hamiltonian = np.kron(np.diag(old.energies), ILOCAL)
    hamiltonian += np.kron(np.eye(nold), site_hamiltonian(onsite))
    hamiltonian += coupling_term(old.last_ann, old.last_cre, old.charges, hopping)
    charges = np.asarray([qold + qsite for qold in old.charges for qsite in QLOCAL], dtype=int)
    values, kept_charges, rotation, cutoff = diagonalize_by_charge(
        hamiltonian, charges, nkeep
    )
    pold = parity(old.charges)
    last_ann_full = tuple(np.kron(pold, C[a]) for a in range(NFLAV))
    last_cre_full = tuple(np.kron(pold, CDAG[a]) for a in range(NFLAV))
    impurity_ann_full = tuple(np.kron(old.impurity_ann[a], ILOCAL) for a in range(NFLAV))
    impurity_cre_full = tuple(np.kron(old.impurity_cre[a], ILOCAL) for a in range(NFLAV))
    composite_ann_full = tuple(np.kron(old.composite_ann[a], ILOCAL) for a in range(NFLAV))
    composite_cre_full = tuple(np.kron(old.composite_cre[a], ILOCAL) for a in range(NFLAV))
    return State(
        iteration=old.iteration + 1,
        energies=values,
        charges=kept_charges,
        last_ann=tuple(transform(rotation, operator) for operator in last_ann_full),
        last_cre=tuple(transform(rotation, operator) for operator in last_cre_full),
        impurity_ann=tuple(transform(rotation, operator) for operator in impurity_ann_full),
        impurity_cre=tuple(transform(rotation, operator) for operator in impurity_cre_full),
        composite_ann=tuple(transform(rotation, operator) for operator in composite_ann_full),
        composite_cre=tuple(transform(rotation, operator) for operator in composite_cre_full),
        truncation_cutoff=cutoff,
    )


def run_nrg(
    onsite: list[np.ndarray],
    hopping: list[np.ndarray],
    impurity_chain: np.ndarray,
    epsilon: float,
    hybridization_scale: float,
    reference_hybridization: float,
    interaction: float,
    sites: int,
    cluster_sites: int,
    nkeep: int,
) -> tuple[State, list[dict[str, float | int]]]:
    coupling = impurity_chain * (hybridization_scale / reference_hybridization)
    state = initialize(onsite[0], coupling, epsilon, interaction)
    flow = [{
        "iteration": 0,
        "kept": len(state.energies),
        "ground_charge": int(state.charges[np.argmin(state.energies)]),
        "cutoff": state.truncation_cutoff,
    }]
    for site in range(1, min(sites, len(onsite))):
        keep = None if site < cluster_sites else nkeep
        state = add_site(state, onsite[site], hopping[site - 1], keep)
        flow.append({
            "iteration": site,
            "kept": len(state.energies),
            "ground_charge": int(state.charges[np.argmin(state.energies)]),
            "cutoff": state.truncation_cutoff,
        })
    return state, flow


def response(state: State, z: complex) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ground = int(np.argmin(state.energies))
    e0 = float(state.energies[ground])
    q0 = int(state.charges[ground])
    plus = np.flatnonzero(state.charges == q0 + 1)
    minus = np.flatnonzero(state.charges == q0 - 1)
    green = np.zeros((2, 2), dtype=complex)
    composite = np.zeros((2, 2), dtype=complex)
    sumrule = np.zeros((2, 2), dtype=complex)
    for a in range(2):
        for b in range(2):
            for state_index in plus:
                gap = state.energies[state_index] - e0
                right = state.impurity_cre[b][state_index, ground]
                gleft = state.impurity_ann[a][ground, state_index]
                fleft = state.composite_ann[a][ground, state_index]
                green[a, b] += gleft * right / (z - gap)
                composite[a, b] += fleft * right / (z - gap)
                sumrule[a, b] += gleft * right
            for state_index in minus:
                gap = state.energies[state_index] - e0
                left = state.impurity_cre[b][ground, state_index]
                gright = state.impurity_ann[a][state_index, ground]
                fright = state.composite_ann[a][state_index, ground]
                green[a, b] += left * gright / (z + gap)
                composite[a, b] += left * fright / (z + gap)
                sumrule[a, b] += left * gright
    return green, composite, sumrule


def read_chain(csv_path: Path, summary_path: Path):
    onsite: list[np.ndarray] = []
    hopping: list[np.ndarray] = []
    with csv_path.open() as handle:
        for row in csv.DictReader(handle):
            def matrix(prefix: str) -> np.ndarray:
                values = []
                for a, b in ((1, 1), (1, 2), (2, 1), (2, 2)):
                    values.append(complex(float(row[f"{prefix}{a}{b}_re"]), float(row[f"{prefix}{a}{b}_im"])))
                return np.asarray(values).reshape(2, 2)
            onsite.append(matrix("a"))
            hopping.append(matrix("b"))
    hopping = hopping[:-1]
    summary = json.loads(summary_path.read_text())
    encoded = summary["block_chain"]["impurity_to_chain_matrix"]
    impurity_chain = np.asarray(
        [[complex(value[0], value[1]) for value in row] for row in encoded]
    )
    return onsite, hopping, impurity_chain


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chain", type=Path, required=True)
    parser.add_argument("--mapping-summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epsilon", type=float, default=0.015170349783418793)
    parser.add_argument("--hybridization", type=float, default=0.10)
    parser.add_argument("--interaction", type=float, default=0.05)
    parser.add_argument("--sites", type=int, default=9)
    parser.add_argument("--cluster-sites", type=int, default=3)
    parser.add_argument("--nkeep", type=int, default=320)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    onsite, hopping, impurity_chain = read_chain(args.chain, args.mapping_summary)
    state, flow = run_nrg(
        onsite,
        hopping,
        impurity_chain,
        args.epsilon,
        args.hybridization,
        0.10,
        args.interaction,
        args.sites,
        args.cluster_sites,
        args.nkeep,
    )
    points = [
        0.011972834346231661 - 0.0028072162737951614j,
        0.011972834346231661 + 0.0028072162737951614j,
        0.0 + 0.01j,
        0.02 + 0.01j,
    ]
    responses = []
    maximum_sumrule_error = 0.0
    for z in points:
        green, composite, sumrule = response(state, z)
        sigma = args.interaction * composite @ np.linalg.inv(green)
        maximum_sumrule_error = max(
            maximum_sumrule_error, float(np.linalg.norm(sumrule - np.eye(2)))
        )
        responses.append({
            "z": [z.real, z.imag],
            "G": [[[float(v.real), float(v.imag)] for v in row] for row in green],
            "F": [[[float(v.real), float(v.imag)] for v in row] for row in composite],
            "Sigma_U": [[[float(v.real), float(v.imag)] for v in row] for row in sigma],
            "sumrule_error": float(np.linalg.norm(sumrule - np.eye(2))),
        })
    result = {
        "status": "PILOT_PASS" if np.all(np.isfinite(state.energies)) else "FAIL",
        "scope": (
            "Hermitian block-chain kept-space NRG pilot. The response is not "
            "yet an Anders-Schiller/DM-NRG complete-basis certificate."
        ),
        "parameters": {
            "epsilon": args.epsilon,
            "hybridization": args.hybridization,
            "interaction": args.interaction,
            "sites": args.sites,
            "cluster_sites": args.cluster_sites,
            "nkeep": args.nkeep,
        },
        "flow": flow,
        "final": {
            "kept": len(state.energies),
            "ground_charge": int(state.charges[np.argmin(state.energies)]),
            "maximum_sumrule_error": maximum_sumrule_error,
        },
        "response": responses,
    }
    (args.out / "python_fullk_nrg_pilot_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))
    if result["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
