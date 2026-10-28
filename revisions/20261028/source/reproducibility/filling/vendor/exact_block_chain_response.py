#!/usr/bin/env python3
"""Exact fixed-charge correction-vector response for a block Wilson chain.

The full-k pilot originally evaluated the Green function only in the final
kept NRG space.  For the low-density Wilson-chain window used by the causal
EP certificate the relevant fixed-charge sectors are small enough to avoid
that approximation completely.  This module constructs the Hermitian
many-body Hamiltonian in the ground-state charge sector and its N +/- 1
sectors, and evaluates the zero-temperature impurity and equation-of-motion
correlators by sparse linear solves.

It exposes ``read_chain``, ``run_nrg`` and ``response`` with the interface
used by the existing center and contour drivers.  ``nkeep`` and
``cluster_sites`` are accepted for compatibility but no state is truncated.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, splu


NFLAV = 2


def read_chain(csv_path: Path, summary_path: Path):
    onsite: list[np.ndarray] = []
    hopping: list[np.ndarray] = []
    with csv_path.open() as handle:
        for row in csv.DictReader(handle):
            def matrix(prefix: str) -> np.ndarray:
                values = []
                for a, b in ((1, 1), (1, 2), (2, 1), (2, 2)):
                    values.append(
                        complex(
                            float(row[f"{prefix}{a}{b}_re"]),
                            float(row[f"{prefix}{a}{b}_im"]),
                        )
                    )
                return np.asarray(values, dtype=complex).reshape(2, 2)

            onsite.append(matrix("a"))
            hopping.append(matrix("b"))
    hopping = hopping[:-1]
    summary = json.loads(summary_path.read_text())
    encoded = summary["block_chain"]["impurity_to_chain_matrix"]
    impurity_chain = np.asarray(
        [[complex(value[0], value[1]) for value in row] for row in encoded],
        dtype=complex,
    )
    return onsite, hopping, impurity_chain


def fixed_charge_basis(orbitals: int, charge: int) -> tuple[np.ndarray, dict[int, int]]:
    if charge < 0 or charge > orbitals:
        return np.asarray([], dtype=object), {}
    states = []
    for occupied in itertools.combinations(range(orbitals), charge):
        state = 0
        for orbital in occupied:
            state |= 1 << orbital
        states.append(state)
    array = np.asarray(states, dtype=object)
    return array, {int(state): index for index, state in enumerate(array)}


def apply_annihilation(state: int, orbital: int) -> tuple[int, int] | None:
    mask = 1 << orbital
    if not state & mask:
        return None
    sign = -1 if (state & (mask - 1)).bit_count() % 2 else 1
    return state ^ mask, sign


def apply_creation(state: int, orbital: int) -> tuple[int, int] | None:
    mask = 1 << orbital
    if state & mask:
        return None
    sign = -1 if (state & (mask - 1)).bit_count() % 2 else 1
    return state | mask, sign


def build_one_body(
    onsite: list[np.ndarray],
    hopping: list[np.ndarray],
    impurity_chain: np.ndarray,
    epsilon: float,
    hybridization_scale: float,
    reference_hybridization: float,
    sites: int,
) -> np.ndarray:
    sites = min(sites, len(onsite))
    orbitals = 2 * (sites + 1)
    matrix = np.zeros((orbitals, orbitals), dtype=complex)
    matrix[0:2, 0:2] = epsilon * np.eye(2)
    coupling = impurity_chain * (hybridization_scale / reference_hybridization)
    matrix[0:2, 2:4] = coupling
    matrix[2:4, 0:2] = coupling.conj().T
    for site in range(sites):
        start = 2 + 2 * site
        matrix[start:start + 2, start:start + 2] = onsite[site]
        if site + 1 < sites:
            following = start + 2
            matrix[start:start + 2, following:following + 2] = hopping[site]
            matrix[following:following + 2, start:start + 2] = hopping[site].conj().T
    return 0.5 * (matrix + matrix.conj().T)


def build_sector_hamiltonian(
    one_body: np.ndarray,
    interaction: float,
    basis: np.ndarray,
    basis_index: dict[int, int],
    tolerance: float = 1.0e-15,
) -> sparse.csr_matrix:
    dimension = len(basis)
    if dimension == 0:
        return sparse.csr_matrix((0, 0), dtype=complex)
    orbitals = one_body.shape[0]
    diagonal = np.real(np.diag(one_body))
    offdiagonal = [
        (i, j, one_body[i, j])
        for i in range(orbitals)
        for j in range(orbitals)
        if i != j and abs(one_body[i, j]) > tolerance
    ]
    rows: list[int] = []
    columns: list[int] = []
    values: list[complex] = []
    for column, encoded in enumerate(basis):
        state = int(encoded)
        value = sum(diagonal[i] for i in range(orbitals) if state & (1 << i))
        if state & 1 and state & 2:
            value += interaction
        rows.append(column)
        columns.append(column)
        values.append(complex(value))
        for i, j, amplitude in offdiagonal:
            removed = apply_annihilation(state, j)
            if removed is None:
                continue
            intermediate, sign1 = removed
            added = apply_creation(intermediate, i)
            if added is None:
                continue
            target, sign2 = added
            row = basis_index.get(target)
            if row is not None:
                rows.append(row)
                columns.append(column)
                values.append(amplitude * sign1 * sign2)
    result = sparse.coo_matrix(
        (np.asarray(values), (np.asarray(rows), np.asarray(columns))),
        shape=(dimension, dimension),
        dtype=complex,
    ).tocsr()
    return 0.5 * (result + result.getH())


def lowest_eigenpair(matrix: sparse.csr_matrix) -> tuple[float, np.ndarray]:
    dimension = matrix.shape[0]
    if dimension == 0:
        raise ValueError("empty charge sector")
    if dimension == 1:
        return float(np.real(matrix[0, 0])), np.ones(1, dtype=complex)
    if dimension <= 256:
        values, vectors = np.linalg.eigh(matrix.toarray())
        return float(values[0]), vectors[:, 0]
    values, vectors = eigsh(matrix, k=1, which="SA", tol=1.0e-12, maxiter=200000)
    return float(values[0]), vectors[:, 0]


def operator_vector(
    source_basis: np.ndarray,
    source_vector: np.ndarray,
    target_index: dict[int, int],
    orbital: int,
    creation: bool,
    require_other_impurity_occupied: bool = False,
) -> np.ndarray:
    result = np.zeros(len(target_index), dtype=complex)
    other = 1 - orbital
    for source_index, encoded in enumerate(source_basis):
        coefficient = source_vector[source_index]
        if coefficient == 0.0:
            continue
        state = int(encoded)
        if require_other_impurity_occupied and not state & (1 << other):
            continue
        applied = apply_creation(state, orbital) if creation else apply_annihilation(state, orbital)
        if applied is None:
            continue
        target, sign = applied
        result[target_index[target]] += sign * coefficient
    return result


@dataclass
class ExactState:
    interaction: float
    one_body: np.ndarray
    ground_charge: int
    ground_energy: float
    ground_basis: np.ndarray
    ground_vector: np.ndarray
    plus_hamiltonian: sparse.csr_matrix
    minus_hamiltonian: sparse.csr_matrix
    plus_vectors_d: tuple[np.ndarray, np.ndarray]
    minus_vectors_d: tuple[np.ndarray, np.ndarray]
    plus_vectors_f: tuple[np.ndarray, np.ndarray]
    minus_vectors_f: tuple[np.ndarray, np.ndarray]
    sumrule: np.ndarray
    sector_dimensions: dict[int, int]
    response_cache: dict[complex, tuple[np.ndarray, np.ndarray, np.ndarray]] = field(
        default_factory=dict
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
) -> tuple[ExactState, list[dict[str, float | int | dict[int, int]]]]:
    del cluster_sites, nkeep
    one_body = build_one_body(
        onsite,
        hopping,
        impurity_chain,
        epsilon,
        hybridization_scale,
        reference_hybridization,
        sites,
    )
    orbitals = one_body.shape[0]
    one_body_eigenvalues = np.linalg.eigvalsh(one_body)
    noninteracting_charge = int(np.count_nonzero(one_body_eigenvalues < 0.0))
    if interaction < 0.0:
        raise ValueError(
            "the rigorous charge-sector pruning used by the exact solver "
            "requires repulsive U >= 0"
        )
    # For U >= 0, H_U is positive semidefinite.  Consequently the exact
    # ground energy in charge sector q is bounded below by the sum of the q
    # lowest one-body eigenvalues.  Build the likely sectors first and then
    # use this bound to exclude all sectors that cannot beat the best energy
    # already found.  This keeps longer low-density chains exact without
    # constructing combinatorially large, rigorously irrelevant sectors.
    sorted_one_body = np.sort(one_body_eigenvalues)
    charge_lower_bounds = np.concatenate(([0.0], np.cumsum(sorted_one_body)))
    seed_charges = {noninteracting_charge}
    sector_data: dict[int, tuple[np.ndarray, dict[int, int], sparse.csr_matrix]] = {}
    ground_records = []

    def build_charge(charge: int) -> None:
        basis, index = fixed_charge_basis(orbitals, charge)
        hamiltonian = build_sector_hamiltonian(one_body, interaction, basis, index)
        energy, vector = lowest_eigenpair(hamiltonian)
        sector_data[charge] = (basis, index, hamiltonian)
        ground_records.append((energy, charge, vector))

    for charge in sorted(seed_charges):
        build_charge(charge)
    best_energy = min(record[0] for record in ground_records)
    for charge in np.argsort(charge_lower_bounds):
        charge = int(charge)
        if charge in sector_data:
            continue
        if charge_lower_bounds[charge] > best_energy + 1.0e-12:
            continue
        build_charge(charge)
        best_energy = min(best_energy, ground_records[-1][0])
    ground_energy, ground_charge, ground_vector = min(ground_records, key=lambda item: item[0])
    if ground_charge == 0 or ground_charge == orbitals:
        raise RuntimeError(f"ground charge {ground_charge} lacks both particle and hole sectors")
    for charge in (ground_charge - 1, ground_charge, ground_charge + 1):
        if charge not in sector_data:
            basis, index = fixed_charge_basis(orbitals, charge)
            sector_data[charge] = (
                basis,
                index,
                build_sector_hamiltonian(one_body, interaction, basis, index),
            )
    ground_basis, _, _ = sector_data[ground_charge]
    plus_basis, plus_index, plus_hamiltonian = sector_data[ground_charge + 1]
    minus_basis, minus_index, minus_hamiltonian = sector_data[ground_charge - 1]
    plus_d = tuple(
        operator_vector(ground_basis, ground_vector, plus_index, a, True)
        for a in range(NFLAV)
    )
    minus_d = tuple(
        operator_vector(ground_basis, ground_vector, minus_index, a, False)
        for a in range(NFLAV)
    )
    plus_f = tuple(
        operator_vector(
            ground_basis,
            ground_vector,
            plus_index,
            a,
            True,
            require_other_impurity_occupied=True,
        )
        for a in range(NFLAV)
    )
    minus_f = tuple(
        operator_vector(
            ground_basis,
            ground_vector,
            minus_index,
            a,
            False,
            require_other_impurity_occupied=True,
        )
        for a in range(NFLAV)
    )
    sumrule = np.zeros((NFLAV, NFLAV), dtype=complex)
    for a in range(NFLAV):
        for b in range(NFLAV):
            sumrule[a, b] = np.vdot(plus_d[a], plus_d[b]) + np.vdot(minus_d[b], minus_d[a])
    state = ExactState(
        interaction=interaction,
        one_body=one_body,
        ground_charge=ground_charge,
        ground_energy=ground_energy,
        ground_basis=ground_basis,
        ground_vector=ground_vector,
        plus_hamiltonian=plus_hamiltonian,
        minus_hamiltonian=minus_hamiltonian,
        plus_vectors_d=plus_d,
        minus_vectors_d=minus_d,
        plus_vectors_f=plus_f,
        minus_vectors_f=minus_f,
        sumrule=sumrule,
        sector_dimensions={charge: len(data[0]) for charge, data in sector_data.items()},
    )
    flow = [{
        "iteration": min(sites, len(onsite)) - 1,
        "kept": int(len(ground_basis)),
        "ground_charge": int(ground_charge),
        "cutoff": float("nan"),
        "sector_dimensions": state.sector_dimensions,
    }]
    return state, flow


def response(state: ExactState, z: complex) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = complex(z)
    cached = state.response_cache.get(key)
    if cached is not None:
        return tuple(value.copy() for value in cached)  # type: ignore[return-value]
    plus_matrix = (
        (z + state.ground_energy) * sparse.eye(state.plus_hamiltonian.shape[0], format="csc")
        - state.plus_hamiltonian.tocsc()
    )
    minus_matrix = (
        (z - state.ground_energy) * sparse.eye(state.minus_hamiltonian.shape[0], format="csc")
        + state.minus_hamiltonian.tocsc()
    )
    plus_solver = splu(plus_matrix)
    minus_solver = splu(minus_matrix)
    plus_solutions_d = [plus_solver.solve(vector) for vector in state.plus_vectors_d]
    minus_solutions_d = [minus_solver.solve(vector) for vector in state.minus_vectors_d]
    plus_solutions_f = [plus_solver.solve(vector) for vector in state.plus_vectors_f]
    minus_solutions_f = [minus_solver.solve(vector) for vector in state.minus_vectors_f]
    green = np.zeros((NFLAV, NFLAV), dtype=complex)
    composite = np.zeros((NFLAV, NFLAV), dtype=complex)
    for a in range(NFLAV):
        for b in range(NFLAV):
            green[a, b] = (
                np.vdot(state.plus_vectors_d[a], plus_solutions_d[b])
                + np.vdot(state.minus_vectors_d[b], minus_solutions_d[a])
            )
            composite[a, b] = (
                np.vdot(state.plus_vectors_f[a], plus_solutions_d[b])
                + np.vdot(state.minus_vectors_d[b], minus_solutions_f[a])
            )
    answer = (green, composite, state.sumrule.copy())
    state.response_cache[key] = tuple(value.copy() for value in answer)
    return answer


def noninteracting_green(state: ExactState, z: complex) -> np.ndarray:
    return np.linalg.inv(z * np.eye(state.one_body.shape[0]) - state.one_body)[0:2, 0:2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chain", type=Path, required=True)
    parser.add_argument("--mapping-summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epsilon", type=float, default=0.015170349783418793)
    parser.add_argument("--hybridization", type=float, default=0.10)
    parser.add_argument("--interaction", type=float, default=0.0)
    parser.add_argument("--sites", type=int, default=11)
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
        0,
        0,
    )
    points = [
        0.011972834346231661 - 0.0028072162737951614j,
        0.011972834346231661 + 0.0028072162737951614j,
        0.0 + 0.01j,
        0.02 + 0.01j,
    ]
    reports = []
    maximum_sumrule_error = float(np.linalg.norm(state.sumrule - np.eye(2)))
    maximum_u0_error = 0.0
    for point in points:
        green, composite, _ = response(state, point)
        sigma = args.interaction * composite @ np.linalg.inv(green)
        u0_error = None
        if args.interaction == 0.0:
            reference = noninteracting_green(state, point)
            u0_error = float(np.linalg.norm(green - reference) / np.linalg.norm(reference))
            maximum_u0_error = max(maximum_u0_error, u0_error)
        reports.append({
            "z": [point.real, point.imag],
            "Sigma_U_norm": float(np.linalg.norm(sigma)),
            "U0_relative_G_error": u0_error,
        })
    passed = bool(
        maximum_sumrule_error < 1.0e-11
        and (args.interaction != 0.0 or maximum_u0_error < 1.0e-10)
    )
    result = {
        "status": "PASS" if passed else "FAIL",
        "scope": "Exact zero-temperature complete-basis correction-vector response on the stated finite block Wilson chain.",
        "parameters": {
            "epsilon": args.epsilon,
            "hybridization": args.hybridization,
            "interaction": args.interaction,
            "sites": args.sites,
        },
        "ground": {
            "charge": state.ground_charge,
            "energy": state.ground_energy,
            "sector_dimensions": state.sector_dimensions,
        },
        "maximum_sumrule_error": maximum_sumrule_error,
        "maximum_U0_relative_G_error": maximum_u0_error if args.interaction == 0.0 else None,
        "response": reports,
        "flow": flow,
    }
    (args.out / "exact_block_chain_response_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
