"""Hermitian, zero-temperature charge solver for the filling reference state.

Uses the unmodified August exact-sector operators. This is finite-chain ED,
not thermodynamic NRG. Degenerate ground states get equal ensemble weights.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
import sys

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, splu

sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor"))
import exact_block_chain_response as exact


class SectorTooLarge(RuntimeError):
    pass


@dataclass
class Member:
    charge: int
    basis: np.ndarray
    vector: np.ndarray
    weight: float = 1.0
    response_state: object | None = None


@dataclass
class Ensemble:
    one_body: np.ndarray
    interaction: float
    energy: float
    members: list[Member]
    sector_dimensions: dict[int, int]
    charge_search: list[dict]
    max_sector_dimension: int
    tolerance: float
    response_cache: dict = field(default_factory=dict)


def sector(one_body, interaction, charge, limit):
    norb = one_body.shape[0]
    dimension = math.comb(norb, charge) if 0 <= charge <= norb else 0
    if dimension > limit:
        raise SectorTooLarge(
            f"charge {charge}, {norb} orbitals: {dimension} states exceed "
            f"max_sector_dimension={limit}; no sector has been silently omitted"
        )
    basis, index = exact.fixed_charge_basis(norb, charge)
    return basis, index, exact.build_sector_hamiltonian(one_body, interaction, basis, index)


def ground_vectors(matrix, target, tolerance):
    """Resolve the entire degenerate manifold, with a bounded sparse fallback."""
    n = matrix.shape[0]
    if n <= 256:
        values, vectors = np.linalg.eigh(matrix.toarray())
    else:
        k = 4
        while True:
            values, vectors = eigsh(matrix, k=min(k, n - 2), which="SA", tol=1e-12)
            order = np.argsort(values)
            values, vectors = values[order], vectors[:, order]
            if values[-1] > target + tolerance:
                break
            k *= 2
            if k > 64:
                raise RuntimeError("unresolved ground degeneracy: more than 32 states")
    selection = np.abs(values - target) <= tolerance
    if not np.any(selection):
        raise RuntimeError("ground eigenvalue failed its reconstruction check")
    # Numerical sparse eigenvectors in a degenerate manifold may not be
    # orthogonal. QR gives the same projector with an orthonormal basis.
    selected, _ = np.linalg.qr(vectors[:, selection])
    return [selected[:, j] for j in range(selected.shape[1])]


def solve(one_body, interaction, max_sector_dimension=50000, tolerance=1e-10):
    one_body = np.asarray(one_body, dtype=complex)
    if not np.isfinite(one_body).all() or interaction < 0:
        raise ValueError("finite Hermitian one-body matrix and repulsive U >= 0 required")
    if np.linalg.norm(one_body - one_body.conj().T) > 1e-10:
        raise ValueError("physical filling requires a Hermitian reference Hamiltonian")
    levels = np.linalg.eigvalsh(one_body)
    bounds = np.concatenate(([0.0], np.cumsum(levels)))
    seed = int(np.count_nonzero(levels < 0))
    records = []
    dimensions = {}
    best = math.inf
    for charge in [seed] + [int(q) for q in np.argsort(bounds) if int(q) != seed]:
        if bounds[charge] > best + tolerance:
            continue
        basis, index, h = sector(one_body, interaction, charge, max_sector_dimension)
        energy, vector = exact.lowest_eigenpair(h)
        dimensions[charge] = len(basis)
        records.append({"charge": charge, "energy": energy, "lower_bound": float(bounds[charge])})
        best = min(best, energy)
    members = []
    for record in records:
        if abs(record["energy"] - best) > tolerance:
            continue
        charge = record["charge"]
        basis, index, h = sector(one_body, interaction, charge, max_sector_dimension)
        for vector in ground_vectors(h, best, tolerance):
            members.append(Member(charge, basis, vector))
    for member in members:
        member.weight = 1.0 / len(members)
    return Ensemble(one_body, interaction, best, members, dimensions, records,
                    max_sector_dimension, tolerance)


def observables(state):
    density = np.zeros((2, 2), dtype=complex)  # density[a,b] = <d_b^dag d_a>
    double = 0.0
    moment = 0.0
    for member in state.members:
        index = {int(s): i for i, s in enumerate(member.basis)}
        weights = np.abs(member.vector) ** 2
        n0 = np.array([bool(int(s) & 1) for s in member.basis], dtype=float)
        n1 = np.array([bool(int(s) & 2) for s in member.basis], dtype=float)
        double += member.weight * float(weights @ (n0 * n1))
        moment += member.weight * float(weights @ ((n0 - n1) ** 2))
        for a in range(2):
            for b in range(2):
                value = 0j
                for col, encoded in enumerate(member.basis):
                    removed = exact.apply_annihilation(int(encoded), a)
                    if removed is None:
                        continue
                    added = exact.apply_creation(removed[0], b)
                    if added is not None:
                        value += (np.conj(member.vector[index[added[0]]])
                                  * removed[1] * added[1] * member.vector[col])
                density[a, b] += member.weight * value
    nd = float(np.trace(density).real)
    density_eigenvalues = np.linalg.eigvalsh(density)
    bounds_error = max(0.0, -nd, nd - 2, -double,
                       double - min(density[0, 0].real, density[1, 1].real),
                       nd - 1 - double, -float(density_eigenvalues.min()),
                       float(density_eigenvalues.max()) - 1)
    residual = abs(moment - (nd - 2 * double))
    if bounds_error > 1e-9 or residual > 1e-9:
        raise RuntimeError("occupation bounds or local-moment identity failed")
    return {
        "nd": nd, "n0": float(density[0, 0].real), "n1": float(density[1, 1].real),
        "double_occupancy": double, "single_occupancy_probability": nd - 2 * double,
        "empty_probability": 1 - nd + double,
        "local_moment_squared": moment, "local_moment_identity_error": residual,
        "double_minus_gaussian_reference": double - float(np.linalg.det(density).real),
        "density_matrix": encode_matrix(density),
        "density_eigenvalues": density_eigenvalues.tolist(),
        "occupation_bounds_error": bounds_error,
        "ground_charges": sorted(set(m.charge for m in state.members)),
        "ground_degeneracy": len(state.members), "grand_ground_energy": state.energy,
        "sector_dimensions": state.sector_dimensions,
        "state_prescription": "T=0 grand-canonical Hermitian finite-chain ground ensemble",
    }


def gaussian_observables(one_body, tolerance=1e-12):
    """Independent U=0 covariance calculation, including half-filled zero modes."""
    levels, vectors = np.linalg.eigh(one_body)
    occupied = np.where(levels < -tolerance, 1., np.where(levels > tolerance, 0., .5))
    density = ((vectors * occupied) @ vectors.conj().T)[:2, :2]
    nd = float(np.trace(density).real)
    double = float(np.linalg.det(density).real)
    return {"nd": nd, "double_occupancy": double,
            "local_moment_squared": nd - 2 * double,
            "density_matrix": encode_matrix(density)}


def response_member(state, member, z):
    if member.response_state is None:
        plus_basis, plus_index, hp = sector(state.one_body, state.interaction,
                                          member.charge + 1, state.max_sector_dimension)
        minus_basis, minus_index, hm = sector(state.one_body, state.interaction,
                                            member.charge - 1, state.max_sector_dimension)
        def vectors(index, creation, composite=False):
            return tuple(exact.operator_vector(member.basis, member.vector, index, a,
                                               creation, composite) for a in range(2))
        pd, md = vectors(plus_index, True), vectors(minus_index, False)
        pf, mf = vectors(plus_index, True, True), vectors(minus_index, False, True)
        member.response_state = hp, hm, pd, md, pf, mf
    hp, hm, pd, md, pf, mf = member.response_state
    plus = splu((z + state.energy) * sparse.eye(hp.shape[0], format="csc") - hp.tocsc()) if hp.shape[0] else None
    minus = splu((z - state.energy) * sparse.eye(hm.shape[0], format="csc") + hm.tocsc()) if hm.shape[0] else None
    green, composite, sumrule = (np.zeros((2, 2), complex) for _ in range(3))
    for a in range(2):
        for b in range(2):
            if plus:
                green[a, b] += np.vdot(pd[a], plus.solve(pd[b]))
                composite[a, b] += np.vdot(pf[a], plus.solve(pd[b]))
            if minus:
                green[a, b] += np.vdot(md[b], minus.solve(md[a]))
                composite[a, b] += np.vdot(md[b], minus.solve(mf[a]))
            sumrule[a, b] = np.vdot(pd[a], pd[b]) + np.vdot(md[b], md[a])
    return green, composite, sumrule


def response(state, z):
    key = complex(z)
    if key not in state.response_cache:
        values = [response_member(state, m, key) for m in state.members]
        state.response_cache[key] = tuple(sum(m.weight * value[j] for m, value in
                                              zip(state.members, values)) for j in range(3))
    return tuple(a.copy() for a in state.response_cache[key])


def sigma_u(state, z):
    green, composite, sumrule = response(state, z)
    if np.linalg.norm(sumrule - np.eye(2)) > 1e-9:
        raise RuntimeError("impurity anticommutator sum rule failed")
    if np.linalg.cond(green) > 1e12:
        raise RuntimeError("interaction self-energy extraction has an ill-conditioned Green matrix")
    return np.linalg.solve(green.T, (state.interaction * composite).T).T


def hartree_fock(state):
    rho = decode_matrix(observables(state)["density_matrix"])
    return state.interaction * (np.trace(rho) * np.eye(2) - rho)


def encode_matrix(a):
    return [[[float(v.real), float(v.imag)] for v in row] for row in a]


def decode_matrix(a):
    return np.asarray([[complex(*v) for v in row] for row in a])
