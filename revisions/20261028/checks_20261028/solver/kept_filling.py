"""Additive occupation layer for the retrieved full-matrix iterative pilot.

Even composite operators are transformed independently at every truncation.
Products of projected fermion matrices do not replace physical n and D.
This layer gives finite-chain kept-space occupations, not an FDM spectrum.
"""
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor"))
import python_fullk_nrg_pilot as pilot
import charge_solver


@dataclass
class State(pilot.State):
    density_operators: tuple
    double_operator: np.ndarray


def project(hamiltonian, charges, last_ann, last_cre, imp_ann, imp_cre,
            comp_ann, comp_cre, density, double, nkeep, iteration):
    energies, q, rotation, cutoff = pilot.diagonalize_by_charge(hamiltonian, charges, nkeep)
    def operators(values):
        return tuple(pilot.transform(rotation, op) for op in values)
    return State(iteration, energies, q, operators(last_ann), operators(last_cre),
                 operators(imp_ann), operators(imp_cre), operators(comp_ann),
                 operators(comp_cre), cutoff, operators(density), pilot.transform(rotation, double))


def initialize(onsite0, coupling, epsilon, interaction):
    h = np.kron(pilot.impurity_hamiltonian(epsilon, interaction), pilot.ILOCAL)
    h += np.kron(pilot.ILOCAL, pilot.site_hamiltonian(onsite0))
    h += pilot.coupling_term(pilot.C, pilot.CDAG, pilot.QLOCAL, coupling)
    charges = np.asarray([a+b for a in pilot.QLOCAL for b in pilot.QLOCAL])
    last_ann = [np.kron(pilot.parity(pilot.QLOCAL), op) for op in pilot.C]
    last_cre = [op.conj().T for op in last_ann]
    imp_ann = [np.kron(op, pilot.ILOCAL) for op in pilot.C]
    imp_cre = [op.conj().T for op in imp_ann]
    comp_ann = [np.kron(pilot.N[1-a] @ pilot.C[a], pilot.ILOCAL) for a in range(2)]
    comp_cre = [op.conj().T for op in comp_ann]
    density = [np.kron(pilot.CDAG[b] @ pilot.C[a], pilot.ILOCAL) for a in range(2) for b in range(2)]
    double = np.kron(pilot.N[0] @ pilot.N[1], pilot.ILOCAL)
    return project(h, charges, last_ann, last_cre, imp_ann, imp_cre, comp_ann, comp_cre,
                   density, double, None, 0)


def add_site(old, onsite, hopping, nkeep):
    n = len(old.energies)
    h = np.kron(np.diag(old.energies), pilot.ILOCAL)
    h += np.kron(np.eye(n), pilot.site_hamiltonian(onsite))
    h += pilot.coupling_term(old.last_ann, old.last_cre, old.charges, hopping)
    q = np.asarray([a+b for a in old.charges for b in pilot.QLOCAL])
    last_ann = [np.kron(pilot.parity(old.charges), op) for op in pilot.C]
    last_cre = [op.conj().T for op in last_ann]
    lift = lambda ops: [np.kron(op, pilot.ILOCAL) for op in ops]
    return project(h, q, last_ann, last_cre, lift(old.impurity_ann), lift(old.impurity_cre),
                   lift(old.composite_ann), lift(old.composite_cre), lift(old.density_operators),
                   np.kron(old.double_operator, pilot.ILOCAL), nkeep, old.iteration+1)


def observables(state, tolerance=1e-11):
    selected = np.flatnonzero(state.energies <= state.energies.min()+tolerance)
    value = lambda op: np.mean(np.diag(op)[selected])
    rho = np.asarray([value(op) for op in state.density_operators]).reshape(2, 2)
    nd = float(np.trace(rho).real)
    double = float(value(state.double_operator).real)
    if (nd < -1e-8 or nd > 2+1e-8 or double < -1e-8 or double > nd/2+1e-8
            or double < nd-1-1e-8):
        raise RuntimeError("kept-space physical occupation bounds failed")
    return {"nd": nd, "n0": float(rho[0, 0].real), "n1": float(rho[1, 1].real),
            "double_occupancy": double, "local_moment_squared": nd-2*double,
            "single_occupancy_probability": nd-2*double, "empty_probability": 1-nd+double,
            "density_matrix": charge_solver.encode_matrix(rho),
            "double_minus_gaussian_reference": double-float(np.linalg.det(rho).real),
            "ground_charges": sorted(set(int(state.charges[i]) for i in selected)),
            "ground_degeneracy": len(selected),
            "state_prescription": "T=0 equal ground ensemble in finite-chain kept space"}


def run(onsite, hopping, coupling, epsilon, interaction, sites, nkeep, cluster_sites=2):
    if sites > len(onsite):
        raise ValueError("requested more sites than the mapped chain contains")
    if nkeep < 16 or nkeep > 1024:
        raise ValueError("this dense pilot accepts 16 <= nkeep <= 1024; use a sparse production backend beyond that")
    state = initialize(onsite[0], coupling, epsilon, interaction)
    flow = [{"iteration": 0, "kept": len(state.energies), **observables(state)}]
    for site in range(1, sites):
        state = add_site(state, onsite[site], hopping[site-1], None if site < cluster_sites else nkeep)
        flow.append({"iteration": site, "kept": len(state.energies), **observables(state)})
    return state, flow
