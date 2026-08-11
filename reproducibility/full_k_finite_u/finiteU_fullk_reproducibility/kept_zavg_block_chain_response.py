#!/usr/bin/env python3
"""z-averaged kept-space response for the matrix block Wilson-chain NRG.

This adapter is deliberately labelled kept-space: it is used only for the
long-chain and Nkeep convergence legs, while the short-chain response and
topology are calibrated against the independent exact complete-basis solver.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import python_fullk_nrg_pilot as kept


@dataclass
class KeptZAverageState:
    states: list[kept.State]
    response_cache: dict[complex, tuple[np.ndarray, np.ndarray, np.ndarray]] = field(
        default_factory=dict
    )


def read_chain(manifest_path: Path, ignored_summary_path: Path):
    del ignored_summary_path
    payload = json.loads(manifest_path.read_text())
    chains = []
    for item in payload["chains"]:
        chain_path = manifest_path.parent / item["chain"]
        summary_path = manifest_path.parent / item["summary"]
        chains.append(kept.read_chain(chain_path, summary_path))
    if not chains:
        raise RuntimeError("empty z-chain manifest")
    return chains, None, None


def run_nrg(
    chains,
    unused_hopping,
    unused_impurity_chain,
    epsilon: float,
    hybridization_scale: float,
    reference_hybridization: float,
    interaction: float,
    sites: int,
    cluster_sites: int,
    nkeep: int,
):
    del unused_hopping, unused_impurity_chain
    states = []
    flows = []
    for onsite, hopping, impurity_chain in chains:
        state, flow = kept.run_nrg(
            onsite,
            hopping,
            impurity_chain,
            epsilon,
            hybridization_scale,
            reference_hybridization,
            interaction,
            sites,
            cluster_sites,
            nkeep,
        )
        states.append(state)
        flows.append(flow)
    charges = [int(flow[-1]["ground_charge"]) for flow in flows]
    summary_flow = [{
        "iteration": sites - 1,
        "kept": int(sum(len(state.energies) for state in states)),
        "ground_charge": charges[0] if len(set(charges)) == 1 else -1,
        "cutoff": float(max(flow[-1]["cutoff"] for flow in flows)),
        "z_ground_charges": charges,
        "z_cutoffs": [float(flow[-1]["cutoff"]) for flow in flows],
        "z_flows": flows,
    }]
    return KeptZAverageState(states=states), summary_flow


def response(state: KeptZAverageState, z: complex):
    key = complex(z)
    cached = state.response_cache.get(key)
    if cached is not None:
        return tuple(value.copy() for value in cached)
    values = [kept.response(member, z) for member in state.states]
    green = sum(item[0] for item in values) / len(values)
    composite = sum(item[1] for item in values) / len(values)
    sumrule = sum(item[2] for item in values) / len(values)
    result = (green, composite, sumrule)
    state.response_cache[key] = tuple(value.copy() for value in result)
    return result
