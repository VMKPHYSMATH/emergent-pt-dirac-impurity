#!/usr/bin/env python3
"""z-averaged exact complete-basis response for block Wilson chains.

The chain argument is a JSON manifest emitted by
``full_k_wilson_preflight.py --emit-all-z-chains``.  Each discretization is
solved independently by ``exact_block_chain_response`` and the impurity and
EOM composite Green functions are averaged before the self-energy ratio is
formed by the existing center/contour drivers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import exact_block_chain_response as exact


@dataclass
class ZAverageState:
    states: list[exact.ExactState]
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
        chains.append(exact.read_chain(chain_path, summary_path))
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
        state, flow = exact.run_nrg(
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
        flows.append(flow[-1])
    charges = [state.ground_charge for state in states]
    summary_flow = [{
        "iteration": sites - 1,
        "kept": int(sum(len(state.ground_basis) for state in states)),
        "ground_charge": charges[0] if len(set(charges)) == 1 else -1,
        "cutoff": None,
        "z_ground_charges": charges,
        "z_sector_dimensions": [state.sector_dimensions for state in states],
    }]
    return ZAverageState(states=states), summary_flow


def response(state: ZAverageState, z: complex):
    key = complex(z)
    cached = state.response_cache.get(key)
    if cached is not None:
        return tuple(value.copy() for value in cached)
    values = [exact.response(member, z) for member in state.states]
    green = sum(item[0] for item in values) / len(values)
    composite = sum(item[1] for item in values) / len(values)
    sumrule = sum(item[2] for item in values) / len(values)
    result = (green, composite, sumrule)
    state.response_cache[key] = tuple(value.copy() for value in result)
    return result
