#!/usr/bin/env python3
"""Validate the archived finite-chain common-contour gate summary."""

import json
from pathlib import Path


SUMMARY = Path(__file__).with_name("fullk_n_scaling_gate_summary.json")


def require(condition, message):
    if not condition:
        raise SystemExit(f"FAIL: {message}")


data = json.loads(SUMMARY.read_text(encoding="utf-8"))
require(data["status"] == "FINITE_N_COMMON_CONTOUR_ROUCHE_PASS", "status")
require(data["chain_lengths"] == [10, 12], "chain lengths")

for resolution in ("resolution_128", "resolution_256"):
    block = data[resolution]
    pair = block["pairwise_rouche"]
    require(pair["pass"], f"{resolution} pass flag")
    require(
        pair["supremum_discriminant_difference"]
        < pair["minimum_base_boundary_modulus"],
        f"{resolution} Rouche inequality",
    )
    require(pair["conservative_ratio"] < 1.0, f"{resolution} ratio")
    for chain in ("10", "12"):
        item = block["chains"][chain]
        require(abs(item["winding"] + 1.0) < 0.05, f"{resolution} N={chain} winding")
        require(
            abs(item["half_winding"] + 0.5) < 0.05,
            f"{resolution} N={chain} half-winding",
        )
        require(item["endpoint_permutation"] == [1, 0], f"{resolution} N={chain} exchange")
        require(
            item["maximum_assignment_to_half_separation_ratio"] < 0.5,
            f"{resolution} N={chain} assignment guard",
        )
        require(item["maximum_sum_rule_error"] < 1e-10, f"{resolution} N={chain} sum rule")

print("PASS: finite-N common-contour Rouche gate summary")

