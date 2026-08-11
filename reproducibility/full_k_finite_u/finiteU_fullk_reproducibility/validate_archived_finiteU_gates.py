#!/usr/bin/env python3
"""Validate the archived finite-U full-k gate summaries."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load(relative: str) -> dict:
    with (ROOT / relative).open("r", encoding="utf-8") as handle:
        return json.load(handle)


contour = load(
    "exact_zavg4_contour_u00025_s12_n64/"
    "fullk_nrg_contour_pilot_summary.json"
)
assert contour["status"] == "COMPLETE_BASIS_TOPOLOGY_PASS"
certificate = contour["certificate"]
assert abs(abs(certificate["winding"]) - 1.0) < 0.05
assert certificate["endpoint_permutation"] == [1, 0]
assert abs(abs(certificate["half_winding"]) - 0.5) < 0.05
assert certificate["minimum_boundary_modulus"] > 1.0e-12
assert certificate["maximum_phase_increment"] < 0.5 * 3.141592653589793
assert certificate["minimum_F_zz_modulus"] > 1.0e-4
assert certificate["maximum_F_z_residual"] < 1.0e-7
assert certificate["maximum_pole_residual"] < 1.0e-7
assert certificate["maximum_assignment_to_half_separation_ratio"] < 0.5
assert certificate["maximum_sumrule_error"] < 1.0e-10
assert certificate["ground_charges"] == [2]

eom = load(
    "exact_eom_identity_u00025_s12_z4/exact_eom_identity_summary.json"
)
assert eom["status"] == "PASS"
assert eom["maximum_absolute_residual"] < 1.0e-12
assert eom["maximum_relative_residual"] < 1.0e-8
assert eom["maximum_sumrule_error"] < 1.0e-10

paths = [
    "exact_zavg4_predict_u0001_s12_i7/fullk_nrg_ep_predictor_summary.json",
    "exact_zavg4_predict_u00015_s12_i7/fullk_nrg_ep_predictor_summary.json",
    "exact_zavg4_predict_u0002_s12_i7/fullk_nrg_ep_predictor_summary.json",
    "exact_zavg4_predict_u00025_cont_s12_i7/fullk_nrg_ep_predictor_summary.json",
]
centers = [load(path) for path in paths]
assert all(item["status"] in {"PILOT_PASS", "CENTER_PASS"} for item in centers)
interactions = [item["parameters"]["U"] for item in centers]
assert interactions == [0.001, 0.0015, 0.002, 0.0025]

coarse = centers[-1]["predicted_center"]
fine = load(
    "exact_zavg8_low_predict_u00025_s12_refine_i3/"
    "fullk_nrg_ep_predictor_summary.json"
)["predicted_center"]
relative_shifts = [
    abs(fine["z"][0] - coarse["z"][0]) / abs(coarse["z"][0]),
    abs(fine["z"][1] - coarse["z"][1]) / abs(coarse["z"][1]),
    abs(fine["epsilon_d"] - coarse["epsilon_d"]) / abs(coarse["epsilon_d"]),
    abs(fine["W"] - coarse["W"]) / abs(coarse["W"]),
]
assert relative_shifts[0] < 3.0e-4
assert relative_shifts[1] < 8.0e-3
assert relative_shifts[2] < 2.0e-3
assert relative_shifts[3] < 4.0e-3

kept = load(
    "kept_exact_calibration_u00025_s12_z4/kept_against_exact_summary.json"
)
assert kept["status"] == "CALIBRATION_COMPLETE"
reports = {row["nkeep"]: row["maximum_errors"] for row in kept["reports"]}
assert reports[960]["Sigma"] < 0.05
assert reports[960]["sumrule"] < 0.02

print("PASS: archived finite-U full-k topology and convergence gates")
print(f"winding={certificate['winding']:.15f}")
print(f"half_winding={certificate['half_winding']:.15f}")
print(
    "Nz=4->8 relative center shifts="
    + ", ".join(f"{100.0 * value:.3f}%" for value in relative_shifts)
)
print(
    "Nkeep=960 max errors: "
    f"Sigma={100.0 * reports[960]['Sigma']:.2f}%, "
    f"sumrule={100.0 * reports[960]['sumrule']:.2f}%"
)
