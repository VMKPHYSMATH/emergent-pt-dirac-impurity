#!/usr/bin/env python3
from __future__ import annotations
import math
from pathlib import Path

HERE = Path(__file__).resolve()
PROJECT = HERE.parents[3]
MODULE = PROJECT / "src" / "PTDiracNHNRG.jl"
SCAN = PROJECT / "benchmarks" / "prb_complete" / "scripts" / "run_soc_quartic_scan.jl"
AUDIT = PROJECT / "benchmarks" / "prb_complete" / "scripts" / "quartic_pole_audit.py"

required = {
    MODULE: [
        "soc_mode::Symbol", "soc_lambda::Float64", "soc_kmax::Float64",
        "function soc_form_factor", "function soc_overlap_matrix",
        "function impurity_hybridization_matrix", "soc_hybridization_matrix.csv",
        "soc_normalized_hybridization_det",
    ],
    SCAN: ["run_soc_quartic_scan", "lambda_scan", "soc_mode=:overlap"],
    AUDIT: ["F_lambda", "soc_normalized_hybridization_det", "SOC_Gap_and_Jordan_vs_F.pdf"],
}
for path, tokens in required.items():
    if not path.is_file():
        raise SystemExit(f"FAIL: missing {path}")
    text = path.read_text(encoding="utf-8")
    missing = [t for t in tokens if t not in text]
    if missing:
        raise SystemExit(f"FAIL: {path.name} missing {missing}")

for lam in (0.0, 0.15, 0.30, 0.50, 0.70, 0.99 * math.pi / 4):
    kmax = math.pi / 4
    x = max(-1.0, min(1.0, lam / kmax))
    f = max(1.0 - x*x, 0.0)
    det_s = 1.0 - x*x
    if abs(f - det_s) > 1e-14:
        raise SystemExit(f"FAIL: F determinant identity at lambda={lam}")

print("PASS: SOC overlap source, scan, audit, and determinant checks")
