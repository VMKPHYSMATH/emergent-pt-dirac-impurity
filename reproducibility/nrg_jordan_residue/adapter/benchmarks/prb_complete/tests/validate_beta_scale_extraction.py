#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "benchmarks" / "prb_complete" / "scripts" / "extract_beta_scales.py"
CONFIG = ROOT / "benchmarks" / "prb_complete" / "config" / "beta_scale_scan.toml"


def ccols(prefix: str, matrix: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in zip(("11", "12", "21", "22"), matrix.ravel()):
        out[f"{prefix}{key}_real"] = float(np.real(value))
        out[f"{prefix}{key}_imag"] = float(np.imag(value))
    return out


def make_run(root: Path, beta: float) -> None:
    run = root / "runs" / f"beta_{beta:.6f}"
    run.mkdir(parents=True)
    delta = 0.075 + 0.05 * beta
    gamma = 0.20 * beta
    (run / "RUN_SUMMARY.txt").write_text(
        "\n".join([
            "PT-DIRAC NON-HERMITIAN NRG ADAPTER",
            f"beta0={beta}",
            "U=2.0; eps_d=-1.0",
            f"Delta_eff={delta}; Gamma_PT={gamma}; Delta_coh=0.001",
            "V=0.2; Gamma_edge=0.12",
        ]) + "\n", encoding="utf-8"
    )

    flow_rows = []
    for n in range(9):
        scale = 3.0 ** (-n / 2)
        parity = n % 2
        correction = 0.4 * (0.18 ** n)
        for dq in (-1, 0, 1):
            for ordinal in range(4):
                base = 0.15 * ordinal + 0.04 * abs(dq) + 0.01 * parity
                e = complex(base + correction * (ordinal + 1), 0.002 * dq)
                flow_rows.append({
                    "iteration": n, "scale": scale, "kept": 100,
                    "ground_charge": 1, "charge": 1 + dq, "ordinal": ordinal,
                    "energy_real": e.real, "energy_imag": e.imag,
                    "energy_abs": abs(e), "max_residual": 1e-10,
                    "biorth_error": 1e-10, "min_pair_overlap": 0.9,
                })
    pd.DataFrame(flow_rows).to_csv(run / "complex_level_flow.csv", index=False)

    residue_rows = []
    N = np.array([[delta, 1j * gamma], [1j * gamma, -delta]], dtype=complex)
    N /= np.linalg.norm(N)
    for n in range(9):
        scale = 3.0 ** (-n / 2)
        for sector in (-1, 1):
            s = (0.018 + 0.012 * abs(beta - 0.5)) * (1.0 + 0.01 * sector)
            center = 0.002j * sector
            zminus, zplus = center - s, center + s
            B = (8e-4 + 1e-4 * beta) * N
            A = 0.08 * np.eye(2, dtype=complex)
            Wdiff = B / s
            Wminus = 0.5 * (A - Wdiff)
            Wplus = 0.5 * (A + Wdiff)
            for rank, (z, W) in enumerate(((zminus, Wminus), (zplus, Wplus))):
                add = W if sector == 1 else np.zeros((2, 2), complex)
                rem = W if sector == -1 else np.zeros((2, 2), complex)
                row = {
                    "iteration": n, "scale": scale, "ground_charge": 1,
                    "charge": 1 + sector, "matrix_rank": rank,
                    "energy_real": z.real / scale, "energy_imag": z.imag / scale,
                    "pole_real": z.real, "pole_imag": z.imag,
                    "matrix_support_abs": float(np.sum(np.abs(W))),
                    "max_residual": 1e-10, "biorth_error": 1e-10,
                    "min_pair_overlap": 0.9,
                }
                row.update(ccols("add", add))
                row.update(ccols("rem", rem))
                residue_rows.append(row)
    pd.DataFrame(residue_rows).to_csv(run / "impurity_transition_residue_matrix.csv", index=False)

    omega = np.linspace(-0.5, 0.5, 1001)
    gamma_tr = 0.05
    gamma_j = 0.08
    Nraw = np.array([[delta, 1j * gamma], [1j * gamma, -delta]], dtype=complex)
    Nnorm = np.linalg.norm(Nraw)
    rows = []
    for w in omega:
        G = np.eye(2, dtype=complex) / (w + 1j * gamma_tr)
        if Nnorm > 0:
            G += 0.2 * (Nraw / Nnorm) / (w + 1j * gamma_j)
        jp = np.vdot(Nraw, G) / Nnorm if Nnorm > 0 else 0j
        rows.append({
            "omega": w,
            "ReG11": G[0, 0].real, "ImG11": G[0, 0].imag,
            "ReG12": G[0, 1].real, "ImG12": G[0, 1].imag,
            "ReG21": G[1, 0].real, "ImG21": G[1, 0].imag,
            "ReG22": G[1, 1].real, "ImG22": G[1, 1].imag,
            "minus_ImTrG_over_pi": -np.imag(np.trace(G)) / np.pi,
            "minus_ImJordanG_over_pi": -np.imag(jp) / np.pi,
        })
    pd.DataFrame(rows).to_csv(run / "kept_space_lehmann.csv", index=False)
    pd.DataFrame([
        {"alpha": a, "beta": b, "weight_real": 1.0 if a == b else 0.0,
         "weight_imag": 0.0, "target": 1.0 if a == b else 0.0,
         "abs_error": 0.01}
        for a in (1, 2) for b in (1, 2)
    ]).to_csv(run / "lehmann_sumrule.csv", index=False)


def main() -> None:
    source = (ROOT / "src" / "PTDiracNHNRG.jl").read_text(encoding="utf-8")
    assert "minus_ImJordanG_over_pi" in source, "core Lehmann-column patch is missing"
    with tempfile.TemporaryDirectory(prefix="ptdirac_beta_scale_test_") as tmp:
        scan = Path(tmp) / "scan"
        for beta in (0.30, 0.50, 0.70):
            make_run(scan, beta)
        out = scan / "analysis"
        subprocess.run([
            sys.executable, str(SCRIPT), str(scan),
            "--scan-config", str(CONFIG), "--out", str(out),
        ], check=True, cwd=ROOT)
        summary = pd.read_csv(out / "beta_scale_summary.csv")
        assert len(summary) == 3
        assert np.isfinite(summary["T_flow"]).all()
        assert np.isfinite(summary["T_pair_split_n5"]).all()
        assert np.allclose(summary["T_trace_HWHM"], 0.05, atol=0.004)
        assert np.allclose(summary["T_J_HWHM"], 0.08, atol=0.006)
        gates = json.loads((out / "beta_scale_gate_summary.json").read_text())
        assert abs(float(gates["beta_EP"]) - 0.5) < 1e-10
        assert gates["thermodynamic_TK_extracted"] is False
    print("SELF-TEST PASS: beta0 operational-scale extraction")


if __name__ == "__main__":
    main()
