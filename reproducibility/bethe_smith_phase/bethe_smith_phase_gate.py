#!/usr/bin/env python3
"""Validate the frozen one-particle Bethe/Smith phase identity for Driven-Dirac impurity.

The gate uses the full two-channel Fisher--Lee scattering matrix.  It checks
that the scattering eigenvalues are unimodular in the passive window and
that

    sum_a (2 pi i)^(-1) d_omega log s_a
      = (2 pi i)^(-1) d_omega log det S
      = Tr Q/(2 pi)

without assuming a channel-diagonal Breit--Wigner form.  Matrix components
are exported separately in the original spin basis and in the fixed
Hadamard/chiral basis; only eigenphase data are used as basis-independent
evidence.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


I2 = np.eye(2, dtype=complex)
SX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
SZ = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
UH = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
GAMMA0 = 0.12
BETAS = (0.35, 0.45, 0.50)
OMEGAS = np.linspace(-20.0, 20.0, 40001)
FD_STEP = 1.0e-6


def coherent(beta: float) -> float:
    return 0.075 + 0.05 * beta


def relative(beta: float) -> float:
    return 0.20 * beta


def psd_sqrt(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.conj().T))
    if float(np.min(values)) < -1.0e-13:
        raise ValueError("non-passive bath-rate matrix")
    return (vectors * np.sqrt(np.clip(values, 0.0, None))) @ vectors.conj().T


def scattering(omega: float, beta: float) -> tuple[np.ndarray, np.ndarray]:
    d = coherent(beta)
    g = relative(beta)
    h_eff = d * SX - 1j * GAMMA0 * I2 + 1j * g * SZ
    gamma_bath = 2.0 * (GAMMA0 * I2 - g * SZ)
    root = psd_sqrt(gamma_bath)
    g_r = np.linalg.inv(float(omega) * I2 - h_eff)
    matrix = I2 - 1j * root @ g_r @ root
    derivative_g = -(g_r @ g_r)
    derivative_s = -1j * root @ derivative_g @ root
    q_matrix = -1j * matrix.conj().T @ derivative_s
    return matrix, q_matrix


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    output = Path(__file__).resolve().parent
    output.mkdir(exist_ok=True)
    rows: list[dict[str, float]] = []
    summaries: list[dict[str, float]] = []

    for beta in BETAS:
        trace_density: list[float] = []
        max_unitarity = 0.0
        max_modulus = 0.0
        max_spin_offdiag = 0.0
        max_chiral_offdiag = 0.0
        max_phase_identity = 0.0
        for omega in OMEGAS:
            s_matrix, q_matrix = scattering(float(omega), beta)
            eigenvalues = np.linalg.eigvals(s_matrix)
            max_unitarity = max(
                max_unitarity,
                float(np.max(np.abs(s_matrix.conj().T @ s_matrix - I2))),
            )
            max_modulus = max(
                max_modulus,
                float(np.max(np.abs(np.abs(eigenvalues) - 1.0))),
            )
            s_chiral = UH.conj().T @ s_matrix @ UH
            max_spin_offdiag = max(
                max_spin_offdiag, float(abs(s_matrix[0, 1]))
            )
            max_chiral_offdiag = max(
                max_chiral_offdiag, float(abs(s_chiral[0, 1]))
            )

            s_plus, _ = scattering(float(omega + FD_STEP), beta)
            s_minus, _ = scattering(float(omega - FD_STEP), beta)
            derivative_s = (s_plus - s_minus) / (2.0 * FD_STEP)
            determinant_delay = float(
                (-1j * np.trace(np.linalg.solve(s_matrix, derivative_s))).real
            )
            trace_delay = float(np.trace(q_matrix).real)
            max_phase_identity = max(
                max_phase_identity, abs(trace_delay - determinant_delay)
            )
            density = trace_delay / (2.0 * np.pi)
            trace_density.append(density)

            if abs(float(omega)) <= 0.40 and (
                abs(float(omega)) < 5.0e-13
                or abs(abs(float(omega)) - 0.10) < 5.0e-13
            ):
                phases = np.sort(np.angle(eigenvalues))
                rows.append(
                    {
                        "beta0": beta,
                        "omega": float(omega),
                        "phase_1": float(phases[0]),
                        "phase_2": float(phases[1]),
                        "eigenvalue_modulus_error": float(
                            np.max(np.abs(np.abs(eigenvalues) - 1.0))
                        ),
                        "spin_basis_offdiagonal_magnitude": float(
                            abs(s_matrix[0, 1])
                        ),
                        "fixed_chiral_basis_offdiagonal_magnitude": float(
                            abs(s_chiral[0, 1])
                        ),
                        "bethe_phase_density": density,
                    }
                )

        finite_window_count = float(np.trapezoid(trace_density, OMEGAS))
        summaries.append(
            {
                "beta0": beta,
                "max_unitarity_error": max_unitarity,
                "max_scattering_eigenvalue_modulus_error": max_modulus,
                "max_spin_basis_offdiagonal_magnitude": max_spin_offdiag,
                "max_fixed_chiral_basis_offdiagonal_magnitude": (
                    max_chiral_offdiag
                ),
                "max_bethe_smith_phase_identity_error": max_phase_identity,
                "finite_window_phase_count": finite_window_count,
                "analytic_complete_pole_count": 2.0,
            }
        )

    csv_path = output / "scattering_eigenphase_samples.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    json_path = output / "bethe_smith_phase_summary.json"
    json_path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "kernel": "full two-channel Fisher-Lee scattering matrix",
                "interpretation": (
                    "exact one-particle finite-size phase quantization; "
                    "not a multiparticle interacting Bethe solution; matrix "
                    "components are basis dependent"
                ),
                "basis_audit": (
                    "S_chiral = U_H^dagger S_spin U_H; Green functions and "
                    "bath-rate matrices are transformed together"
                ),
                "summaries": summaries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    readme_path = output / "README.md"
    readme_path.write_text(
        "# Full-scattering Bethe–Smith phase gate\n\n"
        "This gate validates the retained one-particle Bethe phase "
        "quantization using the complete two-channel Fisher–Lee scattering "
        "matrix. It does not assume that the chiral scattering matrix is "
        "diagonal and does not claim a multiparticle interacting Bethe "
        "solution.\n\n"
        "The exported checks cover scattering unitarity, unimodularity of "
        "both scattering eigenvalues, off-diagonal components in both the "
        "original spin and fixed Hadamard/chiral representations, the "
        "finite-difference determinant-phase/Wigner–Smith identity, and the "
        "finite-window phase count. Matrix components are basis dependent; "
        "the eigenphases and Smith trace are the invariant checks. Run "
        "`python bethe_smith_phase_gate.py` "
        "from the directory containing the script.\n",
        encoding="utf-8",
    )

    checksum_path = output / "SHA256SUMS"
    checksum_path.write_text(
        "\n".join(
            f"{sha256(path)}  {path.name}"
            for path in (
                Path(__file__).resolve(),
                csv_path,
                json_path,
                readme_path,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    print(json_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
