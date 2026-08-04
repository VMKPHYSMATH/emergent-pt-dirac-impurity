#!/usr/bin/env python3
"""Numerical checks for the finite-U contact algebra quoted in the paper."""

import numpy as np


I2 = np.eye(2, dtype=complex)
I4 = np.eye(4, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
P = np.array(
    [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=complex,
)


def rational_r(x):
    """Normalized rational R(x)=(x I+i P)/(x+i)."""
    return (x * I4 + 1j * P) / (x + 1j)


def embed_two(op, first, second):
    """Embed a two-qubit operator in a three-qubit tensor product."""
    out = np.zeros((8, 8), dtype=complex)
    for col in range(8):
        bits = [(col >> (2 - k)) & 1 for k in range(3)]
        local_col = 2 * bits[first] + bits[second]
        for local_row in range(4):
            amp = op[local_row, local_col]
            if amp == 0:
                continue
            new_bits = bits.copy()
            new_bits[first] = local_row // 2
            new_bits[second] = local_row % 2
            row = 4 * new_bits[0] + 2 * new_bits[1] + new_bits[2]
            out[row, col] += amp
    return out


def residual(a, b):
    return np.linalg.norm(a - b, ord="fro")


def main():
    # Three-particle Yang--Baxter equation.
    u, v, w = 0.37 - 0.12j, -0.41 + 0.08j, 0.19 + 0.23j
    r12 = embed_two(rational_r(u - v), 0, 1)
    r13 = embed_two(rational_r(u - w), 0, 2)
    r23 = embed_two(rational_r(v - w), 1, 2)
    ybe_error = residual(r12 @ r13 @ r23, r23 @ r13 @ r12)

    # GL(2,C) covariance, including non-unitary matrices.
    s = np.array([[1.1 + 0.2j, 0.3 - 0.1j], [0.2j, 0.8 - 0.15j]])
    ss = np.kron(s, s)
    covariance_error = np.linalg.norm(
        np.linalg.inv(ss) @ rational_r(0.63 - 0.17j) @ ss
        - rational_r(0.63 - 0.17j),
        ord="fro",
    )

    # Off-EP CPT/metric diagonalization.
    d_value, g_value = 0.70, 0.30
    theta = np.arctanh(g_value / d_value)
    s_cpt = np.cosh(theta / 2) * I2 + np.sinh(theta / 2) * SY
    m = d_value * SZ + 1j * g_value * SX
    split = np.sqrt(d_value**2 - g_value**2)
    diagonalization_error = residual(
        np.linalg.inv(s_cpt) @ m @ s_cpt, split * SZ
    )

    # Dressed-rapidity identity and singlet/triplet eigenvalues.
    p, q = 0.83, -0.27
    epsilon_d, interaction_u, gamma_a = -0.61, 1.74, 0.23

    def bfun(momentum):
        return momentum * (momentum - 2 * epsilon_d - interaction_u)

    delta_b = bfun(p) - bfun(q)
    factorization_error = abs(
        delta_b - (p - q) * (p + q - 2 * epsilon_d - interaction_u)
    )
    x = delta_b / (2 * interaction_u * gamma_a)
    r = rational_r(x)
    triplet = np.array([1, 0, 0, 0], dtype=complex)
    singlet = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
    expected_singlet = (x - 1j) / (x + 1j)
    triplet_error = np.linalg.norm(r @ triplet - triplet)
    singlet_error = np.linalg.norm(r @ singlet - expected_singlet * singlet)

    checks = {
        "YBE residual": ybe_error,
        "GL(2,C) covariance residual": covariance_error,
        "CPT diagonalization residual": diagonalization_error,
        "dressed-rapidity factorization residual": factorization_error,
        "triplet eigenvalue residual": triplet_error,
        "singlet eigenvalue residual": singlet_error,
    }
    for name, value in checks.items():
        print(f"{name}: {value:.3e}")

    tolerance = 1.0e-12
    if max(checks.values()) > tolerance:
        raise SystemExit("contact-algebra validation failed")
    print("all finite-U contact-algebra checks passed")


if __name__ == "__main__":
    main()
