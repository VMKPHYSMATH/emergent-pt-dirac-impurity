#!/usr/bin/env python3
"""Independent small-system checks for the Julia adapter.

This does not execute Julia. It validates the local Fock algebra, the core
single-particle spectrum, the logarithmic-star normalization, and the
biorthogonal Lehmann residue/sum-rule convention on a complete finite model.
"""
from __future__ import annotations

import math
import numpy as np


def annihilation(norb: int, orb: int) -> np.ndarray:
    dim = 1 << norb
    out = np.zeros((dim, dim), complex)
    bit = 1 << orb
    lower = bit - 1
    for ket in range(dim):
        if ket & bit:
            bra = ket ^ bit
            sign = -1.0 if ((ket & lower).bit_count() % 2) else 1.0
            out[bra, ket] = sign
    return out


def lr_eig(h: np.ndarray):
    ev, r = np.linalg.eig(h)
    evl, lraw = np.linalg.eig(h.conj().T)
    unused = list(range(len(evl)))
    l = np.zeros_like(lraw)
    for j, e in enumerate(ev):
        k = min(unused, key=lambda x: abs(evl[x] - e.conjugate()))
        unused.remove(k)
        l[:, j] = lraw[:, k]
        s = np.vdot(l[:, j], r[:, j])
        root = np.sqrt(s)
        r[:, j] /= root
        l[:, j] /= root.conjugate()
    return ev, l, r


def test_local_algebra() -> None:
    d = [annihilation(2, 0), annihilation(2, 1)]
    eye = np.eye(4)
    for a in range(2):
        for b in range(2):
            anti = d[a] @ d[b].conj().T + d[b].conj().T @ d[a]
            target = eye if a == b else np.zeros_like(eye)
            assert np.max(np.abs(anti - target)) < 1e-14


def test_core_spectrum() -> None:
    beta = 0.49
    delta = 0.075 + 0.05 * beta
    gamma = 0.20 * beta
    coh = 1e-3
    h = np.array([[delta, coh + 1j * gamma],
                  [coh + 1j * gamma, -delta]], complex)
    ev = np.linalg.eigvals(h)
    target = np.sqrt(delta**2 + (coh + 1j * gamma)**2)
    assert np.min(np.abs(ev - target)) < 1e-12
    assert np.min(np.abs(ev + target)) < 1e-12


def logarithmic_star(lam=3.0, z=0.5, r=1.0, d=math.pi / 4, n=100):
    energies, weights = [], []
    for i in range(n):
        if i == 0:
            b, a = 1.0, lam ** (-z)
        else:
            b = lam ** (-(i - 1 + z))
            a = lam ** (-(i + z))
        w = 0.5 * (b ** (r + 1) - a ** (r + 1))
        xi = d * (r + 1) / (r + 2) * (
            (b ** (r + 2) - a ** (r + 2)) /
            (b ** (r + 1) - a ** (r + 1))
        )
        energies += [xi, -xi]
        weights += [w, w]
    weights = np.asarray(weights)
    weights /= weights.sum()
    return np.asarray(energies), weights


def test_star() -> None:
    e, w = logarithmic_star()
    assert abs(w.sum() - 1.0) < 1e-14
    assert abs(np.dot(w, e)) < 1e-14
    assert np.all(w > 0)


def test_complete_lehmann_sumrule() -> None:
    # Two-orbital non-Hermitian impurity only. Complete Fock space means the
    # residue sum must equal the canonical anticommutator exactly.
    d = [annihilation(2, 0), annihilation(2, 1)]
    n = [x.conj().T @ x for x in d]
    eps, u = -1.0, 2.0
    delta, gamma, coh = 0.0995, 0.098, 1e-3
    h = eps * (n[0] + n[1]) + u * (n[0] @ n[1])
    h += delta * (n[0] - n[1])
    h += (coh + 1j * gamma) * (d[0].conj().T @ d[1] + d[1].conj().T @ d[0])
    ev, l, r = lr_eig(h)
    charges = np.array([0, 1, 1, 2])
    g = min(range(4), key=lambda j: (ev[j].real, abs(ev[j].imag)))
    q0 = charges[np.argmax(np.abs(r[:, g]))]  # exact sectors are unmixed in Q
    dann = [l.conj().T @ x @ r for x in d]
    dcre = [l.conj().T @ x.conj().T @ r for x in d]
    # Infer eigenstate charge from expectation of total number.
    ntot = n[0] + n[1]
    nrep = l.conj().T @ ntot @ r
    qev = np.rint(np.real(np.diag(nrep))).astype(int)
    q0 = qev[g]
    weight = np.zeros((2, 2), complex)
    for a in range(2):
        for b in range(2):
            for m in np.where(qev == q0 + 1)[0]:
                weight[a, b] += dann[a][g, m] * dcre[b][m, g]
            for m in np.where(qev == q0 - 1)[0]:
                weight[a, b] += dcre[b][g, m] * dann[a][m, g]
    assert np.max(np.abs(weight - np.eye(2))) < 1e-10, weight


def main() -> None:
    test_local_algebra()
    test_core_spectrum()
    test_star()
    test_complete_lehmann_sumrule()
    print("PASS: local algebra, core spectrum, star normalization, and complete Lehmann sum rule")


if __name__ == "__main__":
    main()
