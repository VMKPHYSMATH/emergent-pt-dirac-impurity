#!/usr/bin/env python3
"""Retarded real-frequency splitting diagnostic, not a PT phase diagram.
Controls are fixed in the recentered B_raw(z)-B_raw(0) bath convention.
The first scalar reference root is not identified with a complex resonance.
"""
from __future__ import annotations

import os
import sys
import warnings
from dataclasses import replace

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import full_k_two_cut_ep_certificate as cert  # noqa: E402

ETA = 1e-10  # upper-rim retarded boundary value
P = cert.P  # lambda = 0.10, W = 0.10, K = pi/4, eps_O = 120, Omega = 60
EP = cert.solve_ep(P)
EPS = float(EP["epsilon"][0])
VTX = float(EP["auxiliary_vertex"])


def components(omega: float, p) -> np.ndarray:
    return cert.pauli_components(cert.analytic_physical_self_energy(complex(omega, ETA), VTX, p))


def omega_res(p, lo: float = 1e-4, hi: float = 0.60, n: int = 220):
    f = lambda o: (o - EPS - components(o, p)[0]).real  # noqa: E731
    xs = np.linspace(lo, hi, n)
    vs = np.array([f(x) for x in xs])
    k = np.flatnonzero(vs[:-1] * vs[1:] < 0)
    if len(k) == 0:
        return None
    return brentq(f, xs[k[0]], xs[k[0] + 1], xtol=1e-13)


def cell(lam: float, om: float):
    p = replace(P, soc=float(lam), drive_frequency=float(om))
    o = omega_res(p)
    if o is None:
        return None
    _, jx, jz = components(o, p)
    nch = len(cert.open_channels(float(o), VTX, p))
    sig = cert.analytic_physical_self_energy(complex(o, ETA), VTX, p)
    width = -(sig-sig.conj().T)/(2j)
    return o, jx, jz, nch, float(np.linalg.eigvalsh(width).min())


def threshold(lam: float, lo: float = 40.0, hi: float = 90.0):
    """Omega at which the number of open channels at omega_res changes."""
    nl, nh = cell(lam, lo)[3], cell(lam, hi)[3]
    if nl == nh:
        return np.nan
    for _ in range(26):
        mid = 0.5 * (lo + hi)
        if cell(lam, mid)[3] == nl:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main(out: str) -> None:
    lams = np.linspace(0.015, 0.32, 120)
    oms = np.linspace(4.0, 114.0, 118)
    shape = (len(lams), len(oms))
    ARG = np.full(shape, np.nan)
    MAG = np.full(shape, np.nan)
    RES = np.full(shape, np.nan)
    JX = np.full(shape, np.nan + 0j)
    JZ = np.full(shape, np.nan + 0j)
    NCH = np.full(shape, -1)
    MIN_WIDTH = np.full(shape, np.nan)
    for i, lam in enumerate(lams):
        for j, om in enumerate(oms):
            c = cell(lam, om)
            if c is None:
                continue
            o, jx, jz, nch, width = c
            MIN_WIDTH[i,j] = width
            s2 = jx**2 + jz**2
            ARG[i, j] = np.degrees(np.angle(s2))
            MAG[i, j] = abs(s2)
            RES[i, j] = o
            JX[i, j], JZ[i, j], NCH[i, j] = jx, jz, nch
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(lams)} rows", flush=True)
    th_lams = lams[::4]
    th = np.array([[lam, threshold(lam)] for lam in th_lams])
    np.savez(out, lams=lams, Oms=oms, ARG=ARG, MAG=MAG, RES=RES, JX=JX, JZ=JZ, NCH=NCH,
             thresh=th, eps=EPS, vtx=VTX, eta=ETA, min_halfwidth=MIN_WIDTH,
             cosine_phase=np.real(JX**2+JZ**2)/MAG)
    assert np.nanmin(MIN_WIDTH) > -1e-9, "Non-passive sampled kernel"
    print(f"fixed controls: eps_d = {EPS:.9f}, V_O = {VTX:.9f}")
    print(f"resolved {np.isfinite(ARG).sum()} / {ARG.size} cells; wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "splitting_diagnostic_data.npz")
