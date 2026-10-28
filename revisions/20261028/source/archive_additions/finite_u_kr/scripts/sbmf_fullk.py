from pathlib import Path
"""U -> infinity slave-boson (Coleman, Q=1) saddle for the stationary rotating-drive
full-k kernel of the Dirac-impurity paper (SM App. "Stationary rotating drive").

At the saddle the spinon kernel is the U=0 kernel with W -> rW and eps_d -> eps_t = eps_d + delta_c:
    G_f^{-1}(z) = (z - eps_t) I - Sigma_d(z; W'=rW, V_O).
Saddle equations (T=0, mu=0):
    r^2 + n_f = 1,                n_f = -(1/pi) int_{-inf}^0 Im Tr G_f
    delta_c = -<H_hyb>/(2 r^2),   <H_hyb> = -(2/pi) int_{-inf}^0 Im Tr[Sigma_f G_f]
Both integrals are rotated to the imaginary axis (bound states included automatically)."""
import math
import sys
from dataclasses import replace

import numpy as np
from scipy.integrate import quad
from scipy.optimize import root

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fullk_cert as C  # noqa: E402

I2 = np.eye(2, dtype=complex)
P0 = replace(C.P, soc=0.05, impurity_hybridization=0.10)
EP0 = dict(z=0.011972834346 - 0.002807216274j, eps=0.015170349783, VO=1.213422009614)


def pW(Wp):
    return replace(P0, impurity_hybridization=float(Wp))


def solve_ep(Wp, guess):
    """U=0 nonlinear pole EP at hybridization W' (fixed lambda, eps_O, Omega): unknowns Re z, Im z, V_O."""
    p = pW(Wp)
    res = root(lambda x: C.reduced_residual(x, p, 2e-6), guess, tol=1e-13)
    d = C.reduced_data(res.x, p, 2e-6)
    z, eps, VO = complex(d["z"]), complex(d["epsilon"]), float(d["auxiliary_vertex"])
    F = C.determinant(z, eps.real, VO, p)
    Fz = C.derivative(lambda a: C.determinant(a, eps.real, VO, p), z, 2e-6)
    return dict(ok=res.success, z=z, eps=eps.real, eps_im=eps.imag, VO=VO, F=abs(F), Fz=abs(Fz), x=res.x)


def G_f(z, eps_t, VO, p):
    S = C.analytic_physical_self_energy(z, VO, p)
    return np.linalg.inv((z - eps_t) * I2 - S), S


def _iy_integral(fun, scale=0.01):
    """int_0^inf fun(y) dy with y = scale * t/(1-t)."""
    def g(t):
        y = scale * t / (1.0 - t)
        return fun(y) * scale / (1.0 - t) ** 2
    pts = [scale * x for x in (0.05, 0.2, 0.5, 1.0, 2.0, 5.0)]
    tp = [y / (y + scale) for y in pts]
    val, err = quad(g, 0.0, 1.0, points=tp, limit=400, epsabs=1e-13, epsrel=1e-11)
    return val


def occupations_T0(eps_t, VO, p):
    """n_f and <H_hyb> at T=0, mu=0, from the imaginary axis."""
    nf = 1.0 + _iy_integral(lambda y: np.trace(G_f(1j * y, eps_t, VO, p)[0]).real) / math.pi

    def hyb(y):
        G, S = G_f(1j * y, eps_t, VO, p)
        return np.trace(S @ G).real
    Hh = 2.0 * _iy_integral(hyb) / math.pi
    return nf, Hh


def occupations_T(eps_t, VO, p, T, nmax=200000):
    """Matsubara version at temperature T (tail of Re Tr G ~ -c/w^2 handled analytically)."""
    n = np.arange(nmax)
    w = math.pi * T * (2 * n + 1)
    trG = np.empty(nmax)
    trSG = np.empty(nmax)
    for k, wk in enumerate(w):
        G, S = G_f(1j * wk, eps_t, VO, p)
        trG[k] = np.trace(G).real
        trSG[k] = np.trace(S @ G).real
    # tail: Re Tr G ~ c2/w^2 ; estimate c2 from last points, add sum_{n>=nmax} 1/w_n^2
    c2 = trG[-1] * w[-1] ** 2
    tail = c2 / (math.pi * T) ** 2 * (1.0 / (4.0 * (nmax - 0.5)))  # sum_{n>=N} 1/(2n+1)^2 ~ 1/(4(N-1/2))
    nf = 1.0 + 2.0 * T * (trG.sum() + tail)
    Hh = 2.0 * 2.0 * T * trSG.sum()
    return nf, Hh


if __name__ == "__main__":
    np.set_printoptions(precision=6)
    # 1. reproduce the U=0 certificate point
    ep = solve_ep(0.10, np.array([EP0["z"].real, EP0["z"].imag, EP0["VO"]]))
    print("1. U=0 EP at W=0.10:", "z=%.12f%+.12fi eps=%.12f VO=%.12f |F|=%.1e |Fz|=%.1e Im eps=%.1e"
          % (ep["z"].real, ep["z"].imag, ep["eps"], ep["VO"], ep["F"], ep["Fz"], ep["eps_im"]))
    # 2. physical sheet off the real axis: analytic vs quadrature
    for z in (0.01 + 0.05j, 1e-3j, 0.3j, 2.0j, -0.02 + 1e-3j):
        a = C.analytic_physical_self_energy(z, EP0["VO"], P0)
        q = C.quadrature_physical_self_energy(z, EP0["VO"], P0)
        print("2. z=%s  |analytic - quadrature| / |Sigma| = %.2e" % (z, np.linalg.norm(a - q) / np.linalg.norm(q)))
    # 3. occupations at the U=0 EP point (W'=0.10): T=0 imaginary axis vs Matsubara
    nf0, H0 = occupations_T0(EP0["eps"], EP0["VO"], P0)
    print("3. T=0: n_f=%.8f  <H_hyb>=%.8e" % (nf0, H0))
    for T in (1e-3, 3e-4):
        nfT, HT = occupations_T(EP0["eps"], EP0["VO"], P0, T, nmax=int(4000 / T * 1e-3) + 20000)
        print("   T=%.0e (Matsubara): n_f=%.8f  <H_hyb>=%.8e" % (T, nfT, HT))
    # Lorentzian estimate from the EP pole alone: Tr G ~ 2/(z - z_EP)
    zE = EP0["z"]
    print("   pole-only estimate 2[1/2 - arctan(Re z/|Im z|)/pi] = %.6f" % (2 * (0.5 - math.atan(zE.real / abs(zE.imag)) / math.pi)))
