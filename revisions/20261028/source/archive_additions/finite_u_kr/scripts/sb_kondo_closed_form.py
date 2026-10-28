"""Closed-form U=inf slave-boson (Coleman, Q=1) saddle for the static constant-core model.

Spinon kernel (bath counted once, flat Markov kernel, band edge -D):
    G^R(w)^-1 = (w - et + i r^2 G0) I - r^2 (D0 sx + i Gp sz),   poles z_+- = et - i r^2 G0 +- r^2 s0,
    s0^2 = D0^2 - Gp^2  (EP at s0 = 0).
All integrands are partial fractions in 1/(w - z_+-), so with L(z) = int_{-D}^{inf} f(w) dw/(w - z):
    n_f      = -(1/pi) Im[L(z+) + L(z-)]
    delta_c  = -(G0/pi) Re[L(z+) + L(z-)] + (1/pi) Im[s0 (L(z+) - L(z-))]      (r-stationarity)
    T = 0  : L(z) = ln(-z) - ln(-D - z)
    T > 0  : L(z) = psi(1/2 + i z/(2 pi T)) - ln(D/(2 pi T)) - i pi/2            (D >> T,|z|)
Kondo limit closed form (et -> 0):  r^2 G0 sqrt(1+u) = T_K0 exp[F(u)],  u = s0^2/G0^2,
    T_K0 = D exp(-pi|eps|/(2 G0)),  F(u) = sqrt(u) atan(sqrt(u)) (u>0),  -sqrt(-u) atanh(sqrt(-u)) (u<0)."""
import math
import sys
from dataclasses import replace

import numpy as np
from scipy.special import digamma
from scipy.integrate import quad
from scipy.optimize import fsolve




def L_T0(z, D):
    return complex(np.log(-z) - np.log(-D - z))


def L_T(z, D, T):
    if T == 0.0:
        return L_T0(z, D)
    return complex(digamma(0.5 + 1j * z / (2 * math.pi * T))) - math.log(D / (2 * math.pi * T)) - 0.5j * math.pi


def L_quad(z, D, T, up=4.0):
    f = lambda w: 1.0 / (math.exp(w / T) + 1.0) if abs(w / T) < 700 else (0.0 if w > 0 else 1.0)
    pts = sorted(set([-D, z.real - 50 * abs(z.imag), z.real, z.real + 50 * abs(z.imag), -60 * T, 0.0, 60 * T, up]))
    pts = [p for p in pts if -D <= p <= up]
    re = im = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        re += quad(lambda w: f(w) * ((w - z.real) / ((w - z.real) ** 2 + z.imag ** 2)), a, b, limit=400, epsabs=1e-13)[0]
        im += quad(lambda w: f(w) * (z.imag / ((w - z.real) ** 2 + z.imag ** 2)), a, b, limit=400, epsabs=1e-13)[0]
    return complex(re, im)  # 1/(w - z) = (w - Re z + i Im z)/|w - z|^2 ... sign handled: Im[1/(w-z)] = Im z/|w-z|^2


def equations(r2, et, eps, G0, D0, Gp, D, T, Q=1.0):
    s0 = np.sqrt(complex(D0**2 - Gp**2))
    zp = et - 1j * r2 * G0 + r2 * s0
    zm = et - 1j * r2 * G0 - r2 * s0
    Lp, Lm = L_T(zp, D, T), L_T(zm, D, T)
    nf = -(Lp + Lm).imag / math.pi
    dc = -(G0 / math.pi) * (Lp + Lm).real + (s0 * (Lp - Lm)).imag / math.pi
    return nf, dc, zp, zm


def solve(eps, G0, D0, Gp, D, T, Q=1.0, guess=None):
    """unknowns: x = (ln r^2, et); residuals: constraint and delta_c = et - eps."""
    def res(x):
        r2, et = math.exp(x[0]), x[1]
        nf, dc, _, _ = equations(r2, et, eps, G0, D0, Gp, D, T, Q)
        return [r2 + nf - Q, dc - (et - eps)]
    if guess is None:
        guess = [math.log(closed_r2(eps, G0, D0, Gp, D)), 0.0]
    x, info, ier, message = fsolve(res, guess, xtol=1e-11, full_output=True)
    residual_max = float(np.max(np.abs(res(x))))
    if not np.isfinite(residual_max) or residual_max > 1e-10:
        raise RuntimeError(f"Unconverged Coleman saddle: {message}; residual={residual_max}")
    r2, et = math.exp(x[0]), x[1]
    nf, dc, zp, zm = equations(r2, et, eps, G0, D0, Gp, D, T, Q)
    return dict(r=math.sqrt(r2), r2=r2, et=et, dc=dc, zp=zp, zm=zm, res=res(x), residual_max=residual_max, solver_success=bool(ier == 1), message=message)


def Ffun(u):
    if u >= 0:
        x = math.sqrt(u)
        return x * math.atan(x)
    y = math.sqrt(-u)
    return -y * math.atanh(y)


def closed_r2(eps, G0, D0, Gp, D):
    u = (D0**2 - Gp**2) / G0**2
    TK0 = D * math.exp(-math.pi * abs(eps) / (2 * G0))
    return TK0 * math.exp(Ffun(u)) / (G0 * math.sqrt(1 + u))


if __name__ == "__main__":
    eps, G0, D = -1.0, 0.12, 4.0
    # 1. digamma closed form of L vs adaptive quadrature
    for z, T in ((1e-5 - 1.1e-5j, 1e-6), (-3e-6 - 2e-6j, 1e-6), (2e-5 - 5e-6j, 3e-6)):
        a, b = L_T(z, D, T), L_quad(z, D, T)
        b = complex(b.real, b.imag)
        print("1. L(z=%s, T=%g): digamma %s  quad %s" % (z, T, np.round(a, 8), np.round(b, 8)))
    TK0 = D * math.exp(-math.pi * abs(eps) / (2 * G0))
    print("2. T_K0 = D exp(-pi|eps|/(2 Gamma0)) = %.4e   (D=%g, eps=%g, Gamma0=%g)" % (TK0, D, eps, G0))
    archive = {0.35: 0.010468828715785698, 0.40: 0.01025735315097868, 0.45: 0.01003546481470414,
               0.50: 0.009807411733055882, 0.55: 0.009551970257228173}
    print("3. archive convention D0 = 0.075+0.05 b, Gamma_PT = 0.2 b")
    print("   beta0   u        closed-form r   exact T=0 r   T=1e-6 r     archive r    archive/T=1e-6")
    for b in (0.35, 0.40, 0.45, 0.50, 0.55):
        D0, Gp = 0.075 + 0.05 * b, 0.2 * b
        u = (D0**2 - Gp**2) / G0**2
        rc = math.sqrt(closed_r2(eps, G0, D0, Gp, D))
        s0 = solve(eps, G0, D0, Gp, D, 0.0)
        s1 = solve(eps, G0, D0, Gp, D, 1e-6, guess=[math.log(s0["r2"]), s0["et"]])
        print("   %.2f  %+.4f   %.6f      %.6f      %.6f     %.6f    %.3f" % (b, u, rc, s0["r"], s1["r"], archive[b], archive[b] / s1["r"]))
    print("4. text convention D0 = 0.10, Gamma_PT = 0.2 b: pole structure and SB scales (T=0)")
    print("   beta0   r          width r^2 G0    |z+|,|z-|                 Re z+-           ")
    for b in (0.30, 0.40, 0.45, 0.50, 0.55, 0.58):
        D0, Gp = 0.10, 0.2 * b
        s = solve(eps, G0, D0, Gp, D, 0.0)
        print("   %.2f  %.6f  %.4e     %.4e, %.4e   %+.3e %+.3e   Im: %.3e %.3e"
              % (b, s["r"], s["r2"] * G0, abs(s["zp"]), abs(s["zm"]), s["zp"].real, s["zm"].real, s["zp"].imag, s["zm"].imag))
