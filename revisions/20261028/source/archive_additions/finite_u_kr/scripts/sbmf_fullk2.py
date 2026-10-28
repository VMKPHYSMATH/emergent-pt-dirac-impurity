from pathlib import Path
"""SBMF (U=inf, Q=1, spin-scalar saddle) vs spin-dependent (Fock) renormalization of the
stationary full-k nonlinear pole EP.  Uses sbmf_fullk.py helpers."""
import math
import sys
from dataclasses import replace

import numpy as np
from scipy.optimize import root

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fullk_cert as C  # noqa: E402
from sbmf_fullk import EP0, I2, P0, G_f, _iy_integral, occupations_T, occupations_T0, pW, solve_ep  # noqa: E402

SX, SZ = C.SX, C.SZ
VO0, EPS0, W0 = EP0["VO"], EP0["eps"], 0.10


def density_matrix_T0(eps_t, VO, p, extra=None):
    """physical T=0 impurity density matrix rho = I/2 + (1/pi) int_0^inf Re G(iy) dy (real-symmetric problem)."""
    def Gm(y):
        S = C.analytic_physical_self_energy(1j * y, VO, p)
        if extra is not None:
            S = S + extra
        return np.linalg.inv((1j * y - eps_t) * I2 - S)
    out = np.zeros((2, 2))
    for a in range(2):
        for b in range(2):
            out[a, b] = _iy_integral(lambda y: Gm(y)[a, b].real) / math.pi
    return 0.5 * np.eye(2) + out


def inverse_map(eps_t, Wp, VO, T=0.0):
    """renormalized (eps_t, W') -> bare (eps_d, W) for the U=inf Coleman saddle at fixed V_O."""
    p = pW(Wp)
    if T == 0.0:
        nf, Hh = occupations_T0(eps_t, VO, p)
    else:
        nf, Hh = occupations_T(eps_t, VO, p, T, nmax=int(max(20000, 40.0 / T)))
    r2 = 1.0 - nf
    r = math.sqrt(r2)
    dc = -Hh / (2.0 * r2)
    return dict(nf=nf, Hh=Hh, r=r, dc=dc, eps_d=eps_t - dc, W=Wp / r)


# ---------------------------------------------------------------- reduced EP equations with a constant extra self-energy
def reduced(x, VO, extra, h=2e-6, unknown="W", fixed=None):
    """unknowns (Re z, Im z, W) at fixed V_O  [unknown='W'] or (Re z, Im z, V_O) at fixed W [unknown='VO']."""
    z = complex(x[0], x[1])
    if unknown == "W":
        p, vo = pW(x[2]), VO
    else:
        p, vo = pW(fixed), x[2]

    def comps(arg):
        S = C.resonance_self_energy(arg, vo, p) + extra
        return C.pauli_components(S)
    s0, sx, sz = comps(z)
    ds0, dsx, dsz = C.derivative(comps, z, h)
    a = (sx * dsx + sz * dsz) / (1.0 - ds0)
    disc = a**2 - sx**2 - sz**2
    eps = z - s0 - a
    return np.array([disc.real, disc.imag, eps.imag]), eps


def solve_ep_extra(VO, extra, guess, unknown="W", fixed=None):
    res = root(lambda x: reduced(x, VO, extra, unknown=unknown, fixed=fixed)[0], guess, tol=1e-13)
    _, eps = reduced(res.x, VO, extra, unknown=unknown, fixed=fixed)
    return res.success, res.x, eps.real


if __name__ == "__main__":
    print("A. SBMF at the U=0 EP point, controls (eps_d, W) at fixed V_O=%.6f: z_EP is unchanged" % VO0)
    for T in (0.0, 1e-3, 1e-2, 1e-1):
        m = inverse_map(EPS0, W0, VO0, T)
        print("   T_b=%-6g n_f=%.5f r^2=%.5f r=%.5f delta_c=%.4e  -> bare eps_d=%.6f  W=%.6f"
              % (T, m["nf"], m["r"] ** 2, m["r"], m["dc"], m["eps_d"], m["W"]))

    # Jacobian of the inverse map (eps_t, W') -> (eps_d, W) at T=0
    h1, h2 = 2e-5, 2e-4
    def bare(et, wp):
        m = inverse_map(et, wp, VO0, 0.0)
        return np.array([m["eps_d"], m["W"]])
    Jm = np.column_stack([(bare(EPS0 + h1, W0) - bare(EPS0 - h1, W0)) / (2 * h1),
                          (bare(EPS0, W0 + h2) - bare(EPS0, W0 - h2)) / (2 * h2)])
    print("   Jacobian d(eps_d,W)/d(eps_t,W') =\n", np.round(Jm, 5), "\n   det = %.5f (nonzero: local diffeomorphism, |degree| = 1 carries over)" % np.linalg.det(Jm))

    print("B. controls (eps_d, V_O) at fixed bare W=0.10 (the U=0 witness convention): self-consistent re-solve")
    x = np.array([EP0["z"].real, EP0["z"].imag, VO0])
    r = 1.0
    for it in range(6):
        ok, x, eps_t = solve_ep_extra(None, np.zeros((2, 2)), x, unknown="VO", fixed=W0 * r)
        m = inverse_map(eps_t, W0 * r, x[2], 0.0)
        print("   it %d: W'=%.6f  z_EP=%.8f%+.8fi  eps_t=%.6f  V_O=%.6f  n_f=%.5f  r=%.6f  bare eps_d=%.6f"
              % (it, W0 * r, x[0], x[1], eps_t, x[2], m["nf"], m["r"], m["eps_d"]))
        if abs(m["r"] - r) < 1e-9:
            break
        r = m["r"]

    print("C. EP family in the effective hybridization W' (fixed V_O free, lambda=0.05): n_f stays small")
    x = np.array([EP0["z"].real, EP0["z"].imag, VO0])
    rows = []
    for Wp in [0.10, 0.12, 0.14, 0.17, 0.20, 0.24, 0.28]:
        ok, x, eps_t = solve_ep_extra(None, np.zeros((2, 2)), x, unknown="VO", fixed=Wp)
        if not ok:
            print("   W'=%.3f: continuation failed" % Wp); break
        nf, Hh = occupations_T0(eps_t, x[2], pW(Wp))
        thr = C.upper_threshold(x[2], pW(Wp))
        rows.append((Wp, x[0], x[1], eps_t, x[2], nf, thr))
    x = np.array([EP0["z"].real, EP0["z"].imag, VO0])
    for Wp in [0.08, 0.06, 0.04, 0.03, 0.02]:
        ok, x, eps_t = solve_ep_extra(None, np.zeros((2, 2)), x, unknown="VO", fixed=Wp)
        if not ok:
            print("   W'=%.3f: continuation failed" % Wp); break
        nf, Hh = occupations_T0(eps_t, x[2], pW(Wp))
        thr = C.upper_threshold(x[2], pW(Wp))
        rows.append((Wp, x[0], x[1], eps_t, x[2], nf, thr))
    for row in sorted(rows):
        print("   W'=%.3f  z_EP=%.6f%+.6fi  eps_t=%.6f  V_O=%.5f  n_f=%.4f  r^2=%.4f  (Re z - threshold)/|Im z|=%.3f"
              % (row[0], row[1], row[2], row[3], row[4], row[5], 1 - row[5], (row[1] - row[6]) / abs(row[2])))

    print("D. spin-dependent (Fock) self-energy U[(Tr rho) I - rho] at first order vs block-Wilson continuation")
    rho = density_matrix_T0(EPS0, VO0, P0)
    print("   T=0 density matrix at the U=0 EP:", np.round(rho, 7), " Tr=%.6f" % np.trace(rho))
    s_core = C.pauli_components(C.resonance_self_energy(EP0["z"], VO0, P0))
    print("   kernel components at z_EP (s0, sx, sz):", np.round(s_core, 7))
    bw = {0.001: (0.014839462534, 0.095773861407), 0.0015: (0.014686601715, 0.093853257351),
          0.002: (0.014541530440, 0.092043396625), 0.0025: (0.014403967449, 0.090335818268)}
    x = np.array([EP0["z"].real, EP0["z"].imag, W0])
    for U in (0.0005, 0.001, 0.0015, 0.002, 0.0025):
        rr = rho.copy()
        for it in range(4):  # self-consistent Hartree-Fock on the physical sheet
            ext = U * (np.trace(rr) * np.eye(2) - rr)
            ok, x, eps_d = solve_ep_extra(VO0, ext, x, unknown="W")
            rr_new = density_matrix_T0(eps_d, VO0, pW(x[2]), extra=ext)
            if np.max(np.abs(rr_new - rr)) < 1e-10:
                break
            rr = rr_new
        ref = bw.get(U)
        print("   U=%.4f  HF: z_EP=%.6f%+.6fi  eps_d=%.6f  W=%.6f" % (U, x[0], x[1], eps_d, x[2])
              + ("   | block-Wilson: eps_d=%.6f W=%.6f" % ref if ref else ""))
    # SBMF analogue of the same U: none -- the large-N saddle has no spin-dependent field
