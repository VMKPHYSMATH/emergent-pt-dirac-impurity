import os
"""Finite-U KR/Gutzwiller saddle on the SAME z-shifted 12-site matrix Wilson chains used by the paper's
block-Wilson finite-U calculation, combined with the continuum kernel by averaging inverse-kernel corrections (different from the ED response-ratio average):
    G_d^{-1}(zeta) = avg_z[ R_z^{-1} (zeta - Lambda_z) R_z^{-1} ] - Sigma_bath^II(zeta; W, V_O)
EP: F = dF/dzeta = 0, solved for (zeta, eps_d, W) at fixed V_O, with outer self-consistency of the saddles.
Inputs (read-only): archived chains in the repo; package's KR code (uploaded, run from its src)."""
import csv, json, sys, math
from pathlib import Path
import numpy as np
from scipy.optimize import root

PK = str(Path(__file__).resolve().parents[1] / "kr_package" / "src")
sys.path.insert(0, PK)
import matrix_saddle as m
bath = m.bath
CH = (Path(os.environ["EPT_REPO"]) / "reproducibility/full_k_finite_u/finiteU_fullk_reproducibility/preflight_lowbands_z4" if "EPT_REPO" in os.environ else Path(__file__).resolve().parents[3] / "reproducibility/filling/reference/submitted_chains")
I2 = np.eye(2)
SITES = 12


def read_chain(csv_path, summary_path):
    onsite, hopping = [], []
    with open(csv_path) as h:
        for row in csv.DictReader(h):
            mat = lambda pre: np.array([complex(float(row[f"{pre}{a}{b}_re"]), float(row[f"{pre}{a}{b}_im"]))
                                        for a, b in ((1, 1), (1, 2), (2, 1), (2, 2))]).reshape(2, 2)
            onsite.append(mat("a")); hopping.append(mat("b"))
    hopping = hopping[:-1]
    enc = json.loads(Path(summary_path).read_text())["block_chain"]["impurity_to_chain_matrix"]
    return onsite, hopping, np.array([[complex(v[0], v[1]) for v in r] for r in enc])


CHAINS = []
for c in json.loads((CH / "z_chain_manifest.json").read_text())["chains"]:
    on, hop, imp = read_chain(CH / c["chain"], CH / c["summary"])
    on = [0.5 * (a + a.conj().T) for a in on[:SITES]]
    CHAINS.append((on, hop[:SITES - 1], imp))


def chain_delta(zeta, chain, W):
    """impurity hybridization of one chain, C [zeta - H_chain]^{-1}_00 C^dag, by block continued fraction."""
    on, hop, imp = chain
    C = imp * (W / 0.10)
    zeta = np.atleast_1d(zeta)
    g = np.zeros((len(zeta), 2, 2), complex)
    for n in range(SITES - 1, -1, -1):
        A = zeta[:, None, None] * I2 - on[n]
        if n < SITES - 1:
            A = A - hop[n] @ g @ hop[n].conj().T
        g = np.linalg.inv(A)
    return C @ g @ C.conj().T, C @ C.conj().T


class ChainSaddle(m.Saddle):
    def __init__(self, chain, W, mu=0.0, order=32):
        super().__init__(mu, order)
        self._dch, self.M = chain_delta(self.omega, chain, W)

    def delta(self, vo):
        return self._dch

    def qp(self, R, lam, vo):
        d = self._dch
        G = m.inv2(self.omega[:, None, None] * I2 - lam - R @ d @ R)
        nq = I2 / 2 + np.einsum('w,wij->ij', self.w, G.real) - (lam - self.mu * I2) / (np.pi * self.ymax)
        B = np.einsum('w,wij->ij', self.w, (d @ R @ G + G @ R @ d).real) - (self.M.real @ R + R @ self.M.real) / (np.pi * self.ymax)
        return nq, B


def saddles(U, eps, W, x0s=None):
    out = []
    for k, ch in enumerate(CHAINS):
        s = ChainSaddle(ch, W)
        x0 = None if x0s is None else x0s[k]
        r = s.solve(U, eps, m.VO0, x0)
        if not r["success"]:
            r = s.solve(U, eps, m.VO0, None)
        out.append(r)
    return out


def ep_with_frozen_fields(A, phi, guess):
    """solve det[A(zeta-eps) - phi - Sigma^II(zeta;W)] = 0 = d/dzeta for (Re z, Im z, eps, W) at V_O fixed."""
    def F(zeta, eps, W):
        p = m.replace(m.P, impurity_hybridization=W)
        K = A * (zeta - eps) - phi - bath.resonance_self_energy(zeta, m.VO0, p)
        return complex(np.linalg.det(K))
    def res(v):
        zeta, eps, W = complex(v[0], v[1]), v[2], v[3]
        f = F(zeta, eps, W)
        fz = bath.derivative(lambda a: F(a, eps, W), zeta, 2e-6)
        return [f.real / 1e-5, f.imag / 1e-5, fz.real / 1e-2, fz.imag / 1e-2]
    sol = root(res, guess, tol=1e-13)
    return sol.x, max(abs(np.array(res(sol.x))))


def kr_chain_ep(U, guess):
    v = np.array(guess, float); x0s = None
    for it in range(12):
        eps, W = v[2], v[3]
        sd = saddles(U, eps, W, x0s); x0s = [np.array(r["x"]) for r in sd]
        Rs = [np.array(r["R"]) for r in sd]; Ls = [np.array(r["lambda_matrix"]) for r in sd]
        Rinv = [np.linalg.inv(R) for R in Rs]
        A = np.mean([Ri @ Ri for Ri in Rinv], axis=0)
        Bm = np.mean([Ri @ L @ Ri for Ri, L in zip(Rinv, Ls)], axis=0)
        phi = Bm - eps * A
        vnew, resid = ep_with_frozen_fields(A, phi, v)
        dv = np.max(np.abs(vnew - v)); v = vnew
        if dv < 1e-12:
            break
    return dict(U=U, z=complex(v[0], v[1]), eps=v[2], W=v[3], iterations=it + 1, ep_residual=resid,
                n_d=[r["n_d"] for r in sd], D=[r["double_occupancy"] for r in sd], Z=[r["Z_eigenvalues"] for r in sd],
                saddle_ok=all(r["success"] for r in sd), phi=phi, A=A)


if __name__ == "__main__":
    bw = {0.001: (0.014839462534, 0.095773861407), 0.0025: (0.014403967449, 0.090335818268)}
    bw_z = {0.0025: 0.01177670 - 0.00225815j}
    kr_cont = {0.001: (0.01463397, 0.09268284, 0.01187554 - 0.00238195j), 0.0025: (0.01390201, 0.08286488, 0.01168355 - 0.00187448j)}
    guess = [m.ZEP0.real, m.ZEP0.imag, m.EPS0, 0.10]
    for U in (0.0, 0.001, 0.0025):
        r = kr_chain_ep(U, guess)
        guess = [r["z"].real, r["z"].imag, r["eps"], r["W"]]
        print("U=%.4f  KR on chains: eps_d=%.6f W=%.6f z=%.6f%+.6fi  (iters %d, EP res %.1e, saddles ok %s)  n_d(z-avg)=%.4f D=%.2e Z=%.5f"
              % (U, r["eps"], r["W"], r["z"].real, r["z"].imag, r["iterations"], r["ep_residual"], r["saddle_ok"],
                 np.mean(r["n_d"]), np.mean(r["D"]), np.mean([min(z) for z in r["Z"]])))
        if U in bw:
            print("          block-Wilson ED: eps_d=%.6f W=%.6f %s | KR on continuum: eps_d=%.6f W=%.6f z=%s"
                  % (bw[U][0], bw[U][1], ("z=%.6f%+.6fi" % (bw_z[U].real, bw_z[U].imag)) if U in bw_z else "", kr_cont[U][0], kr_cont[U][1], np.round(kr_cont[U][2], 6)))
