"""Exact interacting PT Anderson control: no fitting and no state truncation.

This is a distinct non-Hermitian model with a spin-independent finite bath,
not the passive full-k driven-bath model. The response is the biorthogonal
analytic continuation of its nondegenerate Hermitian singlet ground state.
All frequencies are finite-system addition/removal energies, not certified
outgoing continuum resonances. Run with BLAS threads=1 for reproducibility.
"""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from scipy import linalg as la

SX = np.array([[0, 1], [1, 0]], complex)
SZ = np.diag([1, -1]).astype(complex)

def basis(modes, charge):
    return np.array([sum(1 << i for i in c)
                     for c in itertools.combinations(range(modes), charge)], dtype=int)

def act(state, i, create=False):
    occupied = bool(state & (1 << i))
    if occupied == create:
        return None
    sign = (-1) ** ((state & ((1 << i) - 1)).bit_count())
    return state ^ (1 << i), sign

def operator(source, target, i, create=False):
    target_index = {int(s): j for j, s in enumerate(target)}
    out = np.zeros((len(target), len(source)), complex)
    for j, s in enumerate(source):
        a = act(int(s), i, create)
        if a is not None:
            out[target_index[a[0]], j] = a[1]
    return out

def onebody(sites, U, hop=.5, field=None, bath_field=0.):
    h = np.zeros((2 * sites, 2 * sites), complex)
    h[:2, :2] = -U / 2 * np.eye(2)
    if field is not None:
        h[:2, :2] += field
    if bath_field:
        h[2:4, 2:4] += bath_field * SX
    for j in range(sites - 1):
        for spin in range(2):
            a, b = 2 * j + spin, 2 * (j + 1) + spin
            h[a, b] = h[b, a] = -hop
    return h

def hamiltonian(h, U, states):
    index = {int(s): j for j, s in enumerate(states)}
    H = np.zeros((len(states), len(states)), complex)
    terms = [(i, j, h[i, j]) for i, j in zip(*np.nonzero(h))]
    for col, encoded in enumerate(states):
        s = int(encoded)
        H[col, col] += U * bool(s & 1) * bool(s & 2)
        for i, j, value in terms:
            a = act(s, int(j))
            if a is None:
                continue
            b = act(a[0], int(i), True)
            if b is not None:
                H[index[b[0]], col] += value * a[1] * b[1]
    return H  # deliberately no Hermitian symmetrization

def spin_flip(states, sites):
    index = {int(s): j for j, s in enumerate(states)}
    P = np.zeros((len(states), len(states)))
    for j, s in enumerate(states):
        occupied = [i for i in range(2 * sites) if int(s) & (1 << i)]
        target = [i ^ 1 for i in occupied]
        inversions = sum(target[a] > target[b]
                         for a in range(len(target)) for b in range(a + 1, len(target)))
        P[index[sum(1 << i for i in target)], j] = (-1) ** inversions
    return P

def encode(a):
    a = np.asarray(a)
    return np.stack([a.real, a.imag], axis=-1).tolist()

class Response:
    def __init__(self, sites, U, delta=.1, hop=.5, form='raising', gamma=None,
                 bath_field=0., reference_energy=None):
        self.sites, self.U, self.delta = sites, U, delta
        N = sites
        self.bases = {q: basis(2 * sites, q) for q in (N - 1, N, N + 1)}
        hz = onebody(sites, U, hop, bath_field=bath_field)
        self.H0 = {q: hamiltonian(hz, U, b) for q, b in self.bases.items()}
        en, vn = la.eigh(self.H0[N])
        self.E0 = float(en[0]) if reference_energy is None else reference_energy
        self.gs = vn[:, 0]
        self.ground_gap = float(en[1] - en[0])
        if self.ground_gap < 1e-10:
            raise ValueError('Ground singlet is not isolated')
        self.cp = [operator(self.bases[N], self.bases[N + 1], s, True) for s in range(2)]
        self.cm = [operator(self.bases[N], self.bases[N - 1], s) for s in range(2)]
        if form == 'raising':
            field = np.array([[0, 2 * delta], [0, 0]], complex)
        elif form == 'pt':
            field = delta * SX + 1j * (delta if gamma is None else gamma) * SZ
        elif form == 'z':
            field = delta * SZ
        else:
            raise ValueError(form)
        self.field = field
        self.h = onebody(sites, U, hop, field, bath_field)
        self.H = {q: hamiltonian(self.h, U, b) for q, b in self.bases.items()}
        K = self.H[N] - self.E0 * np.eye(len(en))
        _, sr, vr = la.svd(K)
        _, sl, vl = la.svd(K.T)
        self.right = vr[-1].conj()
        self.left = vl[-1].conj()
        self.left /= self.left @ self.right
        self.ground_residual = max(la.norm(K @ self.right), la.norm(self.left @ K))
        self.ground_null_gap = float(sr[-2])
        self.leftplus = np.array([self.left @ c.conj().T for c in self.cp])
        self.rightplus = np.column_stack([c @ self.right for c in self.cp])
        self.leftminus = np.array([self.left @ c.conj().T for c in self.cm])
        self.rightminus = np.column_stack([c @ self.right for c in self.cm])
        self.Tp, Zp = la.schur(self.H[N + 1] - self.E0 * np.eye(len(self.bases[N + 1])), output='complex')
        self.Tm, Zm = la.schur(self.E0 * np.eye(len(self.bases[N - 1])) - self.H[N - 1], output='complex')
        self.lp, self.rp = self.leftplus @ Zp, Zp.conj().T @ self.rightplus
        self.lm, self.rm = self.leftminus @ Zm, Zm.conj().T @ self.rightminus
        self.sumrule = self.leftplus @ self.rightplus + (self.leftminus @ self.rightminus).T

    def green(self, z):
        p = la.solve_triangular(z * np.eye(len(self.Tp)) - self.Tp, self.rp)
        m = la.solve_triangular(z * np.eye(len(self.Tm)) - self.Tm, self.rm)
        return self.lp @ p + (self.lm @ m).T

    def direct_green(self, z):
        N = self.sites
        p = la.solve((z + self.E0) * np.eye(len(self.Tp)) - self.H[N + 1], self.rightplus)
        m = la.solve((z - self.E0) * np.eye(len(self.Tm)) + self.H[N - 1], self.rightminus)
        return self.leftplus @ p + (self.leftminus @ m).T

    def observable(self, function, biorth=False):
        values = np.array([function(int(s)) for s in self.bases[self.sites]])
        if biorth:
            return self.left @ (values * self.right)
        return np.vdot(self.gs, values * self.gs)

    def lowest_addition(self):
        N = self.sites
        b = self.bases[N + 1]
        # Work in total Sz=+1/2, fixing the member of the spin doublet.
        sz2 = np.array([sum((1 if i % 2 == 0 else -1) * bool(int(s) & (1 << i))
                          for i in range(2 * self.sites)) for s in b])
        keep = np.flatnonzero(sz2 == 1)
        e, v = la.eigh(self.H0[N + 1][np.ix_(keep, keep)])
        up = np.zeros(len(b), complex)
        up[keep] = v[:, 0]
        mlocal = np.array([bool(int(s) & 1) - bool(int(s) & 2) for s in b], dtype=float)
        slope = float(np.vdot(up, mlocal * up).real)
        Z = float(abs(np.vdot(up, self.cp[0] @ self.gs)) ** 2)
        energy = float(e[0])
        omega = energy - self.E0
        allp = la.eigvalsh(self.H0[N + 1]) - self.E0
        allm = self.E0 - la.eigvalsh(self.H0[N - 1])
        distances = abs(np.r_[allp, allm] - omega)
        gap = float(distances[distances > 1e-8].min())
        return dict(omega=omega, energy=energy, weight=Z, slope=slope,
                    predicted_B_coefficient=Z * slope, isolation_gap=gap,
                    algebraic_multiplicity=int(np.sum(abs(allp - omega) < 1e-8)))

    def contour(self, pole, radius, points):
        phase = np.exp(2j * np.pi * (np.arange(points) + .371) / points)
        G = np.array([self.green(pole + radius * p) for p in phase])
        return [np.mean(G * ((radius * phase) ** k)[:, None, None], axis=0)
                for k in (1, 2, 3)]

def check(sites, U, delta=.1):
    R = Response(sites, U, delta)
    a = R.lowest_addition()
    pole = a['omega']
    contours = []
    for fraction, points in ((.18, 64), (.09, 128)):
        radius = fraction * a['isolation_gap']
        A, B, C = R.contour(pole, radius, points)
        coefficient = np.vdot(R.field, B) / np.vdot(R.field, R.field)
        contours.append(dict(radius=radius, points=points, A=encode(A), B=encode(B),
                             coefficient=encode(coefficient), scalar_residue=float(np.trace(A).real / 2),
                             coefficient_error=float(abs(coefficient - a['predicted_B_coefficient'])),
                             direction_error=float(la.norm(B - coefficient * R.field)),
                             third_pole_coefficient_norm=float(la.norm(C))))
    K = R.H[sites + 1] - a['energy'] * np.eye(len(R.Tp))
    s1, s2 = la.svdvals(K), la.svdvals(K @ K)
    null1, null2 = int(np.sum(s1 < 1e-9)), int(np.sum(s2 < 1e-9))
    obs = lambda s: int(bool(s & 1)) + int(bool(s & 2))
    doub = lambda s: int(bool(s & 1)) * int(bool(s & 2))
    nd, D = float(R.observable(obs).real), float(R.observable(doub).real)
    scalar_error = max(abs(R.observable(obs, True) - nd), abs(R.observable(doub, True) - D))
    z = .21 + .17j
    g = R.green(z)
    covariance = max(abs(g[0, 0] - g[1, 1]), abs(g[1, 0]))
    control = onebody(sites, U, field=delta * SX + 1j * delta * SZ)
    hp = hamiltonian(control, U, R.bases[sites])
    P = spin_flip(R.bases[sites], sites)
    pt_error = la.norm(P @ hp.conj() @ P - hp)
    result = dict(sites=sites, bath_sites=sites-1, U=U, hopping=.5, delta=delta,
                  eps_d=-U/2, state='biorthogonal continuation of isolated Hermitian singlet',
                  ground_energy=R.E0, ground_gap=R.ground_gap, ground_residual=R.ground_residual,
                  nd=nd, double_occupancy=D, P1=nd-2*D, spin_squared=.75*(nd-2*D),
                  invariant_observable_error=float(scalar_error), addition=a,
                  jordan_nullities=[null1, null2], singular_values_near_zero=s1[-3:].tolist(),
                  contours=contours, sumrule_error=float(la.norm(R.sumrule-np.eye(2))),
                  covariance_error=float(covariance), pt_symmetry_error=float(pt_error),
                  direct_schur_error=float(la.norm(g-R.direct_green(z))))
    if U == 0:
        result['U0_onebody_green_error'] = float(la.norm(g - la.inv(z*np.eye(2*sites)-R.h)[:2,:2]))
    assert null1 == 1 and null2 == 2
    assert a['weight'] > 0 and abs(a['slope']) > 1e-8
    assert result['sumrule_error'] < 1e-9
    assert result['direct_schur_error'] < 1e-9
    assert scalar_error < 1e-8 and covariance < 1e-8
    assert max(c['coefficient_error'] for c in contours) < 1e-7
    assert max(c['third_pole_coefficient_norm'] for c in contours) < 1e-7
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sites', type=int, nargs='+', default=[2,4])
    ap.add_argument('--U', type=float, nargs='+', default=[0,.5,2,4,8,16])
    ap.add_argument('--delta', type=float, default=.1)
    ap.add_argument('--output', type=Path, default=Path('results/checks.json'))
    args = ap.parse_args()
    results = []
    for sites in args.sites:
        for U in args.U:
            row = check(sites, U, args.delta)
            results.append(row)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(results, indent=2))
            print(f"sites={sites} U={U:g}: nd={row['nd']:.9f}, D={row['double_occupancy']:.9f}, "
                  f"omega={row['addition']['omega']:.9f}, Zm={row['addition']['predicted_B_coefficient']:.9g}, "
                  f"nullities={row['jordan_nullities']}", flush=True)

if __name__ == '__main__':
    main()
