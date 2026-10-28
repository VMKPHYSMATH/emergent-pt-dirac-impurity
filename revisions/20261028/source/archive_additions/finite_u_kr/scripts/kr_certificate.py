"""Topological certificate (local-discriminant winding + pole exchange) for continuum KR EPs,
with the finite-U saddle re-solved at every control point of the loop.

Uses the validated package code (matrix_saddle.py) read-only; outputs go to ./results.
Loop:  x1 = x1_c + h1 cos(phi),  x2 = x2_c + h2 sin(phi)  with (x1, x2) = (eps_d, V_O) or (eps_d, W).
"""
import json, math, sys, time
from pathlib import Path
import numpy as np
from scipy.optimize import root, linear_sum_assignment

PK = str(Path(__file__).resolve().parents[1] / "kr_package" / "src")
sys.path.insert(0, PK)
import matrix_saddle as m          # noqa: E402
import validate as v               # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)


def fz(z, x, vo, p):
    return complex(m.ddet(z, x, vo, p))


def fzz(z, x, vo, p):
    return complex(m.bath.derivative(lambda a: m.ddet(a, x, vo, p), z, 2e-6))


def critical(z0, x, vo, p):
    sol = root(lambda c: [fz(complex(*c), x, vo, p).real, fz(complex(*c), x, vo, p).imag], [z0.real, z0.imag], tol=1e-12)
    return bool(sol.success), complex(*sol.x)


def pole(z0, x, vo, p):
    def f(c):
        val = m.determinant(complex(*c), x, vo, p)
        return [val.real, val.imag]
    sol = root(f, [z0.real, z0.imag], tol=1e-12)
    zz = complex(*sol.x)
    return bool(sol.success), zz, abs(m.determinant(zz, x, vo, p))


def certificate(row, control, h1, h2, samples=128, order=32):
    U = row["U"]; mu = row["mu"]
    W0 = row["impurity_hybridization"]; vo0 = row["auxiliary_vertex"]; eps0 = row["epsilon"]
    s = m.Saddle(mu, order, p=m.replace(m.P, impurity_hybridization=W0))
    x = np.array(row["x"]); zc = complex(*row["z"])
    recs = []; tracked = []; amax = 0.0; fails = 0; flag_fail = 0
    for phi in np.linspace(0.0, 2 * math.pi, samples + 1):
        eps = eps0 + h1 * math.cos(phi)
        if control == "V_O":
            vo = vo0 + h2 * math.sin(phi); W = W0
        else:
            vo = vo0; W = W0 + h2 * math.sin(phi)
        s.set_W(W)
        sr = s.solve(U, eps, vo, x)
        if not sr["success"]:
            fails += 1
        x = np.array(sr["x"]); p = s.p
        ok, zc = critical(zc, x, vo, p)
        fzabs = abs(fz(zc, x, vo, p))
        flag_fail += (not ok)
        fails += (fzabs > 1e-12)
        D = m.determinant(zc, x, vo, p); c2 = fzz(zc, x, vo, p)
        split = np.sqrt(-2 * D / c2)
        sols = [pole(zc + sgn * split, x, vo, p) for sgn in (1, -1)]
        flag_fail += sum(not a for a, _, _ in sols)
        fails += sum(e > 1e-16 for _, _, e in sols)
        new = np.array([b for _, b, _ in sols])
        if tracked:
            cost = abs(tracked[-1][:, None] - new[None, :])
            r_, c_ = linear_sum_assignment(cost)
            ordered = np.empty(2, complex); ordered[r_] = new[c_]
            half = 0.5 * min(abs(tracked[-1][1] - tracked[-1][0]), abs(ordered[1] - ordered[0]))
            amax = max(amax, float(np.max(cost[r_, c_])) / half)
            new = ordered
        tracked.append(new)
        nch = [len(m.bath.open_channels(zz.real, vo, p)) for zz in new]
        recs.append(dict(phi=float(phi), epsilon=eps, V_O=vo, W=W, n_d=sr["n_d"], saddle_residual=sr["residual_max"], F_z_abs=fzabs,
                         z_c=[zc.real, zc.imag], D=[D.real, D.imag], F_zz=[c2.real, c2.imag],
                         poles=[[zz.real, zz.imag] for zz in new], pole_residual=max(e for _, _, e in sols), open_channels=nch))
    Dl = np.array([complex(*r["D"]) for r in recs])
    inc = np.angle(Dl[1:] / Dl[:-1])
    arr = np.array(tracked)
    r_, c_ = linear_sum_assignment(abs(arr[-1][:, None] - arr[0][None, :]))
    perm = np.empty(2, int); perm[r_] = c_
    sep = arr[:, 1] - arr[:, 0]
    summary = dict(U=U, mu=mu, control=control, h1=h1, h2=h2, samples=samples, order=order,
                   winding=float(inc.sum() / (2 * math.pi)), minimum_boundary_modulus=float(abs(Dl).min()),
                   maximum_phase_increment=float(abs(inc).max()),
                   minimum_F_zz=float(min(abs(complex(*r["F_zz"])) for r in recs)),
                   endpoint_permutation=perm.tolist(),
                   half_winding=float(np.sum(np.angle(sep[1:] / sep[:-1])) / (2 * math.pi)),
                   exchange_error=float(np.max(abs(arr[-1] - arr[0, ::-1]))),
                   minimum_pole_separation=float(abs(sep).min()),
                   maximum_pole_residual=float(max(r["pole_residual"] for r in recs)),
                   maximum_assignment_ratio=float(amax),
                   maximum_saddle_residual=float(max(r["saddle_residual"] for r in recs)),
                   all_two_cut=all(n == 2 for r in recs for n in r["open_channels"]),
                   failures=int(fails), solver_flag_failures=int(flag_fail),
                   maximum_F_z_at_critical=float(max(r["F_z_abs"] for r in recs)))
    summary["certified"] = bool(abs(abs(summary["winding"]) - 1) < 1e-6 and summary["maximum_phase_increment"] < math.pi / 2
                                and summary["endpoint_permutation"] == [1, 0] and abs(abs(summary["half_winding"]) - 0.5) < 1e-6
                                and summary["maximum_assignment_ratio"] < 0.5 and summary["failures"] == 0 and summary["all_two_cut"])
    return summary, recs


def refine(row, control, order=40):
    s = m.Saddle(row["mu"], order, p=m.replace(m.P, impurity_hybridization=row["impurity_hybridization"]))
    if control == "W":
        s.set_W(row["impurity_hybridization"])
    return m.joint_ep(s, row["U"], np.array(row["y"]), None, control)


def extras(row):
    p = m.replace(m.P, impurity_hybridization=row["impurity_hybridization"])
    vo = row["auxiliary_vertex"]; z = complex(*row["z"])
    th = m.bath.upper_threshold(vo, p)
    # physical-sheet half-width matrix of the coherent kernel at omega = Re z: R(-Im Sigma^I)R
    R = np.array(row["R"]); S = m.bath.analytic_physical_self_energy(z.real + 1e-10j, vo, p)
    hw = R @ (-(S - S.conj().T) / 2j) @ R
    ev = np.linalg.eigvalsh((hw + hw.conj().T) / 2)
    lam = np.array(row["lambda_matrix"]); comp = m.components(lam)
    return dict(threshold=th, gap_in_widths=(z.real - th) / abs(z.imag), half_width_eigs=ev.tolist(),
                passivity_margin=float(ev.min() / ev.mean()), VO_offshell=vo / (m.P.auxiliary_energy - m.P.drive_frequency),
                lambda_traceless=float(np.hypot(comp[1], comp[2])), lambda_x=float(comp[1]), lambda_z=float(comp[2]),
                U_over_Gamma=row["U"] / (row["impurity_hybridization"] ** 2 / 4))


if __name__ == "__main__":
    t0 = time.time()
    R0 = Path(PK).parent / "results"
    mu0 = json.loads((R0 / "ep_mu000.json").read_text())
    report = {}
    for Ut in (0.02, 0.2):
        row = min(mu0, key=lambda r: abs(r["U"] - Ut))
        ref = refine(row, "V_O", 40)
        assert ref["success"], ref
        st = v.stability(ref, 48)
        ex = extras(ref)
        summ, recs = certificate(ref, "V_O", 1e-4, 1e-2)
        report["mu0_VO_U%g" % Ut] = dict(ep=dict(z=ref["z"], eps=ref["epsilon"], V_O=ref["auxiliary_vertex"], n_d=ref["n_d"], D=ref["double_occupancy"],
                                                 Z=ref["Z_eigenvalues"], dens=ref["density_eigenvalues"], sv=ref["singular_values"], F=ref["F"], F_z=ref["F_z"],
                                                 F_zz=ref["F_zz"], joint=ref["joint_residual_max"]),
                                         stability=st, extras=ex, certificate=summ)
        (OUT / ("cert_mu0_VO_U%g.json" % Ut)).write_text(json.dumps(dict(ep=ref, stability=st, extras=ex, summary=summ, records=recs), indent=1))
        print(json.dumps(report["mu0_VO_U%g" % Ut], indent=1), flush=True)
        print("elapsed %.0f s" % (time.time() - t0), flush=True)
    (OUT / "cert_summary.json").write_text(json.dumps(report, indent=1))
