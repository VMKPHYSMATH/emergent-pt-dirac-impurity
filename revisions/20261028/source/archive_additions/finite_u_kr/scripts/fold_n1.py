"""Test whether the mean-filling-one KR EP branch ends in a fold (EP-pair annihilation).
U becomes an unknown and the branch is parametrized by the other control:
  mode 'VO': W fixed at 0.10, step V_O (unknowns: saddle x, z, eps_d, U, mu)
  mode 'W' : V_O fixed, step W        (unknowns: saddle x, z, eps_d, U, mu)
Equations: saddle stationarity (7), F = 0 (2), dF/dz = 0 (2), Tr n = 1 (1).
usage: python3 -I fold_n1.py {VO|W} out.json"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import root
from scipy.optimize._numdiff import approx_derivative
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kr_package" / "src"))
import matrix_saddle as m

mode, out = sys.argv[1], Path(sys.argv[2])
log = json.loads((Path(__file__).resolve().parents[1] / ("results/ep_n1_extended_%s.json" % ("VO" if mode == "VO" else "W"))).read_text())
good = [r for r in log if r.get("success", True) and not r.get("failed_attempt")]
r0 = good[-1]
s = m.Saddle(r0["mu"], 32, p=m.replace(m.P, impurity_hybridization=r0["impurity_hybridization"]))


def unpack(v):
    x = v[:7]; z = complex(v[7], v[8]) * .01; eps = v[9] * .01; U = v[10] * .001; mu = v[11] * .01
    return x, z, eps, U, mu


def residual(v, par):
    x, z, eps, U, mu = unpack(v)
    s.set_mu(mu)
    if mode == "VO":
        vo = par; s.set_W(0.10)
    else:
        vo = m.VO0; s.set_W(par)
    F = m.determinant(z, x, vo, s.p); Fz = m.ddet(z, x, vo, s.p)
    return np.r_[s.residual(x, U, eps, vo), F.real / 1e-5, F.imag / 1e-5, Fz.real / .01, Fz.imag / .01,
                 np.trace(m.unpack(x)[2]) - 1.0]


def solve(v0, par):
    f = lambda v: residual(v, par)
    sol = root(f, v0, jac=lambda v: approx_derivative(f, v, method='3-point', abs_step=1e-6), tol=1e-12)
    res = float(np.max(np.abs(f(sol.x))))
    return sol.x, res


v = np.r_[np.array(r0["x"]), r0["z"][0] / .01, r0["z"][1] / .01, r0["epsilon"] / .01, r0["U"] / .001, r0["mu"] / .01]
par = r0["auxiliary_vertex"] if mode == "VO" else r0["impurity_hybridization"]
dpar = float(sys.argv[3]) if len(sys.argv) > 3 else (-0.002 if mode == "VO" else +0.002)
rows = []; prev = None
for it in range(120):
    v_try = v if prev is None else v + (v - prev)      # secant predictor in the parameter
    vn, res = solve(v_try, par)
    if res > 1e-9:
        vn, res = solve(v, par)
    x, z, eps, U, mu = unpack(vn)
    s.set_mu(mu)
    vo = par if mode == "VO" else m.VO0
    d = s.describe(x, U, eps, vo)
    dn = d["density_eigenvalues"]
    row = dict(par=par, U=U, z=[z.real, z.imag], epsilon=eps, mu=mu, n_d=d["n_d"], D=d["double_occupancy"], det_n=dn[0] * dn[1],
               Z=d["Z_eigenvalues"], residual=res, ok=bool(res < 1e-9 and z.imag < 0),
               channels=len(m.bath.open_channels(z.real, vo, s.p)))
    rows.append(row)
    print(json.dumps({k: (round(v_, 8) if isinstance(v_, float) else v_) for k, v_ in row.items() if k != "Z"}), flush=True)
    if not row["ok"]:
        break
    prev, v = v, vn
    par = par + dpar
    if U < 0.0005 or U > 0.05:
        break
    out.write_text(json.dumps(rows, indent=1))
out.write_text(json.dumps(rows, indent=1))
