from pathlib import Path
import json, sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kr_package" / "src"))
import matrix_saddle as m
from scipy.optimize._numdiff import approx_derivative
for U in ("0.02", "0.2"):
    row = json.load(open(Path(__file__).resolve().parents[1] / ("results/cert_mu0_VO_U%s.json" % U)))["ep"]
    s = m.Saddle(row["mu"], 48, p=m.replace(m.P, impurity_hybridization=row["impurity_hybridization"]))
    x = np.array(row["x"]); Uv = row["U"]; eps = row["epsilon"]; vo = row["auxiliary_vertex"]
    def gradient(x):
        f = s.residual(x, Uv, eps, vo); prob = m.unpack(x)[0]; pv = prob[1:]
        J = np.diag(pv) - np.outer(pv, pv); ns = (prob[1] - prob[2]) / 2
        return np.r_[J @ (f[:3] * .02), f[3] * .02 * ns, 2 * f[4:]]
    H = approx_derivative(gradient, x, method='3-point', abs_step=np.r_[np.repeat(2e-5, 4), np.repeat(2e-7, 3)])
    he = H[:4, :4] - H[:4, 4:] @ np.linalg.solve(H[4:, 4:], H[4:, :4])
    w, V = np.linalg.eigh((he + he.T) / 2)
    prob = m.unpack(x)[0]
    print("U=%s probs (e,p+,p-,d)=%s" % (U, np.array2string(prob, precision=3)))
    print("  coordinates: x0=ln(p+/e), x1=ln(p-/e), x2=ln(d/e), x3=theta")
    for k in range(4):
        print("  eig %.3e  vec %s" % (w[k], np.array2string(V[:, k], precision=3, suppress_small=True)))
