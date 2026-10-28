import os
"""Same-control comparison at fixed V_O, controls (eps_d, W), mu = 0, T = 0:
   (1) exact finite-chain embedding (paper, converged App. C centers, read from the archive),
   (2) KR saddle on the same four z-shifted 12-site chains (kr_on_chain.py),
   (3) KR saddle on the continuum bath (package code, control 'W'),
then extend (3) in U and certify it on the paper's Fig. 5 ellipse (half-widths 1e-4 in eps_d, 1e-3 in W)."""
import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kr_on_chain as K          # noqa: E402  (inserts the package src path itself)
import kr_certificate as C       # noqa: E402
m = K.m
REPO = (Path(os.environ.get("EPT_REPO", "../../../../..")) / "reproducibility/full_k_finite_u")
OUT = HERE.parent / "results"; OUT.mkdir(exist_ok=True)

if "EPT_REPO" in os.environ:
    ed = {}
    for tag, U in (("00010", 0.001), ("00015", 0.0015), ("00020", 0.002), ("00025", 0.0025)):
        c = json.loads((REPO / f"finiteU_fullk_center_U{tag}.json").read_text())["predicted_center"]
        ed[U] = dict(eps=c["epsilon_d"], W=c["W"], z=c["z"])
else:
    # Exact copies of the archived center values already tabulated in this package.
    ed = {float(k): v for k,v in json.loads((HERE.parent / "reference_ed_centers.json").read_text()).items()}


def kr_chain(U, guess, maxit=40):
    v = np.array(guess, float); x0s = None
    for it in range(maxit):
        eps, W = v[2], v[3]
        sd = K.saddles(U, eps, W, x0s); x0s = [np.array(r["x"]) for r in sd]
        Rinv = [np.linalg.inv(np.array(r["R"])) for r in sd]; Ls = [np.array(r["lambda_matrix"]) for r in sd]
        A = np.mean([Ri @ Ri for Ri in Rinv], axis=0); Bm = np.mean([Ri @ L @ Ri for Ri, L in zip(Rinv, Ls)], axis=0)
        vnew, resid = K.ep_with_frozen_fields(A, Bm - eps * A, v)
        dv = np.max(np.abs(vnew - v)); v = vnew
        if dv < 1e-13:
            break
    return dict(U=U, z=[v[0], v[1]], eps=v[2], W=v[3], iterations=it + 1, last_step=float(dv), ep_residual=float(resid),
                saddles_ok=all(r["success"] for r in sd), n_d=float(np.mean([r["n_d"] for r in sd])),
                D=float(np.mean([r["double_occupancy"] for r in sd])))


if __name__ == "__main__":
    t0 = time.time()
    res = dict(ed=ed, kr_chain=[], kr_cont_W=[])
    g = [m.ZEP0.real, m.ZEP0.imag, m.EPS0, 0.10]
    for U in (0.001, 0.0015, 0.002, 0.0025):
        r = kr_chain(U, g); g = [r["z"][0], r["z"][1], r["eps"], r["W"]]
        res["kr_chain"].append(r)
        print("KR-chain", json.dumps(r), flush=True)
    s = m.Saddle(0.0, 32)
    x = s.initial(m.EPS0, m.VO0)
    y = np.r_[x, m.ZEP0.real / .01, m.ZEP0.imag / .01, m.EPS0 / .01, 1.0]
    for U in (0.0, 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.0035, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.08, 0.12, 0.2):
        r = m.joint_ep(s, U, y, None, "W")
        keep = {k: r[k] for k in ("U", "epsilon", "auxiliary_vertex", "impurity_hybridization", "mu", "z", "n_d", "double_occupancy",
                                   "Z_eigenvalues", "density_eigenvalues", "lambda_matrix", "R", "x", "y", "F", "F_z", "F_zz",
                                   "singular_values", "joint_residual_max", "success")}
        res["kr_cont_W"].append(keep)
        print("KR-cont-W U=%.4f ok=%s eps=%.8f W=%.8f z=%.8f%+.8fi n_d=%.5f D=%.2e res=%.1e"
              % (U, r["success"], r["epsilon"], r["impurity_hybridization"], r["z"][0], r["z"][1], r["n_d"], r["double_occupancy"],
                 r["joint_residual_max"]), flush=True)
        if not r["success"]:
            break
        y = np.array(r["y"])
    (OUT / "compare_bath.json").write_text(json.dumps(res, indent=1))
    # certificate of the continuum KR EP at U = 0.0025 on the paper's (eps_d, W) ellipse
    row = next(r for r in res["kr_cont_W"] if abs(r["U"] - 0.0025) < 1e-12)
    summ, recs = C.certificate(row, "W", 1e-4, 1e-3, samples=128, order=32)
    (OUT / "cert_cont_W_U0.0025.json").write_text(json.dumps(dict(ep=row, summary=summ, records=recs), indent=1))
    print("CERT (eps_d, W) U=0.0025:", json.dumps(summ, indent=1), flush=True)
    print("elapsed %.0f s" % (time.time() - t0))
