"""Continue the mean-filling-one KR EP branch (package code, read-only) beyond the archived endpoints,
to test whether a retuned EP reaches the local-moment (suppressed double occupancy) regime.
usage: python3 -I extend_n1.py {V_O|W} umax out.json"""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kr_package" / "src"))
import matrix_saddle as m

control, umax, out = sys.argv[1], float(sys.argv[2]), Path(sys.argv[3])
src = (Path(__file__).resolve().parents[1] / "kr_package" / "results")
rows0 = json.loads((src / ("ep_n1.json" if control == "V_O" else "ep_n1_W.json")).read_text())
r0 = rows0[-1]
s = m.Saddle(r0["mu"], 32)
if control == "W":
    s.set_W(r0["impurity_hybridization"])
y = np.array(r0["y"]); U = r0["U"]; step = 0.0005
rows = [dict(r0, source="archived endpoint")]
t0 = time.time()
while U < umax - 1e-12 and time.time() - t0 < 3300:
    Ut = min(U + step, umax)
    r = m.joint_ep(s, Ut, y, 1.0, control)
    keep = {k: r[k] for k in ("U", "success", "mu", "epsilon", "auxiliary_vertex", "impurity_hybridization", "z", "n_d",
                              "double_occupancy", "Z_eigenvalues", "density_eigenvalues", "lambda_matrix", "probabilities",
                              "singular_values", "F_zz", "joint_residual_max", "x", "y")}
    dn = r["density_eigenvalues"]; keep["det_n"] = dn[0] * dn[1]
    print(json.dumps({k: keep[k] for k in ("U", "success", "epsilon", "auxiliary_vertex", "impurity_hybridization", "mu", "z",
                                           "double_occupancy", "det_n", "Z_eigenvalues", "joint_residual_max")}), flush=True)
    if r["success"]:
        rows.append(keep); y = np.array(r["y"]); U = Ut; step = min(step * 1.4, 0.005)
    else:
        step /= 2
        if step < 2e-6:
            keep["failed_attempt"] = True; rows.append(keep); break
    out.write_text(json.dumps(rows, indent=1))
out.write_text(json.dumps(rows, indent=1))
print("done", U, "elapsed %.0f s" % (time.time() - t0))
