from pathlib import Path
import json, sys
import numpy as np
from scipy.optimize import root
from scipy.optimize._numdiff import approx_derivative
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kr_package" / "src"))
import matrix_saddle as m
# EP at fixed U, n_d = 1, W fixed: unknowns x, z, eps, V_O, mu (package joint_ep), at two quadrature orders,
# evaluated at U values bracketing the fold found with order 32.
RD = Path(__file__).resolve().parents[1] / "results"
rows = json.load(open(RD / "fold_n1_VO.json"))
top = max([r for r in rows if r["ok"]], key=lambda r: r["U"])
print("order-32 fold: U_c=%.7f at V_O=%.5f" % (top["U"], top["par"]))
ext = [r for r in json.load(open(RD / "ep_n1_extended_VO.json")) if r.get("success", True) and not r.get("failed_attempt")]
r0 = ext[-1]
for order in (32, 48):
    s = m.Saddle(r0["mu"], order)
    y = np.array(r0["y"]); out = []
    for U in (0.00300, 0.00305, 0.00308, 0.003085, 0.00309, 0.00311):
        r = m.joint_ep(s, U, y, 1.0, "V_O")
        out.append((U, r["success"], r["joint_residual_max"], r["auxiliary_vertex"]))
        if r["success"]:
            y = np.array(r["y"])
    print("order", order, ["U=%.6f ok=%s res=%.0e V_O=%.5f" % o for o in out])
