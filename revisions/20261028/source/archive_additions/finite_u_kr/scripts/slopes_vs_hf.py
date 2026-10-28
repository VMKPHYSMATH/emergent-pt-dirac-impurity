from pathlib import Path
"""Independent check of the package's KR saddle at small U against first-order Hartree-Fock (Fock) slopes
computed with my own continuum code (sbmf_fullk2), for both control pairs, at mu = 0."""
import sys, json, math
import numpy as np
SB = str(Path(__file__).resolve().parent)
PK = str(Path(__file__).resolve().parents[1] / "kr_package" / "src")
sys.path.insert(0, SB)
from sbmf_fullk import EP0, P0
from sbmf_fullk2 import density_matrix_T0, solve_ep_extra
rho = density_matrix_T0(EP0["eps"], EP0["VO"], P0)
x0 = np.array([EP0["z"].real, EP0["z"].imag, 0.10]); y0 = np.array([EP0["z"].real, EP0["z"].imag, EP0["VO"]])
def hf_slopes(unknown):
    out = []
    for U in (1e-6, -1e-6):
        ext = U * (np.trace(rho) * np.eye(2) - rho)
        if unknown == "W":
            ok, x, e = solve_ep_extra(EP0["VO"], ext, x0, unknown="W")
        else:
            ok, x, e = solve_ep_extra(None, ext, y0, unknown="VO", fixed=0.10)
        out.append((e, x[2], complex(x[0], x[1])))
    return [(out[0][i] - out[1][i]) / 2e-6 for i in range(3)]
s_vo = hf_slopes("VO"); s_w = hf_slopes("W")
print("HF (my continuum code), controls (eps,V_O) at fixed W=0.1: d eps/dU=%.4f  dV_O/dU=%.3f  dz/dU=%s" % (s_vo[0], s_vo[1], np.round(s_vo[2], 4)))
print("HF (my continuum code), controls (eps,W) at fixed V_O   : d eps/dU=%.4f  dW/dU=%.4f   dz/dU=%s" % (s_w[0], s_w[1], np.round(s_w[2], 4)))
# package KR results along mu=0, controls (eps,V_O): finite differences from the archived rows
rows = json.load(open(str(Path(PK).parent / "results" / "ep_mu000.json")))
r0, r1 = rows[0], rows[1]
dU = r1["U"] - r0["U"]
print("package KR (archived ep_mu000, U=0->1e-4)          : d eps/dU=%.4f  dV_O/dU=%.3f  dz/dU=%s"
      % ((r1["epsilon"] - r0["epsilon"]) / dU, (r1["auxiliary_vertex"] - r0["auxiliary_vertex"]) / dU,
         np.round(complex(r1["z"][0] - r0["z"][0], r1["z"][1] - r0["z"][1]) / dU, 4)))
# package KR, new run: controls (eps,W) at fixed V_O, mu = 0 (not in the archive)
sys.path.insert(0, PK)
import matrix_saddle as m
s = m.Saddle(0.0, 32)
x = s.initial(); y = np.r_[x, m.ZEP0.real / .01, m.ZEP0.imag / .01, m.EPS0 / .01, 1.0]
prev = None
for U in (0.0, 1e-4, 5e-4, 1e-3, 1.5e-3, 2e-3, 2.5e-3):
    r = m.joint_ep(s, U, y, None, "W")
    print("package KR (new run) mu=0, controls (eps,W): U=%.4f ok=%s eps=%.8f W=%.8f z=%.8f%+.8fi n=%.5f res=%.1e"
          % (U, r["success"], r["epsilon"], r["impurity_hybridization"], r["z"][0], r["z"][1], r["n_d"], r["joint_residual_max"]))
    if r["success"]:
        y = np.array(r["y"])
        if U == 1e-4 and prev is not None:
            print("    -> KR slope (U=0->1e-4): d eps/dU=%.4f dW/dU=%.4f" % ((r["epsilon"] - prev["epsilon"]) / 1e-4, (r["impurity_hybridization"] - prev["impurity_hybridization"]) / 1e-4))
        if U == 0.0: prev = r
print("converged block-Wilson centers (quadratic fit): d eps/dU=-0.3470 dW/dU=-4.459 ; W(U=0.0025)=0.090363 eps=0.014406")
