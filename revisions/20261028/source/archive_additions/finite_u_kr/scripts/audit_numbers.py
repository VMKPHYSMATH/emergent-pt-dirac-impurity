"""Recompute every number quoted in inserts/finiteU_KR_inserts.tex from the stored results; print quoted vs recomputed."""
import json, csv
from pathlib import Path
import numpy as np

H = Path(__file__).resolve().parent
PK = (Path(__file__).resolve().parents[1] / "kr_package" / "results")
mu0 = json.loads((PK / "ep_mu000.json").read_text())
n1 = json.loads((PK / "ep_n1.json").read_text()); n1W = json.loads((PK / "ep_n1_W.json").read_text())
val = json.loads((PK / "validation.json").read_text())
cmp_ = json.loads((H.parent / "results/compare_bath.json").read_text())
cs = json.loads((H.parent / "results/cert_summary.json").read_text())
cW = json.loads((H.parent / "results/cert_cont_W_U0.0025.json").read_text())["summary"]
G0, W0, E0, BW = 0.0025, 0.10, 0.0151703497834, 0.617
rows = []
def q(label, quoted, value):
    rows.append((label, quoted, value))

r = mu0[-1]; c2 = cs["mu0_VO_U0.2"]; c02 = cs["mu0_VO_U0.02"]
q("U/G0 at U=0.2 (80)", 80, r["U"] / G0); q("U/bandwidth (0.32)", 0.32, r["U"] / BW)
q("z_EP U=0.2 (0.02963-0.00261i)", "0.02963-0.00261i", "%.5f%+.5fi" % tuple(r["z"]))
q("eps U=0.2 (0.03193)", 0.03193, r["epsilon"]); q("V_O U=0.2 (1.8389)", 1.8389, r["auxiliary_vertex"])
q("threshold gap (4.15)", 4.15, c2["extras"]["gap_in_widths"]); q("passivity margin (0.847)", 0.847, c2["extras"]["passivity_margin"])
q("V_O/(eO-Om) (0.031)", 0.031, c2["extras"]["VO_offshell"])
dens = r["density_eigenvalues"]; q("D U=0.2 (2.0e-8)", 2.0e-8, r["double_occupancy"]); q("det n (2.2e-7 / 2.23e-7)", 2.23e-7, dens[0] * dens[1])
q("D/det n (0.09)", 0.09, r["double_occupancy"] / (dens[0] * dens[1]))
q("n_d U=0.2 (0.0118)", 0.0118, r["n_d"]); q("Zmin U=0.2 (>=0.994)", 0.994, min(r["Z_eigenvalues"]))
lam = np.array(r["lambda_matrix"]); q("|Lambda| U=0.2 (2.2e-4)", 2.2e-4, np.hypot(lam[0, 1], (lam[0, 0] - lam[1, 1]) / 2))
for k, c in (("U=0.02", c02), ("U=0.2", c2)):
    cc = c["certificate"]
    q(k + " winding / half / perm", "+1, +1/2, (12)", (cc["winding"], cc["half_winding"], cc["endpoint_permutation"]))
    q(k + " min boundary, max phase, min Fzz", "", (cc["minimum_boundary_modulus"], cc["maximum_phase_increment"], cc["minimum_F_zz"]))
    q(k + " max Fz crit, min sep, pole res, assign, saddle res", "", (cc["maximum_F_z_at_critical"], cc["minimum_pole_separation"], cc["maximum_pole_residual"], cc["maximum_assignment_ratio"], cc["maximum_saddle_residual"]))
    q(k + " sv, gap, margin, offshell", "", (c["ep"]["sv"], c["extras"]["gap_in_widths"], c["extras"]["passivity_margin"], c["extras"]["VO_offshell"]))
    q(k + " stability", "positive", (min(c["stability"]["constrained_curvature_eigenvalues"]), c["stability"]["transverse_curvature"], c["stability"]["local_minimum"]))
# weak-U comparison
ed = cmp_["ed"]; krc = cmp_["kr_chain"]; krw = [x for x in cmp_["kr_cont_W"] if x["success"]]
for x in krc:
    e = ed[str(x["U"])] if str(x["U"]) in ed else ed[repr(x["U"])]
    k = next(y for y in krw if abs(y["U"] - x["U"]) < 1e-12)
    q("U=%g ratio KRch/ED W, eps" % x["U"], "", ((x["W"] - W0) / (e["W"] - W0), (x["eps"] - E0) / (e["eps"] - E0)))
    q("U=%g ratio cont/chain W, eps" % x["U"], "", ((k["impurity_hybridization"] - W0) / (x["W"] - W0), (k["epsilon"] - E0) / (x["eps"] - E0)))
u = 0.0025; e = ed["0.0025"]; x = krc[-1]; k = next(y for y in krw if abs(y["U"] - u) < 1e-12)
zb = complex(*k["z"]) + complex(*e["z"]) - complex(*x["z"])
q("best W (0.0834)", 0.0834, k["impurity_hybridization"] + e["W"] - x["W"]); q("best eps (0.01393)", 0.01393, k["epsilon"] + e["eps"] - x["eps"])
q("best z (0.01168-0.00190i)", "0.01168-0.00190i", "%.5f%+.5fi" % (zb.real, zb.imag)); q("ED W (0.0904)", 0.0904, e["W"])
q("cont n_d U=0 (0.03468), at U=0.0025 (0.02407 / 0.024), D (1.9e-6)", "", (krw[0]["n_d"], k["n_d"], k["double_occupancy"]))
w20 = next(y for y in krw if abs(y["U"] - 0.02) < 1e-12)
q("W-branch U=0.02: W 0.03104 eps 0.01054 z 0.01021-0.00025i n 0.0039", "", (w20["impurity_hybridization"], w20["epsilon"], w20["z"], w20["n_d"]))
q("cert (eps,W) U=0.0025 winding, half, perm, phase, assign", "-1, -1/2, (12), 0.144, 0.237", (cW["winding"], cW["half_winding"], cW["endpoint_permutation"], cW["maximum_phase_increment"], cW["maximum_assignment_ratio"]))
# n_d = 1 point
s = val["selected"]; dn = s["density_eigenvalues"]
q("n_d=1: z, eps, V_O, mu", "0.006535-0.002983i, 0.008762, 0.94586, 0.006657", (s["z"], s["epsilon"], s["auxiliary_vertex"], s["mu"]))
q("n_d=1: D, det n, P_single, Z", "0.229, 0.241, 0.542, 0.9974", (s["double_occupancy"], dn[0] * dn[1], s["n_d"] - 2 * s["double_occupancy"], s["Z_eigenvalues"]))
q("n_d=1: singular values", "1.54e-3, 2e-18", s["singular_values"])
q("n_d=1: exchange error, half-winding", "8.5e-16, -1/2", (val["exchange"]["pole_exchange_error"], val["exchange"]["half_winding"]))
st = val["selected_stability"]; q("n_d=1: min curvature, transverse", "7.8e-5, 8.7e-5", (min(st["constrained_curvature_eigenvalues"]), st["transverse_curvature"]))
q("n_d=1 fixed-W branch: U max, D range", "0.003, 0.22<D<0.25", (n1[-1]["U"], min(x["double_occupancy"] for x in n1), max(x["double_occupancy"] for x in n1)))
q("n_d=1 fixed-V_O branch: U max, W max, D range", "0.004", (n1W[-1]["U"], n1W[-1]["impurity_hybridization"], min(x["double_occupancy"] for x in n1W), max(x["double_occupancy"] for x in n1W)))
for oc in val["occupied_checks"]:
    q("occupied U=%g n_d, D, dens, stable" % oc["U"], "", (oc["n_d"], oc["double_occupancy"], oc["density_eigenvalues"], oc["stability"]["local_minimum"]))
q("star check density/force", "1.1e-12, 1.5e-14", (val["star_check"]["density_error"], val["star_check"]["force_error"])); q("radial integral", "1.5e-16", val["radial_integral_error"])
# U=0 coherent coupling for comparison with |Lambda|
q("ratio |Lambda|/ReSigma_z (about a third)", "~1/3", 2.209e-4 / 6.50e-4)
w = max(len(a) for a, _, _ in rows)
for a, b, c in rows:
    print(a.ljust(w), "|", str(b).ljust(34), "|", c)
