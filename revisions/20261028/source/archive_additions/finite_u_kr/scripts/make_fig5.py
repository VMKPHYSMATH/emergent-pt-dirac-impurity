"""Main-text Fig. 6 (file name kept from the draft numbering): finite-U survival of the full-k resonance EP (continuum KR saddle + same-bath comparison).
Reads only stored results; prints every number quoted in the LaTeX inserts."""
import json
from pathlib import Path
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
PKG = (Path(__file__).resolve().parents[1] / "kr_package" / "results")
FIG = HERE.parent / "fig"; FIG.mkdir(exist_ok=True)
G0 = 0.0025                      # U = 0 common half-width W^2/4
W0, EPS0 = 0.10, 0.0151703497834
HF_CONT = (-7.6314, -0.5510)     # first-order Fock slopes (dW/dU, deps/dU), continuum rho, fixed V_O
HF_CHAIN = (-4.5012, -0.3472)    # same with the z-averaged 12-site chain rho
ED_FILL = {0.0: 0.020652, 0.0025: 0.017064}   # App. C, N = 12, T = 0, mu = 0

mpl.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "font.size": 9.0, "axes.labelsize": 9.6, "axes.titlesize": 9.4, "xtick.labelsize": 8.2,
                     "ytick.labelsize": 8.2, "legend.fontsize": 7.6, "axes.linewidth": 0.9, "xtick.direction": "in",
                     "ytick.direction": "in", "xtick.top": True, "ytick.right": True, "pdf.fonttype": 42, "ps.fonttype": 42})
C1, C2, C3, CK = "#1f4e8c", "#c8511b", "#2e7d32", "#333333"

mu0 = json.loads((PKG / "ep_mu000.json").read_text())
cert = json.loads((RES / "cert_mu0_VO_U0.2.json").read_text())
cmp_ = json.loads((RES / "compare_bath.json").read_text())
ed = {float(k): v for k, v in cmp_["ed"].items()}
krc = cmp_["kr_chain"]; krw = [r for r in cmp_["kr_cont_W"] if r["success"]]

U = np.array([r["U"] for r in mu0]); x = U / G0
z = np.array([complex(*r["z"]) for r in mu0]); eps = np.array([r["epsilon"] for r in mu0]); vo = np.array([r["auxiliary_vertex"] for r in mu0])
nd = np.array([r["n_d"] for r in mu0]); D = np.array([r["double_occupancy"] for r in mu0]); Zm = np.array([min(r["Z_eigenvalues"]) for r in mu0])
lam = np.array([np.array(r["lambda_matrix"]) for r in mu0])
ltl = np.hypot(lam[:, 0, 1], (lam[:, 0, 0] - lam[:, 1, 1]) / 2)

fig, axs = plt.subplots(2, 3, figsize=(7.1, 4.75), constrained_layout=True)

# (a) pole exchange at U = 0.2 with the saddle re-solved on the loop
ax = axs[0, 0]
zep = complex(*cert["ep"]["z"]); P = np.array([[complex(*p) for p in r["poles"]] for r in cert["records"]])
for j, (c, lab) in enumerate(((C1, "tracked $z_1$"), (C2, "tracked $z_2$"))):
    w = 1e4 * (P[:, j] - zep)
    ax.plot(w.real, w.imag, color=c, lw=1.4, ls="-" if j == 0 else "--", label=lab)
    k = len(w) // 3
    ax.annotate("", xy=(w[k + 1].real, w[k + 1].imag), xytext=(w[k].real, w[k].imag),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=1.0, mutation_scale=9))
    ax.plot(w[0].real, w[0].imag, "o", mfc="white", mec=c, ms=5, zorder=5)
    ax.plot(w[-1].real, w[-1].imag, "s", color=c, ms=3.5, zorder=6)
ax.plot(0, 0, "*", color=CK, ms=9, zorder=7)
ax.set_xlabel(r"$10^4\,\mathrm{Re}(z-z_{\rm EP})$"); ax.set_ylabel(r"$10^4\,\mathrm{Im}(z-z_{\rm EP})$")
ax.set_title(r"(a) Pole exchange, $U=80\,\bar\Gamma_0$")
ax.set_ylim(-0.95, 1.12)
ax.legend(loc="upper center", ncol=2, frameon=False, handlelength=1.8, borderaxespad=0.2, columnspacing=1.0)

def ux(ax):
    ax.set_xscale("symlog", linthresh=0.1, linscale=0.45); ax.set_xlim(0, 100)
    ax.set_xticks([0, 0.1, 1, 10, 100]); ax.set_xticklabels(["0", "0.1", "1", "10", "100"])
    ax.set_xlabel(r"$U/\bar\Gamma_0$")

# (b) EP energy and gate along the mu = 0 branch
ax = axs[0, 1]
ax.plot(x, 1e3 * z.real, "-o", color=C1, ms=3, lw=1.3, label=r"$\mathrm{Re}\,z_{\rm EP}$")
ax.plot(x, 1e3 * eps, "--s", color=C2, ms=3, lw=1.3, label=r"$\epsilon_d$")
ax.plot(x, -1e3 * z.imag, ":^", color=C3, ms=3, lw=1.5, label=r"$-\mathrm{Im}\,z_{\rm EP}$")
ux(ax); ax.set_ylabel(r"energy $(10^{-3})$"); ax.set_title("(b) Retuned EP, $W=0.10$, $\\mu=0$")
ax.legend(loc="upper left", frameon=False)

# (c) drive vertex and interaction-induced spin field
ax = axs[0, 2]
ax.plot(x, vo, "-o", color=C1, ms=3, lw=1.3, label=r"$V_O$")
ux(ax); ax.set_ylabel(r"$V_O$", color=C1); ax.tick_params(axis="y", colors=C1)
ax2 = ax.twinx(); ax2.plot(x, 1e4 * ltl, "--s", color=C2, ms=3, lw=1.3, label=r"$10^4|\vec\Lambda_f|$")
ax2.set_ylabel(r"$10^4\,|\vec\Lambda_f|$", color=C2); ax2.tick_params(axis="y", colors=C2, direction="in")
ax.set_title("(c) Drive retuning and spin field")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False)

# (d) filling, double occupancy and quasiparticle weight
ax = axs[1, 0]
ax.semilogy(x, nd, "-o", color=C1, ms=3, lw=1.3, label=r"$\langle n_d\rangle$")
ax.semilogy(x, D, "--s", color=C2, ms=3, lw=1.3, label=r"$D_d=\langle n_{d\uparrow}n_{d\downarrow}\rangle$")
ax.semilogy(x[1:], 1 - Zm[1:], ":^", color=C3, ms=3, lw=1.5, label=r"$1-Z_{\min}$")
ux(ax); ax.set_ylim(1e-9, 0.2); ax.set_title("(d) Nearly empty impurity")
ax.legend(loc="lower left", frameon=False)

# (e) same-control comparison at weak U, controls (eps_d, W) at fixed V_O
ax = axs[1, 1]
ue = np.r_[0, sorted(ed)]; we = np.r_[0, [ed[u]["W"] - W0 for u in sorted(ed)]]
uc = np.r_[0, [r["U"] for r in krc]]; wc = np.r_[0, [r["W"] - W0 for r in krc]]
uk = np.array([r["U"] for r in krw if r["U"] <= 0.0025 + 1e-12]); wk = np.array([r["impurity_hybridization"] - W0 for r in krw if r["U"] <= 0.0025 + 1e-12])
uu = np.linspace(0, 0.0026, 20)
ax.plot(1e3 * uu, 1e3 * HF_CHAIN[0] * uu, color="0.5", lw=0.9, ls=(0, (4, 2)), label=r"first order, chain $\rho$")
ax.plot(1e3 * uu, 1e3 * HF_CONT[0] * uu, color="0.5", lw=0.9, ls=(0, (1, 1.5)), label=r"first order, continuum $\rho$")
ax.plot(1e3 * ue, 1e3 * we, "o", color=CK, ms=5, mfc="white", label="ED, 12-site chains")
ax.plot(1e3 * uc, 1e3 * wc, "-x", color=C2, ms=4.5, lw=1.1, label="KR, same chains")
ax.plot(1e3 * uk, 1e3 * wk, "-s", color=C1, ms=3, lw=1.3, label="KR, continuum")
ax.set_xlabel(r"$10^3\,U$"); ax.set_ylabel(r"$10^3\,(W-W_0)$"); ax.set_xlim(0, 2.6); ax.set_ylim(-25, 1)
ax.set_title(r"(e) Same controls $(\epsilon_d,W)$")
ax.legend(loc="lower left", frameon=False, fontsize=7.0, handlelength=2.0, borderaxespad=0.2, labelspacing=0.3)

# (f) impurity filling at the EP: the chains under-resolve the occupied bath
ax = axs[1, 2]
ax.plot(1e3 * np.array(sorted(ED_FILL)), [ED_FILL[u] for u in sorted(ED_FILL)], "o", color=CK, ms=5, mfc="white", label="ED, 12-site chains")
ax.plot(1e3 * np.r_[0, [r["U"] for r in krc]], np.r_[ED_FILL[0.0], [r["n_d"] for r in krc]], "-x", color=C2, ms=4.5, lw=1.1, label="KR, same chains")
ax.plot(1e3 * uk, [r["n_d"] for r in krw if r["U"] <= 0.0025 + 1e-12], "-s", color=C1, ms=3, lw=1.3, label="KR, continuum")
ax.set_xlabel(r"$10^3\,U$"); ax.set_ylabel(r"$\langle n_d\rangle$ at the EP"); ax.set_xlim(-0.08, 2.6); ax.set_ylim(0, 0.04)
ax.set_title("(f) Impurity filling at the EP")
ax.legend(loc="lower left", frameon=False)

for a in axs.flat:
    a.grid(False)
fig.savefig(FIG / "Fig5_finiteU_survival.pdf"); fig.savefig(FIG / "Fig5_finiteU_survival.png", dpi=220)

# ---------- numbers quoted in the inserts ----------
def fit_slope(us, ys):
    A = np.c_[us, np.square(us)]; return np.linalg.lstsq(A, ys, rcond=None)[0]
print("ED-chain quadratic fit dW/dU, deps/dU :", fit_slope(np.array(sorted(ed)), np.array([ed[u]["W"] - W0 for u in sorted(ed)]))[0],
      fit_slope(np.array(sorted(ed)), np.array([ed[u]["eps"] - EPS0 for u in sorted(ed)]))[0])
print("KR-chain quadratic fit dW/dU, deps/dU :", fit_slope(uc[1:], wc[1:])[0], fit_slope(uc[1:], np.array([r["eps"] - EPS0 for r in krc]))[0])
print("KR-cont quadratic fit (U<=0.001)     :", fit_slope(uk[1:3], wk[1:3])[0], fit_slope(uk[1:3], np.array([r["epsilon"] - EPS0 for r in krw[1:3]]))[0])
print("\n U       ED W        KR-ch W     KR-co W     | ED eps      KR-ch eps   KR-co eps  | W ratio ch/ED  eps ratio ch/ED | cont/chain W, eps")
for r in krc:
    u = r["U"]; e = ed[u]; k = next(q for q in krw if abs(q["U"] - u) < 1e-12)
    rw = (r["W"] - W0) / (e["W"] - W0); re_ = (r["eps"] - EPS0) / (e["eps"] - EPS0)
    cw = (k["impurity_hybridization"] - W0) / (r["W"] - W0); ce = (k["epsilon"] - EPS0) / (r["eps"] - EPS0)
    print(" %.4f  %.7f  %.7f  %.7f | %.7f  %.7f  %.7f | %.4f        %.4f          | %.3f %.3f"
          % (u, e["W"], r["W"], k["impurity_hybridization"], e["eps"], r["eps"], k["epsilon"], rw, re_, cw, ce))
u = 0.0025; e = ed[u]; r = krc[-1]; k = next(q for q in krw if abs(q["U"] - u) < 1e-12)
zb = complex(*k["z"]) + complex(*e["z"]) - complex(*r["z"])
print("\nheuristic transferred correction (UNCERTIFIED) U=0.0025: W=%.6f eps=%.6f z=%.6f%+.6fi" % (k["impurity_hybridization"] + e["W"] - r["W"], k["epsilon"] + e["eps"] - r["eps"], zb.real, zb.imag))
print("z at U=0.0025: ED %s  KR-chain %s  KR-cont %s" % (np.round(complex(*e["z"]), 7), np.round(complex(*r["z"]), 7), np.round(complex(*k["z"]), 7)))
print("\nmu=0, fixed W branch:")
for r_, l_ in zip(mu0, ltl):
    print("  U=%.4f U/G=%6.2f eps=%.6f V_O=%.6f z=%.6f%+.6fi n_d=%.5f D=%.2e Zmin=%.5f |Lam_tl|=%.3e" % (r_["U"], r_["U"] / G0, r_["epsilon"], r_["auxiliary_vertex"], r_["z"][0], r_["z"][1], r_["n_d"], r_["double_occupancy"], min(r_["Z_eigenvalues"]), l_))
print("\nfixed V_O (eps, W) continuum branch:")
for r_ in krw:
    print("  U=%.4f eps=%.6f W=%.6f z=%.6f%+.6fi n_d=%.5f D=%.2e U/Gself=%.1f" % (r_["U"], r_["epsilon"], r_["impurity_hybridization"], r_["z"][0], r_["z"][1], r_["n_d"], r_["double_occupancy"], r_["U"] / (r_["impurity_hybridization"] ** 2 / 4)))
