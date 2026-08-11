#!/usr/bin/env python3
"""Render the finite-U full-k EP certificate at journal print scale.

Only archived gate summaries are consumed.  The many-body calculation is not
rerun.  The discriminant and pole-displacement panels use dimensionless
normalizations because their overall scales do not affect the winding or pole
permutation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/prb_figure_matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent


def read_json(name: str) -> dict:
    with (ROOT / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def center_row(name: str) -> tuple[float, float, float, float, float]:
    data = read_json(name)
    center = data["predicted_center"]
    return (
        float(data["parameters"]["U"]),
        float(center["z"][0]),
        float(center["z"][1]),
        float(center["epsilon_d"]),
        float(center["W"]),
    )


mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9.2,
        "axes.labelsize": 10.0,
        "axes.titlesize": 9.4,
        "xtick.labelsize": 8.4,
        "ytick.labelsize": 8.4,
        "legend.fontsize": 8.0,
        "axes.linewidth": 0.9,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def panel_tag(axis, text: str) -> None:
    axis.text(
        -0.15,
        1.04,
        text,
        transform=axis.transAxes,
        fontweight="bold",
        fontsize=11,
        ha="left",
        va="bottom",
    )


def tex_sci(value: float) -> str:
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10.0 ** exponent)
    return rf"{mantissa:.2f}\times10^{{{exponent}}}"


contour_data = read_json("finiteU_fullk_contour_summary.json")
contour = contour_data["contour"]
disc = np.array([complex(*row["D"]) for row in contour])
poles = np.array(
    [[complex(*pole) for pole in row["poles"]] for row in contour],
    dtype=complex,
)
center = contour_data["parameters"]["center"]
z_ep = complex(*center["z"])

u_rows = [
    (0.0, 0.011972834346, -0.002807216274, 0.015170349783, 0.100000000000),
    center_row("finiteU_fullk_center_U00010.json"),
    center_row("finiteU_fullk_center_U00015.json"),
    center_row("finiteU_fullk_center_U00020.json"),
    center_row("finiteU_fullk_center_U00025.json"),
]
u_rows = np.asarray(u_rows, dtype=float)
u_over_gamma = u_rows[:, 0] / 0.0025

kept = read_json("finiteU_fullk_kept_calibration.json")["reports"]
nkeep = np.array([row["nkeep"] for row in kept], dtype=float)
sigma_error = 100.0 * np.array(
    [row["maximum_errors"]["Sigma"] for row in kept], dtype=float
)
sumrule_error = 100.0 * np.array(
    [row["maximum_errors"]["sumrule"] for row in kept], dtype=float
)

disc_scale = float(np.max(np.abs(disc)))
pole_scale = float(np.max(np.abs(poles - z_ep)))
disc_hat = disc / disc_scale
pole_hat = (poles - z_ep) / pole_scale

fig = plt.figure(figsize=(7.08, 4.35), constrained_layout=True)
grid = fig.add_gridspec(2, 2, hspace=0.24, wspace=0.24)

# (a) Brouwer-degree contour, normalized by an irrelevant positive scale.
ax = fig.add_subplot(grid[0, 0])
ax.plot(disc_hat.real, disc_hat.imag, color="#2455A4", lw=2.0)
for idx in (0, 16, 32, 48):
    nxt = (idx + 1) % (len(disc_hat) - 1)
    ax.annotate(
        "",
        xy=(disc_hat[nxt].real, disc_hat[nxt].imag),
        xytext=(disc_hat[idx].real, disc_hat[idx].imag),
        arrowprops=dict(arrowstyle="-|>", color="#2455A4", lw=1.0),
    )
ax.scatter([0], [0], marker="x", s=38, lw=1.3, color="black", zorder=4)
ax.axhline(0, color="0.82", lw=0.7, zorder=0)
ax.axvline(0, color="0.82", lw=0.7, zorder=0)
ax.set_xlabel(r"$\mathrm{Re}\,\widehat{\mathcal{D}}_{\rm loc}$")
ax.set_ylabel(r"$\mathrm{Im}\,\widehat{\mathcal{D}}_{\rm loc}$")
ax.set_box_aspect(1.0)
ax.text(0.04, 0.94, r"$W_C[\mathcal{D}_{\rm loc}]=-1$", transform=ax.transAxes,
        ha="left", va="top",
        bbox=dict(fc="white", ec="none", alpha=0.86, pad=0.2))
ax.text(
    0.96,
    0.05,
    rf"$D_*={tex_sci(disc_scale)}$",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=8.1,
    bbox=dict(fc="white", ec="none", alpha=0.86, pad=0.2),
)
panel_tag(ax, "(a)")

# (b) Pole exchange relative to the interacting EP center.
ax = fig.add_subplot(grid[0, 1])
colors = ("#B63B2E", "#1B8A72")
for branch in range(2):
    dz = pole_hat[:, branch]
    ax.plot(dz.real, dz.imag, color=colors[branch], lw=1.8,
            label=rf"tracked $z_{{{branch+1}}}$")
    ax.scatter(dz[0].real, dz[0].imag, s=30, marker="o",
               facecolor="white", edgecolor=colors[branch], lw=1.1, zorder=4)
    ax.scatter(dz[-2].real, dz[-2].imag, s=28, marker="s",
               facecolor=colors[branch], edgecolor=colors[branch], zorder=4)
ax.scatter([0], [0], s=72, marker="*", color="black", zorder=5,
           label=r"$z_{\rm EP}$")
ax.axhline(0, color="0.82", lw=0.7, zorder=0)
ax.axvline(0, color="0.82", lw=0.7, zorder=0)
ax.set_xlabel(r"$\mathrm{Re}[(z-z_{\rm EP})/z_*]$")
ax.set_ylabel(r"$\mathrm{Im}[(z-z_{\rm EP})/z_*]$")
ax.legend(frameon=False, loc="upper left", handlelength=1.6)
ax.text(0.53, 0.05, r"exchange $(12)$", transform=ax.transAxes)
ax.text(
    0.53,
    0.13,
    rf"$z_*={tex_sci(pole_scale)}$",
    transform=ax.transAxes,
    fontsize=8.1,
)
panel_tag(ax, "(b)")

# (c) Exact center continuation, without a competing right-hand axis.
ax = fig.add_subplot(grid[1, 0])
ax.plot(u_over_gamma, 1.0e3 * u_rows[:, 1], "o-", ms=4.2,
        lw=1.8, color="#2455A4", label=r"$10^3\,\mathrm{Re}\,z_{\rm EP}$")
ax.plot(u_over_gamma, -1.0e3 * u_rows[:, 2], "s-", ms=4.0,
        lw=1.7, color="#B63B2E", label=r"$-10^3\,\mathrm{Im}\,z_{\rm EP}$")
ax.set_xlabel(r"$U/\bar\Gamma_0$")
ax.set_ylabel("pole-center components")
ax.set_xlim(-0.04, 1.04)
ax.legend(frameon=False, loc="center left")
panel_tag(ax, "(c)")

# (d) Separate retuning and kept-space diagnostics; no inset overlaps.
outer = grid[1, 1].subgridspec(2, 1, hspace=0.10)
ax_w = fig.add_subplot(outer[0])
ax_e = fig.add_subplot(outer[1])
ax_w.plot(u_over_gamma, u_rows[:, 4], "^-", ms=4.2, lw=1.7,
          color="#1B8A72")
ax_w.set_ylabel(r"retuned $W$")
ax_w.set_xlim(-0.04, 1.04)
ax_w.set_ylim(0.088, 0.102)
ax_w.tick_params(labelbottom=False)
panel_tag(ax_w, "(d)")

ax_e.plot(nkeep, sigma_error, "o-", ms=3.8, lw=1.5,
          color="#6A3D9A", label=r"$\Sigma_U$")
ax_e.plot(nkeep, sumrule_error, "s--", ms=3.5, lw=1.3,
          color="0.35", label="sum rule")
ax_e.set_yscale("log")
ax_e.set_xlabel(r"kept states $N_{\rm keep}$")
ax_e.set_ylabel("max. error (%)")
ax_e.set_xticks([320, 640, 960])
ax_e.legend(frameon=False, loc="lower left", ncol=2, handlelength=1.4)

for axis in (ax, ax_w, ax_e):
    axis.tick_params(direction="in", top=True, right=True)

fig.savefig(ROOT / "Fig3_finiteU_fullk_exact.pdf", bbox_inches="tight")
fig.savefig(ROOT / "Fig3_finiteU_fullk_exact.png", dpi=400, bbox_inches="tight")
print("wrote Fig3_finiteU_fullk_exact.pdf and .png")
print(f"D_star={disc_scale:.12e}; z_star={pole_scale:.12e}")
