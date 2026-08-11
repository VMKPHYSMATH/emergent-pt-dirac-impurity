#!/usr/bin/env python3
"""Generate a print-legible comparison of the frozen EP core and its control.

The figure uses the exact normalized 4x4 k=0 kernel stated in the manuscript.
Time traces are shown as small multiples rather than overlaid phase portraits.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/prb_figure_matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parent
EPS = -1.0
D_EP = 0.50
W = 0.75
BETAS = np.linspace(-1.5, 1.5, 401)

BLUE = "#2455A4"
RED = "#B63B2E"
GREEN = "#1B8A72"
ORANGE = "#E08214"
GREY = "#555555"
BRANCH_COLORS = (BLUE, RED, GREEN, ORANGE)


mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9.0,
        "axes.labelsize": 9.7,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8.1,
        "ytick.labelsize": 8.1,
        "legend.fontsize": 7.7,
        "axes.linewidth": 0.9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def kernel(beta0: float, coherent: float) -> np.ndarray:
    """Rotated compensated-core kernel in (d+, d-, c+, c-) order."""
    gamma = abs(float(beta0))
    return np.array(
        [
            [EPS + coherent, 1j * gamma, W, 0.0],
            [1j * gamma, EPS - coherent, 0.0, W],
            [W, 0.0, 0.0, 0.0],
            [0.0, W, 0.0, 0.0],
        ],
        dtype=complex,
    )


def eigensystem(beta0: float, coherent: float) -> tuple[np.ndarray, np.ndarray]:
    matrix = kernel(beta0, coherent)
    if np.linalg.norm(matrix - matrix.conj().T) < 1.0e-13:
        values, vectors = np.linalg.eigh(matrix)
    else:
        values, vectors = np.linalg.eig(matrix)
    vectors = vectors / np.linalg.norm(vectors, axis=0, keepdims=True)
    return values, vectors


def match(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    cost = np.abs(previous[:, None] - current[None, :]) ** 2
    _, columns = linear_sum_assignment(cost)
    return current[columns]


def sweep(coherent: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positive = BETAS[BETAS >= 0.0]
    negative = BETAS[BETAS < 0.0][::-1]

    def half(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        spectra = []
        kappas = []
        previous = None
        for beta0 in values:
            eigenvalues, right = eigensystem(float(beta0), coherent)
            if previous is None:
                eigenvalues = eigenvalues[np.argsort(eigenvalues.real)]
            else:
                eigenvalues = match(eigenvalues, previous)
            previous = eigenvalues.copy()
            spectra.append(eigenvalues)
            kappas.append(float(np.linalg.cond(right)))
        return np.asarray(spectra), np.asarray(kappas)

    pos_spectrum, pos_kappa = half(positive)
    neg_spectrum, neg_kappa = half(negative)
    spectrum = np.vstack((neg_spectrum[::-1], pos_spectrum))
    kappa = np.concatenate((neg_kappa[::-1], pos_kappa))
    return spectrum.real, spectrum.imag, kappa


def survival(beta0: float, coherent: float, tmax: float) -> tuple[np.ndarray, np.ndarray]:
    times = np.linspace(0.0, tmax, 720)
    initial = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    values = np.array(
        [initial.conj() @ (expm(-1j * kernel(beta0, coherent) * t) @ initial)
         for t in times]
    )
    scale = max(float(np.max(np.abs(values.real))), 1.0e-15)
    return times, values.real / scale


def tag(axis, label: str) -> None:
    axis.text(-0.19, 1.06, label, transform=axis.transAxes,
              fontweight="bold", fontsize=10.8, ha="left", va="bottom")


def spectrum_panel(slot, real: np.ndarray, imag: np.ndarray, label: str,
                   title: str, show_ep: bool) -> None:
    sub = slot.subgridspec(2, 1, hspace=0.08)
    ax_re = fig.add_subplot(sub[0])
    ax_im = fig.add_subplot(sub[1], sharex=ax_re)
    for branch, color in enumerate(BRANCH_COLORS):
        ax_re.plot(BETAS, real[:, branch], color=color, lw=1.35)
        ax_im.plot(BETAS, imag[:, branch], color=color, lw=1.35)
    for axis in (ax_re, ax_im):
        axis.axhline(0.0, color="0.82", lw=0.6, zorder=0)
        if show_ep:
            for location in (-0.5, 0.5):
                axis.axvline(location, color="0.25", ls=(0, (4, 2)), lw=0.9)
    if show_ep:
        for location in (-0.5, 0.5):
            ax_re.text(location, 0.96, "EP", transform=ax_re.get_xaxis_transform(),
                       ha="center", va="top", fontsize=7.6,
                       bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.2))
    ax_re.set_ylabel(r"$\mathrm{Re}\,E$")
    ax_im.set_ylabel(r"$\mathrm{Im}\,E$")
    ax_im.set_xlabel(r"$\beta_0$")
    ax_re.tick_params(labelbottom=False)
    ax_re.set_title(title, pad=3.0)
    tag(ax_re, label)


def kappa_panel(slot, kappa: np.ndarray, label: str, title: str,
                show_ep: bool) -> None:
    axis = fig.add_subplot(slot)
    axis.plot(BETAS, kappa, color=GREY, lw=2.0)
    if show_ep:
        for location in (-0.5, 0.5):
            axis.axvline(location, color="0.25", ls=(0, (4, 2)), lw=0.9)
            axis.text(location, 0.96, "EP", transform=axis.get_xaxis_transform(),
                      ha="center", va="top", fontsize=7.6,
                      bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.2))
    axis.set_xlabel(r"$\beta_0$")
    axis.set_ylabel(r"$\kappa_{\rm imp}$")
    axis.set_title(title, pad=3.0)
    axis.set_ylim(0.85, 1.08 * float(np.max(kappa)))
    tag(axis, label)


def trace_panel(slot, choices: list[tuple[float, str, str, float]],
                coherent: float, label: str, title: str) -> None:
    sub = slot.subgridspec(3, 1, hspace=0.08)
    axes = []
    for row, (beta0, text, color, tmax) in enumerate(choices):
        axis = fig.add_subplot(sub[row], sharex=axes[0] if axes else None)
        times, values = survival(beta0, coherent, tmax)
        axis.plot(times, values, color=color, lw=1.65)
        axis.axhline(0.0, color="0.82", lw=0.55, zorder=0)
        axis.set_ylim(-1.08, 1.08)
        axis.set_yticks([-1, 0, 1])
        axis.text(0.98, 0.82, text, transform=axis.transAxes,
                  ha="right", va="top", fontsize=7.6,
                  bbox=dict(fc="white", ec="none", alpha=0.86, pad=0.15))
        if row < 2:
            axis.tick_params(labelbottom=False)
        axes.append(axis)
    axes[1].set_ylabel(r"$\mathrm{Re}\,O(t)/\max$")
    axes[-1].set_xlabel(r"$t$")
    axes[0].set_title(title, pad=3.0)
    tag(axes[0], label)


real_ep, imag_ep, kappa_ep = sweep(D_EP)
real_control, imag_control, kappa_control = sweep(0.0)

fig = plt.figure(figsize=(7.08, 4.55), constrained_layout=True)
grid = fig.add_gridspec(2, 3, width_ratios=(1.18, 0.92, 1.15),
                        hspace=0.20, wspace=0.23)

spectrum_panel(grid[0, 0], real_ep, imag_ep, "(a)",
               "coherent EP core: spectrum", True)
kappa_panel(grid[0, 1], kappa_ep, "(b)",
            "eigenvector coalescence", True)
trace_panel(
    grid[0, 2],
    [(0.05, r"below: $\beta_0=0.05$", BLUE, 24.0),
     (0.50, r"EP: $\beta_0=0.50$", ORANGE, 24.0),
     (0.75, r"above: $\beta_0=0.75$", GREEN, 24.0)],
    D_EP,
    "(c)",
    "time-domain response",
)

spectrum_panel(grid[1, 0], real_control, imag_control, "(d)",
               "gain-loss-only control: spectrum", False)
kappa_panel(grid[1, 1], kappa_control, "(e)",
            "no coherent EP peak", False)
trace_panel(
    grid[1, 2],
    [(0.00, r"Hermitian: $\beta_0=0$", BLUE, 9.0),
     (0.25, r"weak: $\beta_0=0.25$", ORANGE, 9.0),
     (0.50, r"strong: $\beta_0=0.50$", GREEN, 9.0)],
    0.0,
    "(f)",
    "matched control response",
)

fig.savefig(ROOT / "Fig2_frozen_kernel_comparison.pdf", bbox_inches="tight")
fig.savefig(ROOT / "Fig2_frozen_kernel_comparison.png", dpi=400,
            bbox_inches="tight")
print("wrote Fig2_frozen_kernel_comparison.pdf and .png")
print(f"kappa(0)={kappa_ep[np.argmin(np.abs(BETAS))]:.12g}")
print(f"sampled kappa max={np.max(kappa_ep):.12g} at beta={BETAS[np.argmax(kappa_ep)]:.6g}")
