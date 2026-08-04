#!/usr/bin/env python3
"""Generate the retained frozen-kernel PRB Figs. 1, 2, and S1.

This executable is deliberately limited to the four-state eigenvalue,
condition-number, survival-amplitude, and Bloch diagnostics retained in v2.
It does not implement a physical density of states, an RG flow, or a Bethe
root/phase diagram. Those calculations have separate causal and algebraic
gates under ``reproducibility/``.

The frozen convention is

    eps_xi = -1, U = 2, r = 1,
    Gamma_PT = c_gamma |beta0|,
    Delta_eff = delta0 + c_delta |beta0|,
    V_eff = c_V |beta0|,

with ``c_gamma=1``, ``delta0=0.375``, ``c_delta=0.25``, and
``c_V=1.5``. The exact local-core condition is therefore
``|beta0|=delta0/(c_gamma-c_delta)=0.5``.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ptdirac_matplotlib")

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.linalg import expm
from scipy.optimize import linear_sum_assignment


@dataclass
class ModelParams:
    eps_xi: float = -1.0
    U: float = 2.0
    beta_values: np.ndarray = field(
        default_factory=lambda: np.linspace(-1.5, 1.5, 401)
    )
    lambda_soc: float = math.pi / 4.0
    c_gamma: float = 1.0
    c_delta: float = 0.25
    delta0: float = 0.375
    c_hybridization: float = 1.5
    impurity_zeeman: float = 0.0
    condensate_amplitude: float = 1.0
    fig1_left_scale: float = 0.10
    fig1_right_scale: float = 1.50
    phase_common_norm: bool = False


P = ModelParams()

COL_W = 3.375
FULL_W = 6.875
COLORS = {
    "blue": "#1f4e9c",
    "red": "#c2342c",
    "green": "#2c8a3d",
    "orange": "#e08214",
    "grey": "#555555",
    "black": "#1a1a1a",
}
LW_MAIN = 2.05
LW_SECONDARY = 1.70
LW_AUX = 1.15
LW_GUIDE = 0.75
LW_SPINE = 1.30
FIXED_TIME = datetime(2026, 7, 30, 12, 0, 0, tzinfo=timezone.utc)


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "STIXGeneral",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "font.size": 9.0,
            "axes.labelsize": 10.0,
            "axes.titlesize": 8.6,
            "xtick.labelsize": 8.4,
            "ytick.labelsize": 8.4,
            "legend.fontsize": 6.4,
            "axes.linewidth": 1.35,
            "lines.linewidth": LW_MAIN,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 4.6,
            "ytick.major.size": 4.6,
            "xtick.minor.size": 2.4,
            "ytick.minor.size": 2.4,
            "xtick.major.width": 1.25,
            "ytick.major.width": 1.25,
            "xtick.minor.width": 0.9,
            "ytick.minor.width": 0.9,
            "xtick.major.pad": 2.4,
            "ytick.major.pad": 2.4,
            "axes.labelpad": 2.6,
            "legend.frameon": False,
            "legend.handlelength": 1.45,
            "legend.handletextpad": 0.45,
            "legend.labelspacing": 0.23,
            "figure.dpi": 220,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.025,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def gauge_tbeta(beta0: float) -> float:
    """Return the gauge-invariant frozen control |beta0| r."""
    return abs(float(beta0)) * float(P.condensate_amplitude)


def channel_scales(beta0: float, flip: bool) -> tuple[float, float, float]:
    tbeta = gauge_tbeta(beta0)
    gamma_pt = P.c_gamma * tbeta
    coherent = P.delta0 + P.c_delta * tbeta if flip else 0.0
    hybridization = P.c_hybridization * tbeta
    return gamma_pt, coherent, hybridization


def build_kernel(beta0: float, k: float = 0.0, flip: bool = True) -> np.ndarray:
    """Return the gauge-fixed four-state frozen kernel."""
    gamma_pt, coherent, hybridization = channel_scales(beta0, flip)
    bath_plus = k * k + P.lambda_soc * k
    bath_minus = k * k - P.lambda_soc * k
    vertex = hybridization / math.sqrt(2.0)
    return np.array(
        [
            [
                P.eps_xi + P.impurity_zeeman + 1j * gamma_pt,
                coherent,
                vertex,
                vertex,
            ],
            [
                coherent,
                P.eps_xi - P.impurity_zeeman - 1j * gamma_pt,
                vertex,
                -vertex,
            ],
            [vertex, vertex, bath_plus, 0.0],
            [vertex, -vertex, 0.0, bath_minus],
        ],
        dtype=complex,
    )


def eigenvector_condition_number(
    matrix: np.ndarray,
) -> tuple[float, np.ndarray]:
    eigenvalues, right = np.linalg.eig(matrix)
    try:
        inverse = np.linalg.inv(right)
        condition = np.linalg.cond(right)
    except np.linalg.LinAlgError:
        inverse = np.linalg.pinv(right)
        condition = np.linalg.norm(right, 2) * np.linalg.norm(inverse, 2)
    return float(np.real(condition)), eigenvalues


def match_branches(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    cost = np.abs(current[None, :] - previous[:, None]) ** 2
    _, columns = linear_sum_assignment(cost)
    return current[columns]


def branch_sweep(beta_values: np.ndarray, flip: bool) -> dict[str, np.ndarray]:
    beta_values = np.asarray(beta_values, dtype=float)
    positive = beta_values[beta_values >= 0.0]
    negative = beta_values[beta_values < 0.0][::-1]

    def one_branch(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        spectra: list[np.ndarray] = []
        conditions: list[float] = []
        previous: np.ndarray | None = None
        for beta0 in values:
            condition, eigenvalues = eigenvector_condition_number(
                build_kernel(float(beta0), flip=flip)
            )
            if previous is None:
                eigenvalues = eigenvalues[np.argsort(np.real(eigenvalues))]
            else:
                eigenvalues = match_branches(eigenvalues, previous)
            previous = eigenvalues.copy()
            spectra.append(eigenvalues)
            conditions.append(condition)
        return np.asarray(spectra), np.asarray(conditions)

    positive_spectrum, positive_condition = one_branch(positive)
    negative_spectrum, negative_condition = one_branch(negative)
    spectrum = np.vstack([negative_spectrum[::-1], positive_spectrum])
    condition = np.concatenate(
        [negative_condition[::-1], positive_condition]
    )
    return {
        "betas": beta_values,
        "real": np.real(spectrum),
        "imag": np.imag(spectrum),
        "condition": condition,
    }


def exact_core_ep() -> float | None:
    denominator = P.c_gamma - P.c_delta
    if abs(P.impurity_zeeman) > 1.0e-14 or abs(denominator) < 1.0e-14:
        return None
    beta_ep = P.delta0 / denominator
    return float(beta_ep) if beta_ep > 0.0 else None


def plotted_ep(beta_values: np.ndarray, flip: bool) -> float | None:
    if not flip:
        return None
    exact = exact_core_ep()
    if exact is None:
        return None
    positive = beta_values >= 0.0
    index = np.argmin(np.abs(beta_values[positive] - exact))
    return float(beta_values[positive][index])


def choose_trajectory_betas(
    beta_values: np.ndarray, beta_ep: float | None, flip: bool
) -> list[float]:
    if not flip:
        return [0.0, 0.25, 0.50]
    reference = 0.50 if beta_ep is None else float(beta_ep)
    targets = [
        P.fig1_left_scale * reference,
        reference,
        P.fig1_right_scale * reference,
    ]
    return [
        float(beta_values[np.argmin(np.abs(beta_values - target))])
        for target in targets
    ]


def estimate_time_window(beta0: float, flip: bool) -> tuple[float, float]:
    eigenvalues = np.linalg.eigvals(build_kernel(beta0, flip=flip))
    ordered = sorted(eigenvalues, key=lambda value: abs(value.imag))
    real_splitting = abs(ordered[0].real - ordered[1].real)
    minimum_imaginary = min(abs(value.imag) for value in ordered)
    if real_splitting > 0.05:
        maximum = 2.5 * 2.0 * math.pi / real_splitting
    elif minimum_imaginary > 1.0e-4:
        maximum = 2.5 * 3.0 / max(minimum_imaginary, 1.0e-4)
    else:
        maximum = 20.0
    maximum = float(np.clip(maximum, 5.0, 35.0))
    return maximum, maximum / 350.0


def survival_amplitude(
    beta0: float,
    flip: bool,
    maximum_time: float | None = None,
    step: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if maximum_time is None or step is None:
        maximum_time, step = estimate_time_window(beta0, flip)
    times = np.arange(0.0, maximum_time, step)
    matrix = build_kernel(beta0, flip=flip)
    initial = np.zeros(4, dtype=complex)
    initial[0] = 1.0
    values = np.array(
        [
            initial.conj() @ (expm(-1j * matrix * time) @ initial)
            for time in times
        ]
    )
    return times, values


def gain_loss_bloch(
    beta0: float, times: np.ndarray, flip: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = build_kernel(beta0, flip=flip)
    initial = np.zeros(4, dtype=complex)
    initial[0] = 1.0
    z_values: list[float] = []
    y_values: list[float] = []
    norms: list[float] = []
    for time in times:
        state = expm(-1j * matrix * float(time)) @ initial
        up, down = state[0], state[1]
        n_up, n_down = abs(up) ** 2, abs(down) ** 2
        norm = max(float(n_up + n_down), 1.0e-30)
        z_values.append(float((n_up - n_down) / norm))
        y_values.append(float(2.0 * np.imag(np.conj(up) * down) / norm))
        norms.append(norm)
    return np.asarray(z_values), np.asarray(y_values), np.asarray(norms)


def interpolate_complex(
    times: np.ndarray, values: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    order = np.argsort(times)
    times = np.asarray(times, dtype=float)[order]
    values = np.asarray(values, dtype=complex)[order]
    return np.interp(grid, times, values.real) + 1j * np.interp(
        grid, times, values.imag
    )


def thicken(axis) -> None:
    for spine in axis.spines.values():
        spine.set_linewidth(LW_SPINE)
    axis.minorticks_on()


def tag(axis, text: str, x: float = 0.045, y: float = 0.93) -> None:
    axis.text(
        x,
        y,
        text,
        transform=axis.transAxes,
        fontsize=9.6,
        fontweight="bold",
        va="top",
        ha="left",
        zorder=30,
        bbox={"fc": "white", "ec": "none", "alpha": 0.82, "pad": 0.35},
    )


def annotate_ep(
    axis, beta_ep: float | None, beta_values: np.ndarray, label: bool
) -> None:
    if beta_ep is None or not np.isfinite(beta_ep):
        return
    lower, upper = float(np.min(beta_values)), float(np.max(beta_values))
    for location in sorted({float(beta_ep), -float(beta_ep)}):
        if lower <= location <= upper:
            axis.axvline(
                location,
                color=COLORS["black"],
                ls=(0, (4, 2)),
                lw=LW_AUX,
                alpha=0.80,
                zorder=2,
            )
            if label:
                axis.text(
                    location,
                    0.98,
                    "EP",
                    transform=axis.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=6.0,
                    bbox={"fc": "white", "ec": "none", "alpha": 0.65},
                )


def fade_plot(axis, x: np.ndarray, y: np.ndarray, color: str) -> None:
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    rgba = np.zeros((len(segments), 4))
    rgba[:, :3] = mpl.colors.to_rgb(color)
    rgba[:, 3] = np.linspace(0.12, 1.0, len(segments))
    axis.add_collection(
        LineCollection(segments, colors=rgba, linewidths=LW_MAIN, zorder=3)
    )
    axis.autoscale_view()


def add_arrows(
    axis, x: np.ndarray, y: np.ndarray, color: str, count: int = 3
) -> None:
    length = len(x)
    for index in np.linspace(
        int(0.12 * length), int(0.88 * length), count
    ).astype(int):
        if 0 < index < length - 1:
            axis.annotate(
                "",
                xy=(x[index + 1], y[index + 1]),
                xytext=(x[index - 1], y[index - 1]),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": color,
                    "lw": 0.1,
                    "mutation_scale": 10.0,
                    "alpha": 0.97,
                },
                zorder=6,
            )


def save_curve(
    out: Path, name: str, x: np.ndarray, y: np.ndarray, header: str
) -> None:
    np.savetxt(
        out / f"{name}.dat",
        np.column_stack([np.asarray(x, float), np.asarray(y, float)]),
        header=header,
    )


def save_figure(figure, out: Path, name: str) -> None:
    figure.savefig(
        out / f"{name}.pdf",
        metadata={
            "Title": f"Driven-Dirac impurity retained frozen-kernel {name}",
            "Author": "Driven-Dirac impurity reproducibility pipeline",
            "CreationDate": FIXED_TIME,
            "ModDate": FIXED_TIME,
        },
    )
    figure.savefig(
        out / f"{name}.png",
        metadata={"Software": "Driven-Dirac impurity retained frozen-kernel generator"},
    )
    plt.close(figure)
    print(f"  wrote {name}.pdf / {name}.png")


def make_main_figure(
    out: Path,
    flip: bool,
    name: str,
    labels: list[str],
    time_maximum: float | None,
    time_points: int,
) -> None:
    sweep = branch_sweep(P.beta_values, flip)
    betas = sweep["betas"]
    real = sweep["real"]
    imaginary = sweep["imag"]
    condition = sweep["condition"]
    beta_ep = plotted_ep(betas, flip)

    save_curve(out, f"{name}_kappa", betas, condition, "beta0 kappa_imp")
    for branch in range(real.shape[1]):
        save_curve(
            out,
            f"{name}_ReE{branch + 1}",
            betas,
            real[:, branch],
            f"beta0 ReE{branch + 1}",
        )
        save_curve(
            out,
            f"{name}_ImE{branch + 1}",
            betas,
            imaginary[:, branch],
            f"beta0 ImE{branch + 1}",
        )

    figure = plt.figure(figsize=(COL_W * 2.04, COL_W * 1.48))
    grid_spec = figure.add_gridspec(2, 2, hspace=0.32, wspace=0.34)
    eigen_grid = grid_spec[0, 0].subgridspec(2, 1, hspace=0.10)
    real_axis = figure.add_subplot(eigen_grid[0])
    imaginary_axis = figure.add_subplot(eigen_grid[1], sharex=real_axis)
    colors = [
        COLORS["blue"],
        COLORS["red"],
        COLORS["green"],
        COLORS["orange"],
    ]
    for branch in range(real.shape[1]):
        color = colors[branch % len(colors)]
        real_axis.plot(betas, real[:, branch], color=color, lw=LW_SECONDARY)
        imaginary_axis.plot(
            betas, imaginary[:, branch], color=color, lw=LW_SECONDARY
        )
    annotate_ep(real_axis, beta_ep, betas, False)
    annotate_ep(imaginary_axis, beta_ep, betas, True)
    imaginary_axis.axhline(
        0.0, color=COLORS["grey"], lw=LW_GUIDE, alpha=0.60
    )
    real_axis.set_ylabel(r"$\mathrm{Re}\,E$")
    imaginary_axis.set_ylabel(r"$\mathrm{Im}\,E$")
    imaginary_axis.set_xlabel(r"$\beta_0$")
    plt.setp(real_axis.get_xticklabels(), visible=False)
    thicken(real_axis)
    thicken(imaginary_axis)
    tag(real_axis, "(a)")

    condition_axis = figure.add_subplot(grid_spec[0, 1])
    condition_plot = np.clip(condition, 1.0, None)
    if np.max(condition_plot) < 50.0:
        condition_axis.plot(
            betas, condition_plot, color=COLORS["grey"], lw=LW_MAIN
        )
        condition_axis.set_ylim(0.95, 1.10 * float(np.max(condition_plot)))
        condition_axis.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        condition_axis.yaxis.set_major_formatter(
            ticker.StrMethodFormatter("{x:g}")
        )
    else:
        condition_axis.semilogy(
            betas, condition_plot, color=COLORS["grey"], lw=LW_MAIN
        )
    annotate_ep(condition_axis, beta_ep, betas, True)
    if not flip:
        condition_axis.set_title(
            "gain--loss control: no coherent local EP", pad=3
        )
    condition_axis.set_xlabel(r"$\beta_0$")
    condition_axis.set_ylabel(r"$\kappa_{\mathrm{imp}}$")
    thicken(condition_axis)
    tag(condition_axis, "(b)")

    time_axis = figure.add_subplot(grid_spec[1, 0])
    phase_axis = figure.add_subplot(grid_spec[1, 1])
    trajectory_betas = choose_trajectory_betas(betas, beta_ep, flip)
    trajectory_colors = [
        COLORS["blue"],
        COLORS["orange"],
        COLORS["green"],
    ]
    raw: list[tuple[float, np.ndarray, np.ndarray]] = []
    for beta0 in trajectory_betas:
        requested_maximum = (
            float(time_maximum)
            if time_maximum is not None
            else (24.0 if flip else None)
        )
        requested_step = (
            requested_maximum / 420.0
            if requested_maximum is not None
            else None
        )
        times, values = survival_amplitude(
            beta0, flip, requested_maximum, requested_step
        )
        raw.append((beta0, times, values))

    start = max(float(np.min(times)) for _, times, _ in raw)
    stop_solver = min(float(np.max(times)) for _, times, _ in raw)
    stop = (
        stop_solver
        if time_maximum is None
        else min(float(time_maximum), stop_solver)
    )
    common_times = np.linspace(start, stop, max(int(time_points), 64))
    trajectories = [
        (
            beta0,
            interpolate_complex(times, values, common_times),
        )
        for beta0, times, values in raw
    ]
    common_norm = (
        max(float(np.max(np.abs(values))) for _, values in trajectories)
        if P.phase_common_norm
        else None
    )

    with (out / f"{name}_phase_beta_choices.txt").open(
        "w", encoding="utf-8"
    ) as handle:
        handle.write(
            f"beta_EP = {beta_ep:.12g}\n"
            if beta_ep is not None
            else "beta_EP = none  # gain/loss-only control\n"
        )
        handle.write(f"phase_common_norm = {P.phase_common_norm}\n")
        for index, (beta0, values) in enumerate(trajectories, start=1):
            handle.write(
                f"curve_{index}: beta0 = {beta0:.12g}, "
                f"max_abs_O = {np.max(np.abs(values)):.12g}\n"
            )

    for index, ((beta0, values), color, label) in enumerate(
        zip(trajectories, trajectory_colors, labels), start=1
    ):
        real_values = values.real
        imaginary_values = values.imag
        real_scale = np.max(np.abs(real_values)) or 1.0
        time_axis.plot(
            common_times,
            real_values / real_scale,
            color=color,
            lw=LW_MAIN,
            label=label,
        )
        save_curve(
            out,
            f"{name}_dyn_{index}_beta_{beta0:.4g}",
            common_times,
            real_values / real_scale,
            "t_common normalized_Re_survival_amplitude",
        )
        phase_norm = common_norm or np.max(np.abs(values)) or 1.0
        x, y = real_values / phase_norm, imaginary_values / phase_norm
        if flip:
            fade_plot(phase_axis, x, y, color)
            add_arrows(phase_axis, x, y, color)
        else:
            phase_axis.plot(
                x, y, color=color, lw=LW_MAIN, alpha=0.92, label=label
            )
        phase_axis.plot(
            [x[0]],
            [y[0]],
            "o",
            color=color,
            ms=4.5,
            mec=COLORS["black"],
            mew=0.6,
            zorder=7,
        )
        save_curve(
            out,
            f"{name}_survival_phase_{index}_beta_{beta0:.4g}",
            x,
            y,
            "Re_survival_amplitude_normalized "
            "Im_survival_amplitude_normalized",
        )

    time_axis.set_xlim(common_times[0], common_times[-1])
    time_axis.set_xlabel(r"$t$")
    time_axis.set_ylabel(r"$\mathrm{Re}\,O(t)/\max$")
    time_axis.legend(
        loc="lower left",
        fontsize=5.7,
        frameon=True,
        framealpha=0.88,
        facecolor="white",
        edgecolor="none",
    )
    thicken(time_axis)
    tag(time_axis, "(c)")

    if flip:
        proxies = [
            Line2D(
                [0], [0], color=trajectory_colors[index], lw=LW_MAIN
            )
            for index in range(3)
        ]
        phase_axis.legend(
            proxies,
            labels,
            loc="upper right",
            fontsize=5.5,
            frameon=True,
            framealpha=0.88,
            facecolor="white",
            edgecolor="none",
        )
        note = r"small $\beta_0$ $\rightarrow$ EP $\rightarrow$ broken"
    else:
        phase_axis.legend(
            loc="upper right",
            fontsize=5.5,
            frameon=True,
            framealpha=0.88,
            facecolor="white",
            edgecolor="none",
        )
        note = "survival-amplitude portrait\nno coherent local EP marker"
    phase_axis.text(
        0.04,
        0.05,
        note,
        transform=phase_axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.4,
        bbox={
            "boxstyle": "round,pad=0.24",
            "fc": "white",
            "ec": COLORS["grey"],
            "lw": 0.6,
            "alpha": 0.84,
        },
    )
    phase_axis.set_xlabel(r"$\mathrm{Re}\,O(t)/\max|O|$")
    phase_axis.set_ylabel(r"$\mathrm{Im}\,O(t)/\max|O|$")
    phase_axis.set_aspect("equal", adjustable="datalim")
    thicken(phase_axis)
    tag(phase_axis, "(d)")
    save_figure(figure, out, name)


def make_bloch_supplement(out: Path, time_points: int) -> None:
    betas = np.asarray(P.beta_values, dtype=float)
    beta_ep = plotted_ep(betas, True)
    trajectory_betas = choose_trajectory_betas(betas, beta_ep, True)
    labels = [r"small $\beta_0$", r"near EP", r"above EP"]
    colors = [COLORS["blue"], COLORS["orange"], COLORS["green"]]
    maximum_time = 15.0
    times = np.linspace(0.0, maximum_time, max(int(time_points), 64))

    figure = plt.figure(figsize=(FULL_W, 2.45))
    grid_spec = figure.add_gridspec(1, 2, wspace=0.34)
    bloch_axis = figure.add_subplot(grid_spec[0, 0])
    norm_axis = figure.add_subplot(grid_spec[0, 1])
    for index, (beta0, color, label) in enumerate(
        zip(trajectory_betas, colors, labels), start=1
    ):
        z, y, norm = gain_loss_bloch(beta0, times)
        fade_plot(bloch_axis, z, y, color)
        add_arrows(bloch_axis, z, y, color)
        bloch_axis.plot(
            [z[0]],
            [y[0]],
            "o",
            color=color,
            ms=4.2,
            mec=COLORS["black"],
            mew=0.55,
            zorder=7,
        )
        norm_axis.plot(
            times, norm / norm[0], color=color, lw=LW_MAIN, label=label
        )
        save_curve(
            out,
            f"SFig1_bloch_{index}_beta_{beta0:.4g}",
            z,
            y,
            "z=(n_up-n_down)/(n_up+n_down) "
            "y=2Im(up*_down)/(n_up+n_down)",
        )
        save_curve(
            out,
            f"SFig1_Nimp_{index}_beta_{beta0:.4g}",
            times,
            norm / norm[0],
            "t_common N_imp(t)/N_imp(0)",
        )

    bloch_axis.set_xlabel(r"$z=(n_\uparrow-n_\downarrow)/N_{\rm imp}$")
    bloch_axis.set_ylabel(
        r"$y=2\,\mathrm{Im}(\psi_\uparrow^*\psi_\downarrow)/N_{\rm imp}$"
    )
    bloch_axis.set_aspect("equal", adjustable="datalim")
    bloch_axis.legend(
        [Line2D([0], [0], color=color, lw=LW_MAIN) for color in colors],
        labels,
        loc="upper right",
        fontsize=5.7,
        frameon=True,
        framealpha=0.88,
        facecolor="white",
        edgecolor="none",
    )
    thicken(bloch_axis)
    tag(bloch_axis, "(a)")
    bloch_axis.set_title("Gain--loss Bloch diagnostic", pad=3)

    norm_axis.axhline(1.0, color=COLORS["grey"], lw=LW_GUIDE, ls=":")
    norm_axis.set_xlabel(r"$t$")
    norm_axis.set_ylabel(r"$N_{\rm imp}(t)/N_{\rm imp}(0)$")
    norm_axis.set_yscale("log")
    norm_axis.set_xlim(0.0, maximum_time)
    norm_axis.set_xticks([0, 3, 6, 9, 12, 15])
    norm_axis.legend(
        loc="best",
        fontsize=5.7,
        frameon=True,
        framealpha=0.88,
        facecolor="white",
        edgecolor="none",
    )
    thicken(norm_axis)
    tag(norm_axis, "(b)")
    norm_axis.set_title("Impurity-sector norm check", pad=3)
    save_figure(figure, out, "SFig1_bloch_diagnostic")


def write_captions(out: Path) -> None:
    text = r"""
% Captions for the retained frozen-kernel Figs. 1, 2, and S1.

\caption{Full coherent impurity kernel. (a) Real and imaginary parts of the four eigenvalues of the projected $4\times4$ non-Hermitian kernel versus $\beta_0$. (b) Impurity eigenvector condition number $\kappa_{\rm imp}$. (c) Representative survival-amplitude dynamics below, near, and above the EP. (d) The corresponding phase portraits.}

\caption{Gain--loss-only control. (a) Four eigenvalue branches with the coherent spin-flip channel removed. (b) $\kappa_{\rm imp}$ shows no coherent local EP marker. (c) Dynamics and (d) survival-amplitude portrait in the same convention as Fig.~1.}
""".strip()
    (out / "captions.tex").write_text(text + "\n", encoding="utf-8")


def run(
    out: Path,
    which: str,
    time_maximum: float | None,
    time_points: int,
) -> None:
    set_style()
    out.mkdir(parents=True, exist_ok=True)
    print(f"Writing figures to: {out}")
    print(
        f"Convention: eps_xi=-U/2={-P.U / 2:.1f}, U={P.U}, "
        f"exact local EP beta0={exact_core_ep()}"
    )
    if which in ("all", "fig1"):
        make_main_figure(
            out,
            True,
            "Fig1",
            ["small beta", "near EP", "above EP"],
            time_maximum,
            time_points,
        )
        make_bloch_supplement(out, time_points)
    if which in ("all", "fig2"):
        make_main_figure(
            out,
            False,
            "Fig2",
            ["Hermitian ref", "weak gain/loss", "strong gain/loss"],
            time_maximum,
            time_points,
        )
    write_captions(out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate retained frozen-kernel PRB Figs. 1, 2, and S1."
    )
    parser.add_argument("--out", type=Path, default=Path("frozen_kernel_output"))
    parser.add_argument(
        "--which", choices=("all", "fig1", "fig2"), default="all"
    )
    parser.add_argument("--time-tmax", type=float, default=None)
    parser.add_argument("--time-points", type=int, default=900)
    parser.add_argument(
        "--fig1-left-scale", type=float, default=P.fig1_left_scale
    )
    parser.add_argument(
        "--fig1-right-scale", type=float, default=P.fig1_right_scale
    )
    parser.add_argument("--phase-common-norm", action="store_true")
    parser.add_argument("--lambda-k", type=float, default=None)
    args = parser.parse_args()
    if args.lambda_k is not None:
        P.lambda_soc = float(args.lambda_k)
    P.fig1_left_scale = float(args.fig1_left_scale)
    P.fig1_right_scale = float(args.fig1_right_scale)
    P.phase_common_norm = bool(args.phase_common_norm)
    run(args.out, args.which, args.time_tmax, args.time_points)


if __name__ == "__main__":
    main()
