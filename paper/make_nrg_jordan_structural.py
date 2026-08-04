#!/usr/bin/env python3
"""Build the scoped finite-U NRG Jordan-residue figure.

The figure deliberately reports only the matrix-residue diagnostics that pass
the numerical audit.  It does not plot the unstable pole-splitting exponent or
the rejected quartic plateau.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


AUDIT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANALYSIS = (
    AUDIT_ROOT
    / "adapter"
    / "PTDirac_NHNRG_adapter"
    / "output"
    / "jordan_comprehensive"
    / "analysis"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "output"


COLORS = {
    "blue": "#2C6FBB",
    "orange": "#E67E22",
    "green": "#2E9B4E",
    "red": "#C33C54",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.linewidth": 0.75,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 6.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.minor.width": 0.55,
            "ytick.minor.width": 0.55,
            "savefig.transparent": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def range_summary(frame: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    return (
        frame.groupby(x, as_index=False)[y]
        .agg(median="median", lower="min", upper="max")
        .sort_values(x)
    )


def quantile_summary(frame: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for value, group in frame.groupby(x):
        data = group[y].to_numpy(float)
        data = data[np.isfinite(data) & (data > 0)]
        if data.size == 0:
            continue
        rows.append(
            {
                x: float(value),
                "median": float(np.median(data)),
                "lower": float(np.quantile(data, 0.16)),
                "upper": float(np.quantile(data, 0.84)),
            }
        )
    return pd.DataFrame(rows).sort_values(x)


def extrapolated_alignment_error(row: pd.Series) -> float:
    """Frobenius-angle error of the extrapolated residue and EP generator."""
    b0 = complex(row["B0_0_real"], row["B0_0_imag"])
    bx = complex(row["Bx0_real"], row["Bx0_imag"])
    by = complex(row["By0_real"], row["By0_imag"])
    bz = complex(row["Bz0_real"], row["Bz0_imag"])
    residue = np.array(
        [[b0 + bz, bx - 1j * by], [bx + 1j * by, b0 - bz]],
        dtype=np.complex128,
    )
    # Reference scan: Delta_eff = delta0 + c_delta*beta0 with
    # delta0=0.075 and c_delta=0.050; Gamma_PT is exported explicitly.
    delta_eff = 0.075 + 0.050 * float(row["beta0"])
    gamma_pt = float(row["Gamma_PT"])
    generator = np.array(
        [[delta_eff, 1j * gamma_pt], [1j * gamma_pt, -delta_eff]],
        dtype=np.complex128,
    )
    denominator = float(np.linalg.norm(generator) * np.linalg.norm(residue))
    if denominator == 0.0:
        return np.nan
    alignment = float(abs(np.vdot(generator, residue)) / denominator)
    return max(1.0 - alignment, 1e-16)


def draw_band(
    ax: plt.Axes,
    summary: pd.DataFrame,
    *,
    x: str,
    color: str,
    marker: str,
    label: str,
    linestyle: str = "-",
    band_alpha: float = 0.12,
) -> None:
    ax.fill_between(
        summary[x].to_numpy(float),
        summary["lower"].to_numpy(float),
        summary["upper"].to_numpy(float),
        color=color,
        alpha=band_alpha,
        linewidth=0,
        zorder=1,
    )
    ax.plot(
        summary[x],
        summary["median"],
        color=color,
        marker=marker,
        markersize=3.5,
        markeredgewidth=0.45,
        linewidth=1.35,
        linestyle=linestyle,
        label=label,
        zorder=2,
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.025,
        0.965,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        fontweight="bold",
    )


def build_figure(analysis_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    ext = pd.read_csv(analysis_dir / "zero_detuning_ensemble.csv")
    raw = pd.read_csv(analysis_dir / "tracked_pairs_all_iterations.csv")

    ext = ext[
        (ext["config_label"] == "reference")
        & (ext["model"] == "scalar")
        & (ext["method"] == "linear_all")
        & (ext["iteration"].isin([4, 5, 6]))
        & (ext["sector"].isin([-1, 1]))
        & (ext["U"] <= 0.1000001)
    ].copy()
    if ext.empty:
        raise RuntimeError("No zero-detuning reference rows survived the figure filter")

    ext["alignment_error0"] = ext.apply(extrapolated_alignment_error, axis=1)
    ext["B0_abs"] = np.hypot(ext["B0_0_real"], ext["B0_0_imag"])
    ext["By_abs"] = np.hypot(ext["By0_real"], ext["By0_imag"])

    reference_raw = raw[
        (raw["config_label"] == "reference")
        & (raw["model"] == "scalar")
        & (raw["iteration"].isin([4, 5, 6]))
        & (raw["sector"].isin([-1, 1]))
        & (raw["U"] <= 0.1000001)
    ].copy()
    reliable = reference_raw["reliable"].astype(str).str.lower().isin(["true", "1"])
    reliable_fraction = float(reliable.mean())
    raw = reference_raw[reliable].copy()
    if raw.empty:
        raise RuntimeError("No reliable branch-tracked rows survived the figure filter")
    raw["alignment_error"] = np.maximum(1.0 - raw["alignment"], 1e-16)

    configure_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.72), constrained_layout=True)

    # (a) Extrapolated Jordan purity at finite U.
    ax = axes[0]
    diagnostics = [
        ("alignment_error0", COLORS["blue"], "o", r"$1-A_J$", "-"),
        ("trace_component_fraction0", COLORS["orange"], "s", r"$f_{\rm tr}$", "--"),
        ("nilpotent_mismatch0", COLORS["green"], "^", r"$\epsilon_N$", "-."),
    ]
    for column, color, marker, label, linestyle in diagnostics:
        draw_band(
            ax,
            range_summary(ext, "U", column),
            x="U",
            color=color,
            marker=marker,
            label=label,
            linestyle=linestyle,
        )
    ax.set_yscale("log")
    ax.set_xlim(-0.003, 0.103)
    ax.set_ylim(1e-13, 8e-3)
    ax.set_xlabel(r"interaction $U$")
    ax.set_ylabel("Jordan-purity error")
    ax.grid(True, which="major", color="#B5B5B5", alpha=0.28, linewidth=0.45)
    ax.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.89),
        columnspacing=0.8,
        handlelength=1.8,
    )
    ax.text(
        0.97,
        0.055,
        r"range: $n=4$-$6$, $q=\pm1$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
    )
    panel_label(ax, "(a)")

    # (b) Pauli decomposition of the extrapolated residue matrix.
    ax = axes[1]
    components = [
        ("Bx0_abs", COLORS["blue"], "o", r"$|B_x|$", "-"),
        ("Bz0_abs", COLORS["orange"], "s", r"$|B_z|$", "--"),
        ("B0_abs", COLORS["green"], "^", r"$|B_0|$", "-."),
        ("By_abs", COLORS["red"], "d", r"$|B_y|$", ":"),
    ]
    for column, color, marker, label, linestyle in components:
        draw_band(
            ax,
            range_summary(ext, "U", column),
            x="U",
            color=color,
            marker=marker,
            label=label,
            linestyle=linestyle,
            band_alpha=0.08,
        )
    ax.set_yscale("log")
    ax.set_xlim(-0.003, 0.103)
    ax.set_ylim(4e-14, 3e-3)
    ax.set_xlabel(r"interaction $U$")
    ax.set_ylabel("extrapolated residue component")
    ax.grid(True, which="major", color="#B5B5B5", alpha=0.28, linewidth=0.45)
    ax.legend(frameon=False, ncol=2, loc="lower right", columnspacing=0.9, handlelength=1.8)
    ax.text(
        0.04,
        0.11,
        r"$|B_x|\simeq |B_z|\gg |B_0|,|B_y|$",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.7,
    )
    panel_label(ax, "(b)")

    # (c) Direct finite-detuning approach to the Jordan matrix direction.
    ax = axes[2]
    raw_diagnostics = [
        ("alignment_error", COLORS["blue"], "o", r"$1-A_J$", "-"),
        ("trace_fraction", COLORS["orange"], "s", r"$f_{\rm tr}$", "--"),
        ("nilpotent_mismatch", COLORS["green"], "^", r"$\epsilon_N$", "-."),
    ]
    for column, color, marker, label, linestyle in raw_diagnostics:
        draw_band(
            ax,
            quantile_summary(raw, "delta_coh", column),
            x="delta_coh",
            color=color,
            marker=marker,
            label=label,
            linestyle=linestyle,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(8e-6, 1.25e-4)
    ax.set_ylim(1e-9, 3e-2)
    ax.set_xlabel(r"coherent detuning $\delta_{\rm coh}$")
    ax.set_ylabel("finite-detuning error")
    ax.grid(True, which="major", color="#B5B5B5", alpha=0.28, linewidth=0.45)
    ax.legend(frameon=False, ncol=1, loc="upper left", bbox_to_anchor=(0.02, 0.88))
    ax.annotate(
        r"Jordan limit",
        xy=(1.05e-5, 2.6e-8),
        xytext=(2.2e-5, 4.5e-9),
        arrowprops={"arrowstyle": "->", "lw": 0.65, "color": "#333333"},
        fontsize=6.6,
        ha="left",
    )
    ax.text(
        0.97,
        0.055,
        rf"reliable pairs: ${reliable_fraction:.3f}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
    )
    panel_label(ax, "(c)")

    output_pdf = output_dir / "pdf" / "Fig_NRG_Jordan_structural_clean.pdf"
    output_png = output_dir / "png" / "Fig_NRG_Jordan_structural_clean.png"
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": "Finite-U NRG Jordan-residue diagnostics",
        "Subject": "Scoped structural NRG check for the pseudo-Hermitian impurity model",
        "Creator": "make_nrg_jordan_structural.py",
    }
    fig.savefig(output_pdf, bbox_inches="tight", pad_inches=0.025, metadata=metadata)
    fig.savefig(output_png, dpi=360, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    return output_pdf, output_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdf, png = build_figure(args.analysis_dir.resolve(), args.output_dir.resolve())
    print(pdf)
    print(png)


if __name__ == "__main__":
    main()
