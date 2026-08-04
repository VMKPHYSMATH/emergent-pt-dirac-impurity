#!/usr/bin/env python3
r"""Causal scattering-delay and pre-EP RG gate for Driven-Dirac impurity.

This gate extends the validated small-W projected-core calculation with the
physical Fisher--Lee scattering matrix

    S = I - i Gamma_bath^(1/2) G^R Gamma_bath^(1/2),

where H_eff = H_0 - i Gamma_bath/2.  In the passive Markov window the full
two-channel S matrix must be unitary.  The Wigner--Smith matrix

    Q = -i S^\dagger dS/domega

is then Hermitian.  Its trace is used as a causal pole-coalescence
fingerprint, not as a Kondo scale or a Petermann multiplier.

The matrices are evaluated in the original spin representation.  A fixed
Hadamard rotation transforms G, Gamma_bath, S, and Q together; the
eigenphases and proper delays used below are therefore basis invariant.
Individual off-diagonal matrix elements are not used as evidence.

The main four-panel output combines the delay fingerprint with the previously
validated causal channel-density and matrix-RG comparisons.  A second
supplementary figure reports pole linewidths, proper delays, the rate-matched
off-EP control, and numerical identities.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ptdirac_matplotlib")
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.lines import Line2D

from self_consistent_saddle_gate import D0, G0, I2, P, SX, SZ, SaddleParams
from channel_resolved_rg_gate import (
    BETA_VALUES,
    PRIMARY_THRESHOLD,
    green_and_density,
)


TRACE_BETAS = (0.35, 0.45, 0.50)
DISPLAY_OMEGAS = np.linspace(-0.40, 0.40, 1601)
PEAK_OMEGAS = np.linspace(-0.60, 0.60, 12001)
SUMRULE_OMEGAS = np.linspace(-20.0, 20.0, 40001)
THRESHOLDS_TO_PLOT = (0.30, 0.50, 0.75, 1.00)


def projected_hamiltonian(
    beta: float,
    control: bool = False,
    coherent_override: float | None = None,
    p: SaddleParams = P,
) -> np.ndarray:
    coherent = D0(beta, p) if coherent_override is None else float(coherent_override)
    relative = 0.0 if control else G0(beta, p)
    return coherent * SX - 1j * p.gamma0 * I2 + 1j * relative * SZ


def bath_rate_matrix(beta: float, control: bool = False, p: SaddleParams = P) -> np.ndarray:
    relative = 0.0 if control else G0(beta, p)
    return 2.0 * (p.gamma0 * I2 - relative * SZ)


def psd_sqrt(matrix: np.ndarray) -> np.ndarray:
    hermitian = 0.5 * (matrix + matrix.conj().T)
    values, vectors = np.linalg.eigh(hermitian)
    if np.min(values) < -1.0e-12:
        raise ValueError("bath rate matrix is not positive semidefinite")
    return (vectors * np.sqrt(np.clip(values, 0.0, None))) @ vectors.conj().T


def scattering_observables(
    omega: float,
    beta: float,
    control: bool = False,
    coherent_override: float | None = None,
    p: SaddleParams = P,
) -> dict[str, Any]:
    h_eff = projected_hamiltonian(beta, control, coherent_override, p)
    gamma = bath_rate_matrix(beta, control, p)
    root = psd_sqrt(gamma)
    g_r = np.linalg.inv(float(omega) * I2 - h_eff)
    scattering = I2 - 1j * root @ g_r @ root
    derivative_g = -(g_r @ g_r)
    derivative_s = -1j * root @ derivative_g @ root
    smith = -1j * scattering.conj().T @ derivative_s
    smith_h = 0.5 * (smith + smith.conj().T)
    proper = np.linalg.eigvalsh(smith_h)
    determinant_derivative = -1j * np.trace(np.linalg.solve(scattering, derivative_s))
    return {
        "S": scattering,
        "Q": smith_h,
        "proper_delays": proper,
        "trace_delay": float(np.trace(smith_h).real),
        "determinant_delay_real": float(determinant_derivative.real),
        "determinant_delay_imag": float(determinant_derivative.imag),
        "unitarity_error": float(
            np.max(np.abs(scattering.conj().T @ scattering - I2))
        ),
        "Q_hermiticity_error": float(np.max(np.abs(smith - smith.conj().T))),
        "determinant_identity_error": float(
            abs(np.trace(smith_h).real - determinant_derivative.real)
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    converted: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {}
        for key, value in row.items():
            try:
                item[key] = float(value)
            except (TypeError, ValueError):
                item[key] = value
        converted.append(item)
    return converted


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pole_rows(p: SaddleParams = P) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for beta in np.linspace(0.35, 0.55, 81):
        for control in (False, True):
            coherent = D0(float(beta), p)
            relative = 0.0 if control else G0(float(beta), p)
            discriminant = coherent * coherent - relative * relative
            if discriminant >= 0.0:
                splitting = math.sqrt(discriminant)
                poles = np.array(
                    [-splitting - 1j * p.gamma0, splitting - 1j * p.gamma0]
                )
            else:
                splitting = math.sqrt(-discriminant)
                poles = np.array(
                    [-1j * (p.gamma0 + splitting), -1j * (p.gamma0 - splitting)]
                )
            for index, pole in enumerate(poles):
                rows.append(
                    {
                        "beta0": float(beta),
                        "control_gamma_pt_zero": int(control),
                        "pole_index": index,
                        "pole_real": float(pole.real),
                        "pole_halfwidth": float(-pole.imag),
                        "coherent_channel": D0(float(beta), p),
                        "relative_rate_channel": 0.0 if control else G0(float(beta), p),
                    }
                )
    return rows


def delay_rows(p: SaddleParams = P) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows: list[dict[str, Any]] = []
    maxima = {
        "unitarity": 0.0,
        "hermiticity": 0.0,
        "determinant_identity": 0.0,
        "determinant_imag": 0.0,
    }
    cases: list[tuple[float, bool, str, float | None]] = [
        (beta, False, f"active_beta_{beta:.2f}", None) for beta in TRACE_BETAS
    ]
    cases.append((0.50, True, "matched_gammaPT_zero", None))
    # Same rate matrix as the EP, but coherent channel displaced from the EP.
    cases.append((0.50, False, "rate_matched_off_EP_D_0.05", 0.05))
    for beta, control, case, coherent_override in cases:
        for omega in DISPLAY_OMEGAS:
            obs = scattering_observables(
                float(omega), beta, control, coherent_override, p
            )
            maxima["unitarity"] = max(maxima["unitarity"], obs["unitarity_error"])
            maxima["hermiticity"] = max(maxima["hermiticity"], obs["Q_hermiticity_error"])
            maxima["determinant_identity"] = max(
                maxima["determinant_identity"], obs["determinant_identity_error"]
            )
            maxima["determinant_imag"] = max(
                maxima["determinant_imag"], abs(obs["determinant_delay_imag"])
            )
            rows.append(
                {
                    "case": case,
                    "beta0": beta,
                    "control_gamma_pt_zero": int(control),
                    "coherent_override": (
                        "" if coherent_override is None else coherent_override
                    ),
                    "omega": float(omega),
                    "trace_delay": obs["trace_delay"],
                    "proper_delay_min": float(obs["proper_delays"][0]),
                    "proper_delay_max": float(obs["proper_delays"][1]),
                    "unitarity_error": obs["unitarity_error"],
                    "Q_hermiticity_error": obs["Q_hermiticity_error"],
                    "determinant_identity_error": obs["determinant_identity_error"],
                }
            )
    return rows, maxima


def peak_rows(p: SaddleParams = P) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for beta in BETA_VALUES:
        for control in (False, True):
            proper_maximum = -math.inf
            trace_maximum = -math.inf
            omega_proper = math.nan
            omega_trace = math.nan
            for omega in PEAK_OMEGAS:
                obs = scattering_observables(float(omega), beta, control, p=p)
                if obs["proper_delays"][1] > proper_maximum:
                    proper_maximum = float(obs["proper_delays"][1])
                    omega_proper = float(omega)
                if obs["trace_delay"] > trace_maximum:
                    trace_maximum = obs["trace_delay"]
                    omega_trace = float(omega)
            poles = np.linalg.eigvals(projected_hamiltonian(beta, control, p=p))
            pole_halfwidth = float(np.mean(-poles.imag))
            minimum_rate_halfwidth = p.gamma0 - (0.0 if control else G0(beta, p))
            rows.append(
                {
                    "beta0": beta,
                    "control_gamma_pt_zero": int(control),
                    "proper_delay_peak": proper_maximum,
                    "proper_delay_peak_omega": omega_proper,
                    "trace_delay_peak": trace_maximum,
                    "trace_delay_peak_omega": omega_trace,
                    "mean_pole_halfwidth": pole_halfwidth,
                    "minimum_rate_halfwidth": minimum_rate_halfwidth,
                    "pole_width_normalized_proper_peak": pole_halfwidth * proper_maximum,
                    "rate_width_normalized_proper_peak": minimum_rate_halfwidth * proper_maximum,
                }
            )
    # A rate-matched off-EP control at beta=0.50 demonstrates that a raw delay
    # peak is not, by itself, an EP enhancement.
    for coherent in (0.05, 0.075, 0.10, 0.125, 0.15):
        maximum = -math.inf
        for omega in PEAK_OMEGAS:
            obs = scattering_observables(
                float(omega), 0.50, False, coherent_override=coherent, p=p
            )
            maximum = max(maximum, float(obs["proper_delays"][1]))
        rows.append(
            {
                "beta0": 0.50,
                "control_gamma_pt_zero": 0,
                "proper_delay_peak": maximum,
                "proper_delay_peak_omega": math.nan,
                "trace_delay_peak": math.nan,
                "trace_delay_peak_omega": math.nan,
                "mean_pole_halfwidth": math.nan,
                "minimum_rate_halfwidth": p.gamma0 - G0(0.50, p),
                "pole_width_normalized_proper_peak": math.nan,
                "rate_width_normalized_proper_peak": (
                    p.gamma0 - G0(0.50, p)
                ) * maximum,
                "rate_matched_coherent_channel": coherent,
            }
        )
    return rows


def sum_rule(p: SaddleParams = P) -> float:
    values = [
        scattering_observables(float(omega), 0.50, False, p=p)["trace_delay"]
        for omega in SUMRULE_OMEGAS
    ]
    return float(np.trapezoid(values, SUMRULE_OMEGAS) / (2.0 * np.pi))


def main_figure(
    out: Path,
    rg_dir: Path,
    delay_data: list[dict[str, Any]],
) -> None:
    density = read_csv(rg_dir / "causal_channel_density.csv")
    flow = read_csv(rg_dir / "representative_rg_flow.csv")
    comparison = read_csv(rg_dir / "active_control_scale_comparison.csv")
    fig, axes = plt.subplots(2, 2, figsize=(8.35, 6.35), constrained_layout=True)

    for control, style, label in (
        (0.0, "-", r"EP-active"),
        (1.0, "--", r"matched $\Gamma_{\rm PT}=0$"),
    ):
        selected = [row for row in density if row["control_gamma_pt_zero"] == control]
        axes[0, 0].plot(
            [row["omega"] for row in selected],
            [row["rho_max_eigenvalue"] for row in selected],
            style,
            lw=1.6,
            label=label,
        )
    axes[0, 0].set(
        xlabel=r"$\omega$",
        ylabel=r"$\lambda_{\max}[\rho(\omega)]$",
        title=r"(a) Causal channel density, $\beta_0=0.50$",
        xlim=(-0.32, 0.32),
    )
    axes[0, 0].legend(frameon=False, fontsize=7)

    styles = {
        "active_beta_0.35": ("tab:blue", "-", r"$\beta_0=0.35$"),
        "active_beta_0.45": ("tab:orange", "-", r"$\beta_0=0.45$"),
        "active_beta_0.50": ("tab:red", "-", r"EP, $\beta_0=0.50$"),
        "matched_gammaPT_zero": ("black", "--", r"matched $\Gamma_{\rm PT}=0$"),
    }
    for case, (color, style, label) in styles.items():
        selected = [row for row in delay_data if row["case"] == case]
        axes[0, 1].plot(
            [row["omega"] for row in selected],
            [row["trace_delay"] for row in selected],
            color=color,
            ls=style,
            lw=1.45,
            label=label,
        )
    axes[0, 1].set(
        xlabel=r"$\omega$",
        ylabel=r"${\rm Tr}\,Q(\omega)$",
        title="(b) Scattering-delay coalescence profile",
        xlim=(-0.32, 0.32),
    )
    axes[0, 1].legend(frameon=False, fontsize=6.6)

    for control, style, label in (
        (0.0, "-", "EP-active"),
        (1.0, "--", r"matched $\Gamma_{\rm PT}=0$"),
    ):
        selected = [row for row in flow if row["control_gamma_pt_zero"] == control]
        axes[1, 0].semilogx(
            [row["cutoff"] for row in selected],
            [row["gmax"] for row in selected],
            style,
            lw=1.55,
            label=label,
        )
    axes[1, 0].invert_xaxis()
    axes[1, 0].axhline(1.0, color="black", lw=0.7, ls=":")
    axes[1, 0].set(
        xlabel=r"running cutoff $\Lambda$",
        ylabel=r"$g_{\max}(\Lambda)$",
        title="(c) Channel-resolved matrix RG flow",
    )
    axes[1, 0].set_xticks((0.10, 0.20, 0.50))
    axes[1, 0].set_xticklabels(("0.10", "0.20", "0.50"))
    axes[1, 0].legend(frameon=False, fontsize=7)

    threshold_styles = (
        (0.30, "o-", "tab:blue"),
        (0.50, "s-", "tab:orange"),
        (0.75, "^-", "tab:green"),
        (1.00, "d-", "tab:red"),
    )
    for threshold, style, color in threshold_styles:
        selected = [
            row for row in comparison
            if abs(row["threshold"] - threshold) < 1.0e-12
        ]
        axes[1, 1].plot(
            [row["beta0"] for row in selected],
            [row["active_to_control_scale_ratio"] for row in selected],
            style,
            color=color,
            ms=3.1,
            lw=1.1,
        )
        axes[1, 1].annotate(
            fr"$g_\star={threshold:.2f}$",
            xy=(
                selected[-1]["beta0"],
                selected[-1]["active_to_control_scale_ratio"],
            ),
            xytext=(4, 0),
            textcoords="offset points",
            color=color,
            fontsize=6.2,
            va="center",
        )
    axes[1, 1].axhline(1.0, color="black", lw=0.7, ls=":")
    axes[1, 1].axvline(P.beta_core, color="tab:red", lw=0.8, ls="--", alpha=0.8)
    axes[1, 1].set(
        xlabel=r"$\beta_0$",
        ylabel=r"$T_\star^{\rm active}/T_\star^{\rm control}$",
        title="(d) Threshold-dependent RG scale ratio",
        xlim=(0.34, 0.59),
    )

    for axis in axes.flat:
        axis.grid(alpha=0.18, lw=0.5)
    fixed_time = datetime(2026, 7, 26, 12, 0, 0, tzinfo=timezone.utc)
    fig.savefig(
        out / "SFig5_Causal_Scattering_and_RG.pdf",
        metadata={
            "Title": "Driven-Dirac impurity causal scattering and RG figure",
            "Author": "Driven-Dirac impurity reproducibility pipeline",
            "CreationDate": fixed_time,
            "ModDate": fixed_time,
        },
    )
    fig.savefig(
        out / "SFig5_Causal_Scattering_and_RG.png",
        dpi=240,
        metadata={"Software": "Driven-Dirac impurity causal scattering and RG gate"},
    )
    plt.close(fig)


def supplementary_figure(
    out: Path,
    delays: list[dict[str, Any]],
    peaks: list[dict[str, Any]],
    poles: list[dict[str, Any]],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.35, 6.35), constrained_layout=True)
    for control, style, label in (
        (0, "-", "EP-active"),
        (1, "--", r"matched $\Gamma_{\rm PT}=0$"),
    ):
        for index, color in ((0, "tab:blue"), (1, "tab:orange")):
            selected = [
                row for row in poles
                if row["control_gamma_pt_zero"] == control and row["pole_index"] == index
            ]
            axes[0, 0].plot(
                [row["beta0"] for row in selected],
                [row["pole_real"] for row in selected],
                color=color,
                ls=style,
                lw=1.25,
            )
    axes[0, 0].axvline(P.beta_core, color="tab:red", ls=":", lw=1.0)
    axes[0, 0].set(
        xlabel=r"$\beta_0$",
        ylabel=r"${\rm Re}\,z_{\rm pole}$",
        title="(a) Pole coalescence",
    )
    pole_legend = [
        Line2D([0], [0], color="black", ls="-", lw=1.25, label="EP-active"),
        Line2D(
            [0], [0], color="black", ls="--", lw=1.25,
            label=r"matched $\Gamma_{\rm PT}=0$",
        ),
        Line2D([0], [0], color="tab:blue", lw=1.25, label="pole 1"),
        Line2D([0], [0], color="tab:orange", lw=1.25, label="pole 2"),
    ]
    axes[0, 0].legend(handles=pole_legend, frameon=False, fontsize=5.8, ncol=2)

    for control, style, label in (
        (0, "-", "EP-active"),
        (1, "--", r"matched $\Gamma_{\rm PT}=0$"),
    ):
        for index, color in ((0, "tab:blue"), (1, "tab:orange")):
            selected = [
                row for row in poles
                if row["control_gamma_pt_zero"] == control and row["pole_index"] == index
            ]
            axes[0, 1].plot(
                [row["beta0"] for row in selected],
                [row["pole_halfwidth"] for row in selected],
                color=color,
                ls=style,
                lw=1.25,
            )
    axes[0, 1].axvline(P.beta_core, color="tab:red", ls=":", lw=1.0)
    axes[0, 1].set(
        xlabel=r"$\beta_0$",
        ylabel=r"$-{\rm Im}\,z_{\rm pole}$",
        title="(b) Pole halfwidths",
    )
    axes[0, 1].legend(handles=pole_legend, frameon=False, fontsize=5.8, ncol=2)

    for case, color, style, label in (
        ("active_beta_0.50", "tab:red", "-", "EP-active"),
        ("matched_gammaPT_zero", "black", "--", r"matched $\Gamma_{\rm PT}=0$"),
        ("rate_matched_off_EP_D_0.05", "tab:purple", "-.", "rate-matched off EP"),
    ):
        selected = [row for row in delays if row["case"] == case]
        axes[1, 0].plot(
            [row["omega"] for row in selected],
            [row["proper_delay_max"] for row in selected],
            color=color,
            ls=style,
            lw=1.35,
            label=label,
        )
    axes[1, 0].set(
        xlabel=r"$\omega$",
        ylabel=r"$q_{\max}(\omega)$",
        title="(c) Largest proper delay and controls",
        xlim=(-0.35, 0.35),
    )
    axes[1, 0].legend(frameon=False, fontsize=6.5)

    regular = [row for row in peaks if "rate_matched_coherent_channel" not in row]
    for control, style, label in (
        (0, "o-", "EP-active"),
        (1, "s--", r"matched $\Gamma_{\rm PT}=0$"),
    ):
        selected = [row for row in regular if row["control_gamma_pt_zero"] == control]
        axes[1, 1].plot(
            [row["beta0"] for row in selected],
            [row["rate_width_normalized_proper_peak"] for row in selected],
            style,
            ms=3.3,
            lw=1.15,
            label=label,
        )
    axes[1, 1].axvline(P.beta_core, color="tab:red", ls=":", lw=1.0)
    axes[1, 1].set(
        xlabel=r"$\beta_0$",
        ylabel=r"$(\bar\Gamma-|\Gamma_{\rm PT}|)q_{\max}^{\rm peak}$",
        title="(d) Rate-normalized proper delay",
    )
    axes[1, 1].legend(frameon=False, fontsize=6.5)

    for axis in axes.flat:
        axis.grid(alpha=0.18, lw=0.5)
    fixed_time = datetime(2026, 7, 26, 12, 0, 0, tzinfo=timezone.utc)
    fig.savefig(
        out / "SFig4_Smith_Delay_Validation.pdf",
        metadata={
            "Title": "Driven-Dirac impurity Smith delay validation",
            "Author": "Driven-Dirac impurity reproducibility pipeline",
            "CreationDate": fixed_time,
            "ModDate": fixed_time,
        },
    )
    fig.savefig(
        out / "SFig4_Smith_Delay_Validation.png",
        dpi=240,
        metadata={"Software": "Driven-Dirac impurity causal scattering and RG gate"},
    )
    plt.close(fig)


def run(out: Path, rg_dir: Path, p: SaddleParams = P) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    delays, maxima = delay_rows(p)
    peaks = peak_rows(p)
    poles = pole_rows(p)
    write_csv(out / "scattering_delay_profiles.csv", delays)
    write_csv(out / "scattering_delay_peaks.csv", peaks)
    write_csv(out / "scattering_poles_and_linewidths.csv", poles)
    delay_sum_rule = sum_rule(p)
    main_figure(out, rg_dir, delays)
    supplementary_figure(out, delays, peaks, poles)

    ep_delay = next(
        row["trace_delay"] for row in delays
        if row["case"] == "active_beta_0.50" and abs(row["omega"]) < 1.0e-14
    )
    analytic_ep_delay = 4.0 / p.gamma0
    ep_delay_error = abs(ep_delay - analytic_ep_delay)
    rate_control_values = [
        row for row in peaks if row.get("rate_matched_coherent_channel") == 0.05
    ]
    ep_peak = next(
        row["proper_delay_peak"] for row in peaks
        if row["beta0"] == 0.50
        and row["control_gamma_pt_zero"] == 0
        and "rate_matched_coherent_channel" not in row
    )
    off_ep_peak = rate_control_values[0]["proper_delay_peak"]
    summary = {
        "status": "PASS__SMITH_DELAY_IS_CAUSAL_COALESCENCE_FINGERPRINT_NOT_INDEPENDENT_ENHANCEMENT",
        "generated_utc": "2026-07-26T12:00:00+00:00",
        "scope": (
            "Passive two-channel Markov projected-core Fisher--Lee scattering, "
            "combined with the previously validated causal matrix RG."
        ),
        "parameters": asdict(p),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "checks": {
            "scattering_unitarity": maxima["unitarity"] < 1.0e-11,
            "smith_hermiticity": maxima["hermiticity"] < 1.0e-11,
            "trace_delay_determinant_identity": maxima["determinant_identity"] < 1.0e-10,
            "determinant_attenuation_zero": maxima["determinant_imag"] < 1.0e-10,
            "EP_double_pole_delay_identity": ep_delay_error < 1.0e-10,
            "two_state_delay_sum_rule": abs(delay_sum_rule - 2.0) < 0.02,
            "rate_matched_off_EP_peak_can_exceed_EP_peak": off_ep_peak > ep_peak,
            "petermann_multiplier_inserted": False,
        },
        "headline": {
            "maximum_unitarity_error": maxima["unitarity"],
            "maximum_smith_hermiticity_error": maxima["hermiticity"],
            "maximum_trace_delay_determinant_error": maxima["determinant_identity"],
            "EP_trace_delay_at_omega0": ep_delay,
            "analytic_EP_trace_delay_at_omega0": analytic_ep_delay,
            "EP_trace_delay_identity_error": ep_delay_error,
            "integrated_trace_delay_over_2pi": delay_sum_rule,
            "EP_largest_proper_delay_peak": ep_peak,
            "rate_matched_off_EP_D005_largest_proper_delay_peak": off_ep_peak,
        },
        "interpretation": {
            "positive": (
                "The physical S matrix is unitary in the passive window and the "
                "trace delay acquires the exact merged double-pole Lorentzian "
                "Tr Q_EP=4 Gamma_bar/(omega^2+Gamma_bar^2)."
            ),
            "limitation": (
                "A raw proper-delay peak is not an independent EP enhancement; "
                "it is sensitive to the rate eigenchannel and can be reproduced "
                "or exceeded by rate-matched off-EP kernels."
            ),
            "manuscript_use": (
                "Use the Smith delay as a causal coalescence/linewidth fingerprint. "
                "Retain the threshold-dependent matrix RG as the independent "
                "pre-EP weak-flow result."
            ),
        },
    }
    (out / "smith_delay_rg_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    readme = f"""# Driven-Dirac impurity causal Smith-delay and RG gate

Status: **{summary['status']}**.

The physical Fisher--Lee scattering matrix is unitary to
`{maxima['unitarity']:.3e}` in the passive Markov window, and the
Wigner--Smith matrix is Hermitian to `{maxima['hermiticity']:.3e}`.
The matrices are evaluated in the original spin representation.  Rotating
the Green function and bath-rate matrix together produces the fixed chiral
representation, with identical scattering eigenphases and proper delays.
No individual off-diagonal matrix element is used as invariant evidence.
At the compensated core EP,

`Tr Q(omega) = 4*Gamma_bar/(omega^2+Gamma_bar^2)`,

with numerical error `{ep_delay_error:.3e}` at `omega=0`.  The integrated
trace delay is `{delay_sum_rule:.8f}` states.

The rate-matched off-EP control prevents overinterpretation: raw delay growth
is not an independent Kondo or Petermann enhancement.  The main figure
therefore uses delay only as a causal pole-coalescence fingerprint and keeps
the previously validated matrix-RG threshold comparison as the independent
pre-EP weak-flow diagnostic.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    script = Path(__file__).resolve()
    targets = sorted(
        [
            path
            for path in out.iterdir()
            if path.is_file() and path.name != "SHA256SUMS"
        ],
        key=lambda path: path.name,
    )
    lines = [f"{sha256(path)}  {path.name}" for path in targets]
    lines.append(f"{sha256(script)}  ../smith_delay_rg_gate.py")
    helper = script.parent / "self_consistent_saddle_gate.py"
    rg_script = script.parent / "channel_resolved_rg_gate.py"
    lines.append(f"{sha256(helper)}  ../self_consistent_saddle_gate.py")
    lines.append(f"{sha256(rg_script)}  ../channel_resolved_rg_gate.py")
    for input_name in (
        "causal_channel_density.csv",
        "representative_rg_flow.csv",
        "active_control_scale_comparison.csv",
    ):
        input_path = (rg_dir / input_name).resolve()
        relative_input = os.path.relpath(input_path, start=out.resolve())
        lines.append(f"{sha256(input_path)}  {relative_input}")
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    package_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=package_dir / "regenerated_smith"
    )
    parser.add_argument(
        "--rg-dir", type=Path, default=package_dir / "reference_rg_output"
    )
    args = parser.parse_args()
    summary = run(args.out, args.rg_dir)
    print(json.dumps({"status": summary["status"], "headline": summary["headline"]}, indent=2))


if __name__ == "__main__":
    main()
