#!/usr/bin/env python3
"""Aggregate raw complete-basis DM-NRG response without imposing an EP model.

Peak locations are detected directly from the largest singular value of the
computed dissipative response matrix.  The analyzer neither fixes the number
of peaks nor fits a square-root or exceptional-point functional form.
"""
from __future__ import annotations

import argparse
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences, peak_widths


@dataclass
class RunData:
    path: Path
    summary: dict[str, Any]
    spectrum: pd.DataFrame
    time: pd.DataFrame | None
    quality: pd.DataFrame | None


def load_runs(root: Path) -> list[RunData]:
    runs: list[RunData] = []
    for summary_path in sorted(root.rglob("dmnrg_response_summary.toml")):
        with summary_path.open("rb") as f:
            summary = tomllib.load(f)
        spectrum_path = summary_path.with_name("dmnrg_response_spectra.csv")
        if not spectrum_path.exists():
            continue
        spectrum = pd.read_csv(spectrum_path)
        time_path = summary_path.with_name("dmnrg_response_time.csv")
        quality_path = summary_path.with_name("dmnrg_response_quality.csv")
        time = pd.read_csv(time_path) if time_path.exists() else None
        quality = pd.read_csv(quality_path) if quality_path.exists() else None
        runs.append(RunData(summary_path.parent, summary, spectrum, time, quality))
    return runs


def components(summary: dict[str, Any]) -> list[str]:
    return [str(x) for x in summary.get("components", ["x", "z"])]


def response_matrices(run: RunData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    comps = components(run.summary)
    n = len(comps)
    omega = run.spectrum["omega"].to_numpy(float)
    chi = np.zeros((len(omega), n, n), dtype=complex)
    for ia, a in enumerate(comps):
        for ib, b in enumerate(comps):
            re = run.spectrum[f"ReChi_{a}{b}"].to_numpy(float)
            im = run.spectrum[f"ImChi_{a}{b}"].to_numpy(float)
            chi[:, ia, ib] = re + 1j * im
    dissipative = -np.imag(chi) / np.pi
    return omega, chi, dissipative


def singular_signals(chi: np.ndarray, dissipative: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s_chi = np.array([np.linalg.svd(m, compute_uv=False)[0] for m in chi])
    s_diss = np.array([np.linalg.svd(m, compute_uv=False)[0] for m in dissipative])
    return s_chi, s_diss


def detect_peaks(omega: np.ndarray, signal: np.ndarray, eta: float) -> list[dict[str, float]]:
    finite = np.isfinite(signal)
    if finite.sum() < 5:
        return []
    y = np.where(finite, np.abs(signal), 0.0)
    ymax = float(np.max(y))
    if not math.isfinite(ymax) or ymax <= 0.0:
        return []
    dw = float(np.median(np.diff(omega)))
    minimum_distance = max(1, int(round(max(2.0 * eta, 2.0 * dw) / dw)))
    peaks, _ = find_peaks(y, prominence=0.015 * ymax, distance=minimum_distance)
    if len(peaks) == 0:
        return []
    prominences = peak_prominences(y, peaks)[0]
    widths = peak_widths(y, peaks, rel_height=0.5)[0] * dw
    records = [
        {
            "index": int(i),
            "frequency": float(omega[i]),
            "height": float(y[i]),
            "prominence": float(prom),
            "width": float(width),
        }
        for i, prom, width in zip(peaks, prominences, widths)
        if omega[i] > max(0.0, 0.5 * dw)
    ]
    records.sort(key=lambda x: x["prominence"], reverse=True)
    return records




def time_domain_diagnostic(run: RunData) -> tuple[float, str, int]:
    """Return the strongest nonzero FFT frequency from raw response components."""
    if run.time is None or len(run.time) < 8:
        return math.nan, "", 0
    t = run.time["time"].to_numpy(float)
    dt = float(np.median(np.diff(t)))
    if not math.isfinite(dt) or dt <= 0.0:
        return math.nan, "", 0
    candidate_columns = [c for c in run.time.columns if c.startswith("ReChi_")]
    best_frequency = math.nan
    best_component = ""
    best_power = -math.inf
    best_zero_crossings = 0
    for column in candidate_columns:
        y = run.time[column].to_numpy(float)
        finite = np.isfinite(y)
        if finite.sum() < 8:
            continue
        y = np.where(finite, y, 0.0)
        y = y - np.mean(y)
        scale = float(np.max(np.abs(y)))
        if scale <= 0.0:
            continue
        windowed = y * np.hanning(len(y))
        frequencies = np.fft.rfftfreq(len(y), d=dt)
        power = np.abs(np.fft.rfft(windowed))
        if len(power) <= 1:
            continue
        index = 1 + int(np.argmax(power[1:]))
        if power[index] > best_power:
            thresholded = y.copy()
            thresholded[np.abs(thresholded) < 0.05 * scale] = 0.0
            signs = np.sign(thresholded)
            nonzero = signs[signs != 0]
            crossings = int(np.sum(nonzero[1:] * nonzero[:-1] < 0)) if len(nonzero) > 1 else 0
            best_power = float(power[index])
            best_frequency = float(frequencies[index])
            best_component = column.removeprefix("ReChi_")
            best_zero_crossings = crossings
    return best_frequency, best_component, best_zero_crossings


def quality_metrics(run: RunData) -> tuple[float, float]:
    if run.quality is None or run.quality.empty:
        return math.nan, math.nan
    return (
        float(run.quality["selected_fraction"].min()),
        float(run.quality["selected_fraction"].median()),
    )


def summarize_run(run: RunData) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    omega, chi, dissipative = response_matrices(run)
    response_norm, dissipative_norm = singular_signals(chi, dissipative)
    eta = float(run.summary.get("eta", math.nan))
    peaks = detect_peaks(omega, dissipative_norm, eta)
    qmin, qmed = quality_metrics(run)
    time_fft_frequency, time_fft_component, time_zero_crossings = time_domain_diagnostic(run)

    strongest = peaks[0] if len(peaks) >= 1 else None
    second = peaks[1] if len(peaks) >= 2 else None
    two = sorted([strongest, second], key=lambda p: p["frequency"]) if second else []
    low = two[0] if two else strongest
    high = two[1] if two else None

    zero_index = int(np.argmin(np.abs(omega)))
    chi0 = chi[zero_index]
    eig0 = np.linalg.eigvals(chi0)
    sv0 = np.linalg.svd(chi0, compute_uv=False)

    summary = {
        "run_dir": str(run.path),
        "beta0": float(run.summary.get("beta0", math.nan)),
        "U": float(run.summary.get("U", math.nan)),
        "soc_ratio": float(run.summary.get("soc_ratio", math.nan)),
        "soc_lambda": float(run.summary.get("soc_lambda", math.nan)),
        "z_shift": float(run.summary.get("z_shift", math.nan)),
        "equilibrium_gate": bool(run.summary.get("equilibrium_gate", False)),
        "response_computed": bool(run.summary.get("response_computed", False)),
        "max_centered_imaginary_energy": float(run.summary.get("max_centered_imaginary_energy", math.nan)),
        "density_trace_error": float(run.summary.get("density_trace_error", math.nan)),
        "transition_count": int(run.summary.get("transition_count", 0)),
        "selected_fraction_min": qmin,
        "selected_fraction_median": qmed,
        "detected_peak_count": len(peaks),
        "peak_low_frequency": low["frequency"] if low else math.nan,
        "peak_low_width": low["width"] if low else math.nan,
        "peak_low_prominence": low["prominence"] if low else math.nan,
        "peak_high_frequency": high["frequency"] if high else math.nan,
        "peak_high_width": high["width"] if high else math.nan,
        "peak_high_prominence": high["prominence"] if high else math.nan,
        "peak_splitting": abs(high["frequency"] - low["frequency"]) if high and low else math.nan,
        "chi0_singular_1": float(sv0[0]),
        "chi0_singular_2": float(sv0[1]) if len(sv0) > 1 else math.nan,
        "chi0_eig1_real": float(np.real(eig0[0])),
        "chi0_eig1_imag": float(np.imag(eig0[0])),
        "chi0_eig2_real": float(np.real(eig0[1])) if len(eig0) > 1 else math.nan,
        "chi0_eig2_imag": float(np.imag(eig0[1])) if len(eig0) > 1 else math.nan,
        "time_fft_peak_frequency": time_fft_frequency,
        "time_fft_component": time_fft_component,
        "time_zero_crossings": time_zero_crossings,
    }
    peak_rows: list[dict[str, Any]] = []
    for rank, peak in enumerate(peaks, start=1):
        peak_rows.append({
            "run_dir": str(run.path),
            "beta0": summary["beta0"],
            "U": summary["U"],
            "soc_ratio": summary["soc_ratio"],
            "z_shift": summary["z_shift"],
            "equilibrium_gate": summary["equilibrium_gate"],
            "rank_by_prominence": rank,
            **peak,
        })
    curve = pd.DataFrame({
        "omega": omega,
        "response_singular_1": response_norm,
        "dissipative_singular_1": dissipative_norm,
    })
    return summary, peak_rows, curve


def group_tag(U: float, soc: float) -> str:
    return f"U_{U:g}_soc_{soc:g}".replace("-", "m").replace(".", "p")


def plot_group(group: pd.DataFrame, curves: dict[str, pd.DataFrame], peaks: pd.DataFrame,
               out: Path, U: float, soc: float) -> None:
    group = group.sort_values("beta0")
    tag = group_tag(U, soc)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for _, row in group.iterrows():
        subset = peaks[peaks["run_dir"] == row["run_dir"]]
        marker = "o" if row["equilibrium_gate"] else "x"
        for _, p in subset.iterrows():
            ax.scatter(row["beta0"], p["frequency"], marker=marker,
                       s=35, alpha=0.9)
    ax.set_xlabel(r"$\beta_0$")
    ax.set_ylabel("detected response-peak frequency")
    ax.set_title(fr"Raw DM-NRG peaks: $U={U:g}$, $\lambda/k_{{\max}}={soc:g}$")
    ax.axvline(0.5, linestyle="--", linewidth=1)
    fig.tight_layout()
    fig.savefig(out / f"DMNRG_peak_frequencies_{tag}.pdf")
    fig.savefig(out / f"DMNRG_peak_frequencies_{tag}.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    valid = np.isfinite(group["peak_splitting"])
    ax.plot(group.loc[valid, "beta0"], group.loc[valid, "peak_splitting"],
            marker="o")
    failed = ~group["equilibrium_gate"] & valid
    ax.scatter(group.loc[failed, "beta0"], group.loc[failed, "peak_splitting"],
               marker="x", s=55)
    ax.set_xlabel(r"$\beta_0$")
    ax.set_ylabel("separation of two most prominent peaks")
    ax.set_title(fr"No imposed EP fit: $U={U:g}$, $\lambda/k_{{\max}}={soc:g}$")
    ax.axvline(0.5, linestyle="--", linewidth=1)
    fig.tight_layout()
    fig.savefig(out / f"DMNRG_peak_splitting_{tag}.pdf")
    fig.savefig(out / f"DMNRG_peak_splitting_{tag}.png", dpi=220)
    plt.close(fig)

    available = [r for _, r in group.iterrows() if r["run_dir"] in curves]
    if available:
        omega_union = curves[available[0]["run_dir"]]["omega"].to_numpy()
        matrix = np.full((len(available), len(omega_union)), np.nan)
        betas = []
        for i, r in enumerate(available):
            c = curves[r["run_dir"]]
            matrix[i] = c["dissipative_singular_1"].to_numpy()
            betas.append(r["beta0"])
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        beta_min, beta_max = min(betas), max(betas)
        if beta_max == beta_min:
            beta_min -= 0.005
            beta_max += 0.005
        image = ax.imshow(matrix, aspect="auto", origin="lower",
                          extent=[omega_union.min(), omega_union.max(),
                                  beta_min, beta_max], interpolation="nearest")
        fig.colorbar(image, ax=ax, label="largest dissipative singular value")
        ax.set_xlabel(r"frequency $\omega$")
        ax.set_ylabel(r"$\beta_0$")
        ax.set_title(fr"Raw response map: $U={U:g}$, $\lambda/k_{{\max}}={soc:g}$")
        fig.tight_layout()
        fig.savefig(out / f"DMNRG_response_map_{tag}.pdf")
        fig.savefig(out / f"DMNRG_response_map_{tag}.png", dpi=220)
        plt.close(fig)

    representative = group.sort_values("beta0")
    if len(representative) > 5:
        indices = np.unique(np.linspace(0, len(representative)-1, 5).round().astype(int))
        representative = representative.iloc[indices]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    plotted = False
    for _, row in representative.iterrows():
        run_dir = Path(row["run_dir"])
        time_path = run_dir / "dmnrg_response_time.csv"
        if not time_path.exists():
            continue
        td = pd.read_csv(time_path)
        column = "ReChi_xx" if "ReChi_xx" in td.columns else next(
            (c for c in td.columns if c.startswith("ReChi_")), None)
        if column is None:
            continue
        y = td[column].to_numpy(float)
        scale = np.nanmax(np.abs(y))
        if not np.isfinite(scale) or scale <= 0.0:
            continue
        ax.plot(td["time"], y/scale, label=fr"$\beta_0={row['beta0']:g}$")
        plotted = True
    if plotted:
        ax.set_xlabel("time")
        ax.set_ylabel("normalized raw response")
        ax.set_title(fr"DM-NRG time traces: $U={U:g}$, $\lambda/k_{{\max}}={soc:g}$")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"DMNRG_time_traces_{tag}.pdf")
        fig.savefig(out / f"DMNRG_time_traces_{tag}.png", dpi=220)
    plt.close(fig)


def write_audit(summary: pd.DataFrame, out: Path) -> None:
    lines = [
        "# Complete-basis zero-temperature DM-NRG response audit",
        "",
        "No square-root law, EP location, number of peaks, or spin-resolved Kondo scale is imposed by the extraction.",
        "Peak candidates are local maxima of the largest singular value of the dissipative response matrix.",
        "",
        f"- runs analyzed: **{len(summary)}**",
        f"- runs passing the real-spectrum equilibrium gate: **{int(summary['equilibrium_gate'].sum())}**",
        f"- runs with at least one detected peak: **{int((summary['detected_peak_count'] >= 1).sum())}**",
        f"- runs with at least two detected peaks: **{int((summary['detected_peak_count'] >= 2).sum())}**",
        "",
        "Exploratory complex-spectrum responses are retained but are not assigned an equilibrium thermodynamic interpretation.",
        "The reported widths include the configured broadening eta and are not deconvolved Kondo scales.",
        "",
        "| beta0 | U | soc ratio | equilibrium | peaks | low peak | high peak | splitting | min retained weight |",
        "|---:|---:|---:|:---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in summary.sort_values(["U", "soc_ratio", "beta0", "z_shift"]).iterrows():
        def f(x: float) -> str:
            return "--" if not math.isfinite(float(x)) else f"{float(x):.7g}"
        lines.append(
            f"| {r.beta0:g} | {r.U:g} | {r.soc_ratio:g} | "
            f"{'pass' if r.equilibrium_gate else 'exploratory'} | "
            f"{int(r.detected_peak_count)} | {f(r.peak_low_frequency)} | "
            f"{f(r.peak_high_frequency)} | {f(r.peak_splitting)} | "
            f"{f(r.selected_fraction_min)} |"
        )
    (out / "DMNRG_RESPONSE_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    root = args.root.resolve()
    out = (args.out or root / "analysis").resolve()
    out.mkdir(parents=True, exist_ok=True)

    runs = load_runs(root)
    if not runs:
        raise SystemExit(f"no dmnrg_response_summary.toml with spectra found under {root}")

    summaries: list[dict[str, Any]] = []
    peak_rows: list[dict[str, Any]] = []
    curves: dict[str, pd.DataFrame] = {}
    for run in runs:
        summary, peaks, curve = summarize_run(run)
        summaries.append(summary)
        peak_rows.extend(peaks)
        curves[str(run.path)] = curve

    summary_df = pd.DataFrame(summaries)
    peaks_df = pd.DataFrame(peak_rows)
    summary_df.to_csv(out / "dmnrg_response_summary.csv", index=False)
    peaks_df.to_csv(out / "dmnrg_response_peaks.csv", index=False)
    for (U, soc), group in summary_df.groupby(["U", "soc_ratio"], dropna=False):
        plot_group(group, curves, peaks_df, out, float(U), float(soc))
    write_audit(summary_df, out)
    print(out / "DMNRG_RESPONSE_AUDIT.md")


if __name__ == "__main__":
    main()
