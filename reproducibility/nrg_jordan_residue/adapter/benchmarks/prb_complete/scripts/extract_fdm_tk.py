#!/usr/bin/env python3
"""Aggregate complete-basis/FDM entropy scales over beta0 and z shifts."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import tomllib
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_toml_relaxed(path: Path) -> dict[str, Any]:
    """Read TOML and normalize legacy Julia NaN/Inf spellings."""
    text = path.read_text(encoding="utf-8")
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        mapping = {"NaN": "nan", "Inf": "inf", "-Inf": "-inf"}
        fixed = re.sub(
            r"(?m)(=\s*)(-?Inf|NaN)(\s*(?:#.*)?$)",
            lambda m: m.group(1) + mapping[m.group(2)] + m.group(3),
            text,
        )
        return tomllib.loads(fixed)


def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def aggregate(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    manifest = json.loads((root / "fdm_tk_manifest.json").read_text(encoding="utf-8"))
    beta_ep = float(manifest.get("beta_EP", 0.5))
    raw: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    for item in manifest["runs"]:
        run = Path(item["output"])
        summary_path = run / "fdm_tk_summary.toml"
        thermo_path = run / "fdm_thermodynamics.csv"
        row: dict[str, Any] = {"beta0": float(item["beta0"]),
                               "z_shift": float(item["z_shift"]),
                               "run_dir": str(run)}
        if not summary_path.exists():
            row.update({"status": "missing", "tk_valid": False, "T_K_entropy": math.nan})
            raw.append(row)
            continue
        summary = load_toml_relaxed(summary_path)
        row.update(summary)
        row["status"] = "complete"
        raw.append(row)
        if thermo_path.exists():
            for r in read_csv(thermo_path):
                curves.append({"beta0": row["beta0"], "z_shift": row["z_shift"],
                               **{k: float(v) for k, v in r.items()}})
    return raw, curves, beta_ep


def summarize(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    betas = sorted({float(r["beta0"]) for r in raw})
    out: list[dict[str, Any]] = []
    for beta in betas:
        group = [r for r in raw if float(r["beta0"]) == beta]
        vals = [float(r["T_K_entropy"]) for r in group
                if bool(r.get("tk_valid", False)) and finite(r.get("T_K_entropy"))]
        if vals:
            logs = np.log(vals)
            tk = float(np.exp(np.mean(logs)))
            spread = float(np.std(logs, ddof=1)) if len(vals) > 1 else 0.0
            low = tk * math.exp(-spread)
            high = tk * math.exp(spread)
        else:
            tk = low = high = math.nan
        real_fraction = np.mean([bool(r.get("spectrum_real", False)) for r in group]) if group else 0.0
        count_fraction = np.mean([bool(r.get("complete_basis_count_pass", False)) for r in group]) if group else 0.0
        max_imag = max([float(r.get("max_centered_imaginary_energy", math.nan))
                        for r in group if finite(r.get("max_centered_imaginary_energy"))], default=math.nan)
        low_entropy = np.nanmedian([float(r.get("low_temperature_entropy", math.nan)) for r in group])
        max_entropy = np.nanmedian([float(r.get("maximum_impurity_entropy", math.nan)) for r in group])
        out.append({"beta0": beta, "z_count": len(group), "valid_z_count": len(vals),
                    "T_K_entropy_zavg": tk, "T_K_entropy_low": low,
                    "T_K_entropy_high": high, "real_spectrum_fraction": real_fraction,
                    "complete_basis_count_fraction": count_fraction,
                    "max_centered_imaginary_energy": max_imag,
                    "median_low_temperature_entropy": low_entropy,
                    "median_maximum_impurity_entropy": max_entropy})
    return out


def make_plots(summary: list[dict[str, Any]], curves: list[dict[str, Any]], beta_ep: float,
               out: Path) -> None:
    beta = np.array([r["beta0"] for r in summary], float)
    tk = np.array([r["T_K_entropy_zavg"] for r in summary], float)
    lo = np.array([r["T_K_entropy_low"] for r in summary], float)
    hi = np.array([r["T_K_entropy_high"] for r in summary], float)
    valid = np.isfinite(tk) & (tk > 0)

    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    if valid.any():
        yerr = np.vstack([tk[valid]-lo[valid], hi[valid]-tk[valid]])
        ax.errorbar(beta[valid], tk[valid], yerr=yerr, marker="o", capsize=3,
                    label=r"$T_K^{S}$: $S_{\rm imp}=\frac{1}{2}\ln 2$")
    invalid = ~valid
    if invalid.any():
        floor = np.nanmin(tk[valid])*0.6 if valid.any() else 1e-7
        ax.scatter(beta[invalid], np.full(invalid.sum(), floor), marker="x",
                   label="thermodynamic gate failed")
    ax.axvline(beta_ep, linestyle="--", linewidth=1.2, label=r"nominal $\beta_{\rm EP}$")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\beta_0$")
    ax.set_ylabel(r"entropy crossover scale $T_K^S$")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out/"FDM_TK_entropy_vs_beta0.pdf")
    fig.savefig(out/"FDM_TK_entropy_vs_beta0.png", dpi=250)
    plt.close(fig)

    # Curves at z closest to 0.5; select a manageable representative set.
    if curves:
        unique_beta = sorted({float(r["beta0"]) for r in curves})
        if len(unique_beta) > 7:
            idx = np.linspace(0, len(unique_beta)-1, 7).round().astype(int)
            selected = {unique_beta[i] for i in idx}
        else:
            selected = set(unique_beta)
        fig, ax = plt.subplots(figsize=(6.4, 4.7))
        for b in sorted(selected):
            choices = sorted({float(r["z_shift"]) for r in curves if float(r["beta0"]) == b},
                             key=lambda z: abs(z-0.5))
            if not choices:
                continue
            z = choices[0]
            rows = sorted([r for r in curves if float(r["beta0"]) == b and
                           float(r["z_shift"]) == z], key=lambda r: r["temperature"])
            T = np.array([r["temperature"] for r in rows])
            S = np.array([r["entropy_impurity"] for r in rows])
            if np.isfinite(S).any():
                ax.plot(T, S, label=fr"$\beta_0={b:g}$")
        ax.axhline(0.5*math.log(2), linestyle="--", linewidth=1.0,
                   label=r"$\frac{1}{2}\ln 2$")
        ax.axhline(math.log(2), linestyle=":", linewidth=1.0,
                   label=r"$\ln 2$")
        ax.set_xscale("log")
        ax.set_xlabel(r"temperature $T$")
        ax.set_ylabel(r"impurity entropy $S_{\rm imp}$")
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(out/"FDM_impurity_entropy_curves.pdf")
        fig.savefig(out/"FDM_impurity_entropy_curves.png", dpi=250)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    imag = np.array([r["max_centered_imaginary_energy"] for r in summary], float)
    realfrac = np.array([r["real_spectrum_fraction"] for r in summary], float)
    ax.semilogy(beta, np.maximum(imag, 1e-18), marker="o",
                label="max centered imaginary energy")
    ax2 = ax.twinx()
    ax2.plot(beta, realfrac, marker="s", linestyle="--",
             label="real-spectrum z fraction")
    ax.axvline(beta_ep, linestyle="--", linewidth=1.0)
    ax.set_xlabel(r"$\beta_0$")
    ax.set_ylabel("complex-spectrum diagnostic")
    ax2.set_ylabel("fraction passing equilibrium gate")
    lines = ax.get_lines()+ax2.get_lines()
    labels = [l.get_label() for l in lines if not l.get_label().startswith("_")]
    ax.legend([l for l in lines if not l.get_label().startswith("_")], labels, loc="best")
    fig.tight_layout()
    fig.savefig(out/"FDM_thermodynamic_quality_vs_beta0.pdf")
    fig.savefig(out/"FDM_thermodynamic_quality_vs_beta0.png", dpi=250)
    plt.close(fig)


def report(summary: list[dict[str, Any]], beta_ep: float, out: Path) -> None:
    valid = [r for r in summary if finite(r["T_K_entropy_zavg"])]
    lines = ["# Complete-basis/FDM entropy-scale audit", "",
             "The calculation uses all NRG discarded states with Anders--Schiller environment multiplicities.",
             "The reported scale is defined only by the explicit criterion",
             r"`S_imp(T_K^S)=ln(2)/2`; it is not a spectral linewidth.", "",
             f"- nominal beta_EP: `{beta_ep}`",
             f"- beta points: **{len(summary)}**",
             f"- beta points with valid entropy scale: **{len(valid)}**", "",
             "| beta0 | valid z / total | T_K^S (z average) | real-spectrum fraction | max Im(E-E0) |",
             "|---:|---:|---:|---:|---:|"]
    for r in summary:
        tk = r["T_K_entropy_zavg"]
        tktext = f"{tk:.8g}" if finite(tk) else "--"
        lines.append(f"| {r['beta0']:.6g} | {r['valid_z_count']}/{r['z_count']} | {tktext} | "
                     f"{r['real_spectrum_fraction']:.3f} | {r['max_centered_imaginary_energy']:.3e} |")
    lines += ["", "## Interpretation boundary", "",
              "A `T_K^S` entry is emitted only when the complete-basis count closes exactly,",
              "the density matrix normalizes, the spectrum is real in the relative frame,",
              "the low-temperature entropy is screened, and an ln(2) local-moment regime is resolved.",
              "Complex-spectrum points are retained as adverse controls and receive no thermodynamic T_K label."]
    (out/"FDM_TK_AUDIT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    raw, curves, beta_ep = aggregate(args.root.resolve())
    summary = summarize(raw)
    raw_fields = sorted({k for r in raw for k in r.keys()})
    write_csv(args.out/"fdm_tk_per_run.csv", raw_fields, raw)
    fields = list(summary[0].keys()) if summary else []
    if fields:
        write_csv(args.out/"fdm_tk_vs_beta0.csv", fields, summary)
    make_plots(summary, curves, beta_ep, args.out)
    report(summary, beta_ep, args.out)
    print(args.out/"FDM_TK_AUDIT.md")


if __name__ == "__main__":
    main()
