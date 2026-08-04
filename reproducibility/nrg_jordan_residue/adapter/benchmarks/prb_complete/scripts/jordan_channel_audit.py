#!/usr/bin/env python3
"""Jordan-channel survival audit for PTDirac NH-NRG transition residues.

This script reads every run directory below OUTPUT_ROOT containing
RUN_SUMMARY.txt and impurity_transition_residue_matrix.csv. It selects an
impurity-supported pair independently in each (iteration, charge) sector,
forms

    A = W_+ + W_-
    B = 0.5 (W_+ - W_-) (z_+ - z_-),

and decomposes B into trace and Pauli/Jordan-active components. The bare EP
nilpotent direction is

    N_EP = Delta_eff sigma_z + i Gamma_PT sigma_x.

A matrix Jordan residue survives when B is finite, nearly traceless, and
aligned with N_EP. This audit does not insert or assume the quartic pole law.
"""
from __future__ import annotations

import argparse
import itertools
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

MATRIX_KEYS = ("11", "12", "21", "22")


@dataclass(frozen=True)
class RunParameters:
    beta0: float
    U: float
    eps_d: float
    delta_eff: float
    gamma_pt: float
    delta_coh: float
    V: float
    soc_lambda: float
    F_lambda: float


def _float(text: str, default: float = math.nan) -> float:
    try:
        return float(text.strip())
    except Exception:
        return default


def parse_summary(path: Path) -> RunParameters:
    vals: dict[str, float] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        for item in raw.split(";"):
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            vals[key.strip()] = _float(value)
    required = ("beta0", "U", "eps_d", "Delta_eff", "Gamma_PT", "Delta_coh", "V")
    missing = [key for key in required if key not in vals or not np.isfinite(vals[key])]
    if missing:
        raise ValueError(f"{path}: missing numeric fields {missing}")
    return RunParameters(
        beta0=vals["beta0"],
        U=vals["U"],
        eps_d=vals["eps_d"],
        delta_eff=vals["Delta_eff"],
        gamma_pt=vals["Gamma_PT"],
        delta_coh=vals["Delta_coh"],
        V=vals["V"],
        soc_lambda=vals.get("soc_lambda", 0.0),
        F_lambda=vals.get("F_lambda", math.nan),
    )


def matrix_from_row(row: pd.Series) -> np.ndarray:
    prefix = "add" if int(row["charge"]) == int(row["ground_charge"]) + 1 else "rem"
    vals = [
        complex(float(row[f"{prefix}{key}_real"]), float(row[f"{prefix}{key}_imag"]))
        for key in MATRIX_KEYS
    ]
    return np.asarray(vals, dtype=np.complex128).reshape(2, 2)


def select_pair(group: pd.DataFrame, pool: int, support_fraction: float):
    if len(group) < 2:
        return None
    g = group.sort_values(["matrix_support_abs", "matrix_rank"], ascending=[False, True]).head(pool).copy()
    maximum = float(g["matrix_support_abs"].max())
    g = g[g["matrix_support_abs"] >= support_fraction * maximum]
    if len(g) < 2:
        return None
    best = None
    rows = [row for _, row in g.iterrows()]
    for first, second in itertools.combinations(rows, 2):
        z1 = complex(float(first["pole_real"]), float(first["pole_imag"]))
        z2 = complex(float(second["pole_real"]), float(second["pole_imag"]))
        key = (
            abs(z1 - z2),
            -min(float(first["matrix_support_abs"]), float(second["matrix_support_abs"])),
            int(first["matrix_rank"]) + int(second["matrix_rank"]),
        )
        if best is None or key < best[0]:
            best = (key, first, second)
    return None if best is None else (best[1], best[2])


def ccols(prefix: str, value: complex) -> dict[str, float]:
    return {
        f"{prefix}_real": float(np.real(value)),
        f"{prefix}_imag": float(np.imag(value)),
        f"{prefix}_abs": float(abs(value)),
    }


def audit_run(run_dir: Path, root: Path, args: argparse.Namespace) -> list[dict]:
    params = parse_summary(run_dir / "RUN_SUMMARY.txt")
    data = pd.read_csv(run_dir / "impurity_transition_residue_matrix.csv")
    rows: list[dict] = []

    # Nilpotent direction of the compensated local PT core. Delta_coh is a
    # detuning and is intentionally not included in N_EP.
    N = np.asarray(
        [[params.delta_eff, 1j * params.gamma_pt],
         [1j * params.gamma_pt, -params.delta_eff]],
        dtype=np.complex128,
    )
    Nnorm = float(np.linalg.norm(N))

    wanted_iterations = None if args.iterations is None else set(args.iterations)
    for (iteration, charge), group in data.groupby(["iteration", "charge"], sort=True):
        iteration = int(iteration)
        if wanted_iterations is not None and iteration not in wanted_iterations:
            continue
        selected = select_pair(group, args.pair_pool, args.support_fraction)
        if selected is None:
            continue
        r1, r2 = selected
        z1 = complex(float(r1["pole_real"]), float(r1["pole_imag"]))
        z2 = complex(float(r2["pole_real"]), float(r2["pole_imag"]))
        if (z1.real, z1.imag) <= (z2.real, z2.imag):
            minus, plus, zminus, zplus = r1, r2, z1, z2
        else:
            minus, plus, zminus, zplus = r2, r1, z2, z1

        Wminus = matrix_from_row(minus)
        Wplus = matrix_from_row(plus)
        dz = zplus - zminus
        A = Wplus + Wminus
        B = 0.5 * (Wplus - Wminus) * dz

        b0 = 0.5 * np.trace(B)
        bx = 0.5 * (B[0, 1] + B[1, 0])
        by = (B[1, 0] - B[0, 1]) / (2j)
        bz = 0.5 * (B[0, 0] - B[1, 1])
        Bnorm = float(np.linalg.norm(B))
        trace_fraction = (
            float(abs(np.trace(B)) / (math.sqrt(2.0) * Bnorm)) if Bnorm > 0 else math.nan
        )
        alignment = (
            float(abs(np.vdot(N, B)) / (Nnorm * Bnorm)) if Nnorm > 0 and Bnorm > 0 else math.nan
        )
        max_residual = max(float(minus["max_residual"]), float(plus["max_residual"]))
        biorth_error = max(float(minus["biorth_error"]), float(plus["biorth_error"]))
        support = min(float(minus["matrix_support_abs"]), float(plus["matrix_support_abs"]))
        reliable = (
            max_residual <= args.max_residual
            and biorth_error <= args.max_biorth_error
            and support >= args.min_support
        )
        survives = (
            reliable
            and Bnorm > 0.0
            and alignment >= args.min_alignment
            and trace_fraction <= args.max_trace_fraction
        )

        base = {
            "run": str(run_dir.relative_to(root)),
            "iteration": iteration,
            "charge": int(charge),
            "ground_charge": int(plus["ground_charge"]),
            "rank_minus": int(minus["matrix_rank"]),
            "rank_plus": int(plus["matrix_rank"]),
            "beta0": params.beta0,
            "U": params.U,
            "eps_d": params.eps_d,
            "Delta_eff": params.delta_eff,
            "Gamma_PT": params.gamma_pt,
            "Delta_coh": params.delta_coh,
            "V": params.V,
            "soc_lambda": params.soc_lambda,
            "F_lambda": params.F_lambda,
            "pair_gap_abs": float(abs(dz)),
            "support_min": support,
            "A_frobenius": float(np.linalg.norm(A)),
            "B_frobenius": Bnorm,
            "trace_fraction": trace_fraction,
            "jordan_alignment": alignment,
            "max_residual": max_residual,
            "biorth_error": biorth_error,
            "reliable": bool(reliable),
            "jordan_survives": bool(survives),
        }
        base.update(ccols("z_minus", zminus))
        base.update(ccols("z_plus", zplus))
        base.update(ccols("delta_z", dz))
        base.update(ccols("B_trace_half", b0))
        base.update(ccols("B_x", bx))
        base.update(ccols("B_y", by))
        base.update(ccols("B_z", bz))
        for idx, key in enumerate(MATRIX_KEYS):
            base.update(ccols(f"B_{key}", B.flat[idx]))
        rows.append(base)
    return rows


def plot_metric(df: pd.DataFrame, out: Path, metric: str, ylabel: str, logy: bool = False) -> None:
    if plt is None or df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    # Plot medians to avoid an unreadable line for every charge/run.
    grouped = df.groupby(["iteration"], as_index=False)[metric].median()
    ax.plot(grouped["iteration"], grouped[metric], marker="o")
    ax.set_xlabel("NRG iteration")
    ax.set_ylabel(ylabel)
    if logy:
        positive = grouped[metric] > 0
        if positive.any():
            ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=180)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def write_report(df: pd.DataFrame, out_dir: Path, args: argparse.Namespace) -> None:
    lines = [
        "# Jordan-channel survival audit",
        "",
        "This audit uses the full 2x2 transition-residue matrices. It does not use the channel trace alone and does not impose the quartic pole equation.",
        "",
        "A selected pole pair is classified as a resolved Jordan-active pair when:",
        f"- max residual <= `{args.max_residual:.3g}`",
        f"- biorthogonality error <= `{args.max_biorth_error:.3g}`",
        f"- minimum matrix support >= `{args.min_support:.3g}`",
        f"- alignment with `Delta_eff sigma_z + i Gamma_PT sigma_x` >= `{args.min_alignment:.3g}`",
        f"- normalized trace fraction <= `{args.max_trace_fraction:.3g}`",
        "",
    ]
    if df.empty:
        lines += ["No valid pole pairs were found.", ""]
    else:
        resolved = df[df["jordan_survives"]]
        lines += [
            f"- selected pairs: **{len(df)}**",
            f"- resolved Jordan-active pairs: **{len(resolved)}**",
            f"- runs represented: **{df['run'].nunique()}**",
            "",
            "## Iteration summary",
            "",
            "| iteration | pairs | resolved | median gap | median |B| | median trace fraction | median alignment |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for iteration, group in df.groupby("iteration", sort=True):
            lines.append(
                f"| {int(iteration)} | {len(group)} | {int(group['jordan_survives'].sum())} | "
                f"{group['pair_gap_abs'].median():.6g} | {group['B_frobenius'].median():.6g} | "
                f"{group['trace_fraction'].median():.6g} | {group['jordan_alignment'].median():.6g} |"
            )
        lines += ["", "## Interpretation", ""]
        if not resolved.empty:
            best = resolved.sort_values(
                ["iteration", "jordan_alignment", "trace_fraction"],
                ascending=[True, False, True],
            ).head(12)
            lines += [
                "The full matrix residue survives in the rows below even when its trace is strongly suppressed:",
                "",
                "| run | n | q | U | lambda | gap | |B| | trace fraction | alignment |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
            for _, row in best.iterrows():
                lines.append(
                    f"| `{row['run']}` | {int(row['iteration'])} | {int(row['charge'])} | "
                    f"{row['U']:.6g} | {row['soc_lambda']:.6g} | {row['pair_gap_abs']:.6g} | "
                    f"{row['B_frobenius']:.6g} | {row['trace_fraction']:.6g} | {row['jordan_alignment']:.6g} |"
                )
            lines += [""]
        lines += [
            "A finite, nearly traceless B aligned with the nilpotent EP direction demonstrates survival of the Jordan matrix residue. It does not by itself establish the quartic splitting law; that requires the pole gap to obey the predicted U and F scaling under a controlled branch-tracked scan.",
            "",
        ]
    (out_dir / "JORDAN_CHANNEL_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iterations", nargs="*", type=int)
    parser.add_argument("--pair-pool", type=int, default=8)
    parser.add_argument("--support-fraction", type=float, default=0.05)
    parser.add_argument("--max-residual", type=float, default=1e-8)
    parser.add_argument("--max-biorth-error", type=float, default=1e-6)
    parser.add_argument("--min-support", type=float, default=1e-4)
    parser.add_argument("--min-alignment", type=float, default=0.95)
    parser.add_argument("--max-trace-fraction", type=float, default=0.10)
    args = parser.parse_args()

    root = args.output_root.resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted(
        p.parent for p in root.rglob("impurity_transition_residue_matrix.csv")
        if (p.parent / "RUN_SUMMARY.txt").exists()
    )
    rows: list[dict] = []
    skipped: list[str] = []
    for run_dir in run_dirs:
        try:
            rows.extend(audit_run(run_dir, root, args))
        except Exception as exc:
            skipped.append(f"{run_dir}: {exc}")

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["run", "iteration", "charge"]).reset_index(drop=True)
    df.to_csv(out_dir / "jordan_channel_pairs.csv", index=False)
    write_report(df, out_dir, args)
    plot_metric(df, out_dir / "Jordan_B_vs_iteration", "B_frobenius", r"median $\|B\|_F$", logy=True)
    plot_metric(df, out_dir / "Jordan_alignment_vs_iteration", "jordan_alignment", "median Jordan alignment")
    plot_metric(df, out_dir / "Jordan_trace_fraction_vs_iteration", "trace_fraction", "median normalized trace fraction", logy=True)
    if skipped:
        (out_dir / "SKIPPED_RUNS.txt").write_text("\n".join(skipped) + "\n", encoding="utf-8")
    print(f"Wrote {len(df)} selected pairs from {len(run_dirs)} run directories to {out_dir}")
    if not df.empty:
        print(f"Resolved Jordan-active pairs: {int(df['jordan_survives'].sum())}")


if __name__ == "__main__":
    main()
