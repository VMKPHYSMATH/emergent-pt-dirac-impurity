#!/usr/bin/env python3
"""Non-circular pole-pair, quartic-residual, and Jordan-residue audit.

Reads NH-NRG run directories containing:
  RUN_SUMMARY.txt
  impurity_transition_residue_matrix.csv

The script does not alter the Hamiltonian and does not insert s_eff.  It tests
whether a selected impurity-supported pole pair is compatible with a supplied
quartic hypothesis

    s_NRG^4 - s0^2 s_NRG^2 = U^2 b_map^2 F,

where s_NRG=(z_+-z_-)/2 and
s0^2=Delta_eff^2 + (Delta_coh + i Gamma_PT)^2 in the adapter convention.

It also forms the pairwise Jordan coefficient matrix

    B_pair = 0.5 * (W_+ - W_-) * (z_+ - z_-).

A finite B_pair as the pole gap closes is evidence for a double-pole limit.
"""
from __future__ import annotations

import argparse
import itertools
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    plt = None
    _MPL_ERROR = exc
else:
    _MPL_ERROR = None


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
    soc_lambda: float = 0.0
    soc_kmax: float = math.nan
    F_lambda: float = 1.0
    soc_normalized_hybridization_det: float = 1.0


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
        beta0=vals["beta0"], U=vals["U"], eps_d=vals["eps_d"],
        delta_eff=vals["Delta_eff"], gamma_pt=vals["Gamma_PT"],
        delta_coh=vals["Delta_coh"], V=vals["V"],
        soc_lambda=vals.get("soc_lambda", 0.0),
        soc_kmax=vals.get("soc_kmax", math.nan),
        F_lambda=vals.get("F_lambda", 1.0),
        soc_normalized_hybridization_det=vals.get("soc_normalized_hybridization_det", 1.0),
    )


def matrix_from_row(row: pd.Series) -> np.ndarray:
    prefix = "add" if int(row["charge"]) == int(row["ground_charge"]) + 1 else "rem"
    vals = []
    for key in MATRIX_KEYS:
        vals.append(complex(float(row[f"{prefix}{key}_real"]), float(row[f"{prefix}{key}_imag"])))
    return np.asarray(vals, dtype=np.complex128).reshape(2, 2)


def select_pair(group: pd.DataFrame, pool: int, support_fraction: float) -> tuple[pd.Series, pd.Series] | None:
    if len(group) < 2:
        return None
    g = group.sort_values(["matrix_support_abs", "matrix_rank"], ascending=[False, True]).head(pool).copy()
    max_support = float(g["matrix_support_abs"].max())
    g = g[g["matrix_support_abs"] >= support_fraction * max_support]
    if len(g) < 2:
        return None

    best = None
    rows = [row for _, row in g.iterrows()]
    for a, b in itertools.combinations(rows, 2):
        za = complex(float(a["pole_real"]), float(a["pole_imag"]))
        zb = complex(float(b["pole_real"]), float(b["pole_imag"]))
        distance = abs(za - zb)
        support = min(float(a["matrix_support_abs"]), float(b["matrix_support_abs"]))
        key = (distance, -support, int(a["matrix_rank"]) + int(b["matrix_rank"]))
        if best is None or key < best[0]:
            best = (key, a, b)
    return None if best is None else (best[1], best[2])


def mapped_b(params: RunParameters, mode: str, scale: float, explicit: float | None) -> float:
    if mode == "hybridization":
        return abs(params.V)
    if mode == "beta0_scaled":
        return abs(params.beta0 * scale)
    if mode == "gamma_pt":
        return abs(params.gamma_pt)
    if mode == "explicit":
        if explicit is None or not np.isfinite(explicit):
            raise ValueError("--b-value is required with --b-mode explicit")
        return abs(float(explicit))
    if mode == "none":
        return math.nan
    raise ValueError(f"unknown b mode {mode}")


def complex_columns(prefix: str, value: complex) -> dict[str, float]:
    return {f"{prefix}_real": float(np.real(value)), f"{prefix}_imag": float(np.imag(value)), f"{prefix}_abs": float(abs(value))}


def audit_run(run_dir: Path, args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    params = parse_summary(run_dir / "RUN_SUMMARY.txt")
    df = pd.read_csv(run_dir / "impurity_transition_residue_matrix.csv")
    pair_rows: list[dict] = []
    component_rows: list[dict] = []
    run_name = str(run_dir.relative_to(args.output_root))
    bmap = mapped_b(params, args.b_mode, args.b_scale, args.b_value)
    F_run = params.soc_normalized_hybridization_det if args.F is None else args.F
    target = params.U**2 * bmap**2 * F_run if np.isfinite(bmap) else math.nan
    s0_sq = params.delta_eff**2 + complex(params.delta_coh, params.gamma_pt)**2

    for (iteration, charge), group in df.groupby(["iteration", "charge"], sort=True):
        if args.iteration is not None and int(iteration) != args.iteration:
            continue
        selected = select_pair(group, args.pair_pool, args.support_fraction)
        if selected is None:
            continue
        r1, r2 = selected
        z1 = complex(float(r1["pole_real"]), float(r1["pole_imag"]))
        z2 = complex(float(r2["pole_real"]), float(r2["pole_imag"]))
        if (z1.real, z1.imag) <= (z2.real, z2.imag):
            minus, plus = r1, r2
            zminus, zplus = z1, z2
        else:
            minus, plus = r2, r1
            zminus, zplus = z2, z1

        Wminus = matrix_from_row(minus)
        Wplus = matrix_from_row(plus)
        dz = zplus - zminus
        s_nrg = 0.5 * dz
        lhs = s_nrg**4 - s0_sq * s_nrg**2
        if np.isfinite(target) and target != 0.0:
            ratio = lhs / target
            rel_error = abs(lhs - target) / abs(target)
        else:
            ratio = complex(math.nan, math.nan)
            rel_error = math.nan
        A = Wplus + Wminus
        B = 0.5 * (Wplus - Wminus) * dz

        base = {
            "run": run_name,
            "iteration": int(iteration),
            "charge": int(charge),
            "ground_charge": int(plus["ground_charge"]),
            "rank_minus": int(minus["matrix_rank"]),
            "rank_plus": int(plus["matrix_rank"]),
            "support_minus": float(minus["matrix_support_abs"]),
            "support_plus": float(plus["matrix_support_abs"]),
            "beta0": params.beta0,
            "U": params.U,
            "eps_d": params.eps_d,
            "Delta_eff": params.delta_eff,
            "Gamma_PT": params.gamma_pt,
            "Delta_coh": params.delta_coh,
            "V": params.V,
            "soc_lambda": params.soc_lambda,
            "soc_kmax": params.soc_kmax,
            "F_lambda": params.F_lambda,
            "soc_normalized_hybridization_det": params.soc_normalized_hybridization_det,
            "F_assumed": F_run,
            "b_mode": args.b_mode,
            "b_mapped": bmap,
            "quartic_target": target,
            "quartic_relative_error": rel_error,
            "B_frobenius": float(np.linalg.norm(B)),
            "A_frobenius": float(np.linalg.norm(A)),
            "pair_gap_abs": float(abs(dz)),
        }
        base.update(complex_columns("z_minus", zminus))
        base.update(complex_columns("z_plus", zplus))
        base.update(complex_columns("delta_z", dz))
        base.update(complex_columns("s_nrg", s_nrg))
        base.update(complex_columns("s0_sq", s0_sq))
        base.update(complex_columns("quartic_lhs", lhs))
        base.update(complex_columns("quartic_ratio", ratio))
        pair_rows.append(base)

        for idx, key in enumerate(MATRIX_KEYS):
            comp = dict(base)
            comp["component"] = key
            comp.update(complex_columns("W_minus", Wminus.flat[idx]))
            comp.update(complex_columns("W_plus", Wplus.flat[idx]))
            comp.update(complex_columns("A_pair", A.flat[idx]))
            comp.update(complex_columns("B_pair", B.flat[idx]))
            component_rows.append(comp)

    return pair_rows, component_rows


def write_report(pair_df: pd.DataFrame, out: Path, args: argparse.Namespace, skipped: list[str]) -> None:
    lines = [
        "# Quartic pole and Jordan-residue audit",
        "",
        "## Scope",
        "",
        "This audit does not modify the NH-NRG Hamiltonian and does not insert the quartic scale by hand.",
        "It selects impurity-supported pole pairs from the full 2x2 transition-residue matrices and evaluates",
        "the quartic left-hand side and pairwise Jordan coefficient.",
        "",
        f"- F source = `{'RUN_SUMMARY normalized hybridization determinant' if args.F is None else f'fixed {args.F:.12g}'}`",
        f"- b mapping = `{args.b_mode}`",
        f"- pair pool = `{args.pair_pool}`",
        f"- support fraction = `{args.support_fraction}`",
        f"- iteration filter = `{args.iteration if args.iteration is not None else 'all'}`",
        "",
        "## Interpretation",
        "",
        "A quartic comparison is meaningful only when the selected b mapping is the same physical amplitude",
        "used in the manuscript quartic. In soc_mode=overlap, F(lambda) is read from RUN_SUMMARY and is",
        "the determinant of the normalized matrix hybridization. This tests a minimal projected SOC-overlap",
        "embedding; it is not a full momentum-resolved block-Lanczos representation of the chiral dispersion.",
        "",
    ]
    if pair_df.empty:
        lines += ["No valid pole pairs were found.", ""]
    else:
        iter0 = pair_df[pair_df["iteration"] == (args.iteration if args.iteration is not None else 0)]
        sample = iter0 if not iter0.empty else pair_df
        finite = sample[np.isfinite(sample["quartic_relative_error"])]
        lines += [
            f"- selected pole pairs: **{len(pair_df)}**",
            f"- runs represented: **{pair_df['run'].nunique()}**",
            f"- median selected gap: **{sample['pair_gap_abs'].median():.8g}**",
            f"- median Jordan |B| Frobenius norm: **{sample['B_frobenius'].median():.8g}**",
        ]
        if not finite.empty:
            lines += [
                f"- median quartic relative error: **{finite['quartic_relative_error'].median():.8g}**",
                f"- minimum quartic relative error: **{finite['quartic_relative_error'].min():.8g}**",
            ]
        lines += ["", "No acceptance/rejection label is assigned automatically; inspect convergence across detuning, U, Lambda, and nkeep.", ""]
    if skipped:
        lines += ["## Skipped directories", ""] + [f"- `{x}`" for x in skipped] + [""]
    (out / "QUARTIC_POLE_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")


def make_plots(pair_df: pd.DataFrame, out: Path) -> None:
    if plt is None or pair_df.empty:
        return
    d0 = pair_df[pair_df["iteration"] == 0].copy()
    if d0.empty:
        d0 = pair_df.copy()

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for run, g in d0.groupby("run"):
        ax.scatter(g["U"], g["pair_gap_abs"], label=run, s=28)
    ax.set_xlabel("U")
    ax.set_ylabel(r"selected pole gap $|z_+-z_-|$")
    ax.set_yscale("log")
    if d0["run"].nunique() <= 10:
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "Quartic_Pole_Gap_vs_U.pdf")
    fig.savefig(out / "Quartic_Pole_Gap_vs_U.png", dpi=180)
    plt.close(fig)

    finite = d0[np.isfinite(d0["quartic_relative_error"])].copy()
    if not finite.empty:
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        ax.scatter(finite["U"], finite["quartic_relative_error"], s=28)
        ax.set_xlabel("U")
        ax.set_ylabel("quartic relative error")
        ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(out / "Quartic_Residual_vs_U.pdf")
        fig.savefig(out / "Quartic_Residual_vs_U.png", dpi=180)
        plt.close(fig)

    positive = d0[(d0["Delta_coh"] > 0) & (d0["B_frobenius"] > 0)].copy()
    if not positive.empty:
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        ax.scatter(positive["Delta_coh"], positive["B_frobenius"], s=28, label=r"$\|B_{pair}\|_F$")
        ax.scatter(positive["Delta_coh"], positive["pair_gap_abs"], s=28, marker="x", label="pole gap")
        ax.set_xlabel(r"$\delta_{coh}$")
        ax.set_ylabel("magnitude")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "Jordan_Residue_and_Gap_vs_Detuning.pdf")
        fig.savefig(out / "Jordan_Residue_and_Gap_vs_Detuning.png", dpi=180)
        plt.close(fig)

    soc = d0[np.isfinite(d0.get("F_lambda", np.nan))].copy()
    main_soc = soc[soc["run"].str.contains("lambda_scan/", regex=False)]
    if not main_soc.empty:
        soc = main_soc
    if not soc.empty and soc["soc_lambda"].nunique() > 1:
        # Avoid charge-conjugate duplicate points in summary plots.
        soc = soc.sort_values(["run", "charge"]).groupby("run", as_index=False).first()
        soc = soc.sort_values("F_lambda")
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        ax.plot(soc["F_lambda"], soc["pair_gap_abs"], "o-", label="pole gap")
        ax.plot(soc["F_lambda"], soc["B_frobenius"], "s-", label=r"$\|B_{pair}\|_F$")
        ax.set_xlabel(r"$F(\lambda)=1-(\lambda/k_{max})^2$")
        ax.set_ylabel("magnitude")
        ax.set_yscale("log")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "SOC_Gap_and_Jordan_vs_F.pdf")
        fig.savefig(out / "SOC_Gap_and_Jordan_vs_F.png", dpi=180)
        plt.close(fig)

        finite_soc = soc[np.isfinite(soc["quartic_relative_error"])]
        if not finite_soc.empty:
            fig, ax = plt.subplots(figsize=(6.2, 4.2))
            ax.plot(finite_soc["F_lambda"], finite_soc["quartic_relative_error"], "o-")
            ax.set_xlabel(r"$F(\lambda)$")
            ax.set_ylabel("quartic relative error")
            ax.set_yscale("log")
            fig.tight_layout()
            fig.savefig(out / "SOC_Quartic_Residual_vs_F.pdf")
            fig.savefig(out / "SOC_Quartic_Residual_vs_F.png", dpi=180)
            plt.close(fig)


def _parse_F(value: str) -> float | None:
    if value.lower() == "auto":
        return None
    return float(value)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output_root", type=Path, help="root containing NH-NRG run directories")
    p.add_argument("--out", type=Path, default=Path("quartic_pole_audit"))
    p.add_argument("--F", type=_parse_F, default=None,
                   help="SOC form factor, or 'auto' to read F_lambda from RUN_SUMMARY (default)")
    p.add_argument("--b-mode", choices=("hybridization", "beta0_scaled", "gamma_pt", "explicit", "none"), default="hybridization")
    p.add_argument("--b-scale", type=float, default=1.0, help="energy scale for beta0_scaled mapping")
    p.add_argument("--b-value", type=float, default=None, help="explicit quartic amplitude")
    p.add_argument("--iteration", type=int, default=None, help="analyze one Wilson iteration only")
    p.add_argument("--pair-pool", type=int, default=8)
    p.add_argument("--support-fraction", type=float, default=0.05)
    return p


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_root = args.output_root.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.output_root.exists():
        raise FileNotFoundError(args.output_root)

    run_dirs = sorted({p.parent for p in args.output_root.rglob("impurity_transition_residue_matrix.csv")})
    pair_rows: list[dict] = []
    component_rows: list[dict] = []
    skipped: list[str] = []
    for run_dir in run_dirs:
        try:
            p_rows, c_rows = audit_run(run_dir, args)
            pair_rows.extend(p_rows)
            component_rows.extend(c_rows)
        except Exception as exc:
            skipped.append(f"{run_dir}: {exc}")

    pair_df = pd.DataFrame(pair_rows)
    comp_df = pd.DataFrame(component_rows)
    pair_df.to_csv(args.out / "quartic_pole_pairs.csv", index=False)
    comp_df.to_csv(args.out / "jordan_pair_components.csv", index=False)
    write_report(pair_df, args.out, args, skipped)
    make_plots(pair_df, args.out)

    print(f"runs found: {len(run_dirs)}")
    print(f"pole pairs written: {len(pair_df)}")
    print(f"output: {args.out}")
    if plt is None:
        print(f"plots skipped: matplotlib unavailable ({_MPL_ERROR})", file=sys.stderr)
    return 0 if len(pair_df) else 2


if __name__ == "__main__":
    raise SystemExit(main())
