#!/usr/bin/env python3
"""Extract operational low-energy scales from a PTDirac NH-NRG beta0 scan.

The script deliberately distinguishes four diagnostics:

* T_flow: entry scale of the same-parity NRG level-flow fixed-point regime.
* T_pair_n: half splitting of a full-matrix, impurity-supported pole pair at
  fixed iteration n, branch tracked continuously in beta0.
* T_trace_HWHM: HWHM of the kept-space trace spectral peak, only when the
  sum-rule and positivity gates pass.
* T_J_HWHM: HWHM of the absolute Jordan-projected kept-space spectrum, again
  gated by the kept-space sum rule.

None of these is silently renamed a thermodynamic Kondo temperature.  A true
thermodynamic T_K requires finite-temperature impurity thermodynamics or an
explicitly justified universal susceptibility/entropy criterion.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from common import load_toml, parse_run_summary

MATRIX_KEYS = ("11", "12", "21", "22")


@dataclass
class PairCandidate:
    beta0: float
    iteration: int
    sector: int
    zminus: complex
    zplus: complex
    Wminus: np.ndarray
    Wplus: np.ndarray
    support: float
    max_residual: float
    biorth_error: float
    alignment: float
    trace_fraction: float
    mismatch: float

    @property
    def center(self) -> complex:
        return 0.5 * (self.zplus + self.zminus)

    @property
    def split(self) -> complex:
        return 0.5 * (self.zplus - self.zminus)

    @property
    def B(self) -> np.ndarray:
        return (self.Wplus - self.Wminus) * self.split

    @property
    def Bnorm(self) -> float:
        return float(np.linalg.norm(self.B))


@dataclass
class RunData:
    run_dir: Path
    summary: dict[str, Any]
    flow: pd.DataFrame
    residues: pd.DataFrame
    weights: pd.DataFrame | None
    lehmann: pd.DataFrame | None
    sumrule: pd.DataFrame | None

    @property
    def beta0(self) -> float:
        return float(self.summary["beta0"])

    @property
    def delta_eff(self) -> float:
        return float(self.summary["Delta_eff"])

    @property
    def gamma_pt(self) -> float:
        return float(self.summary["Gamma_PT"])


def read_optional(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def find_runs(root: Path) -> list[RunData]:
    run_dirs = sorted(
        p.parent for p in root.rglob("RUN_SUMMARY.txt")
        if (p.parent / "complex_level_flow.csv").exists()
        and (p.parent / "impurity_transition_residue_matrix.csv").exists()
    )
    runs: list[RunData] = []
    for run_dir in run_dirs:
        summary = parse_run_summary(run_dir / "RUN_SUMMARY.txt")
        if "beta0" not in summary:
            continue
        runs.append(RunData(
            run_dir=run_dir,
            summary=summary,
            flow=pd.read_csv(run_dir / "complex_level_flow.csv"),
            residues=pd.read_csv(run_dir / "impurity_transition_residue_matrix.csv"),
            weights=read_optional(run_dir / "impurity_transition_weights.csv"),
            lehmann=read_optional(run_dir / "kept_space_lehmann.csv"),
            sumrule=read_optional(run_dir / "lehmann_sumrule.csv"),
        ))
    return sorted(runs, key=lambda r: r.beta0)


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        x = float(value)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def beta_ep(run: RunData) -> float:
    d0 = safe_float(run.summary.get("Delta_eff")) - safe_float(run.summary.get("beta0")) * 0.0
    # Prefer reconstructing the exact affine coefficients from multiple runs in
    # caller; this fallback returns the nominal current-run proximity only.
    return math.nan if not np.isfinite(d0) else run.beta0


def infer_beta_ep(runs: list[RunData]) -> float:
    if len(runs) < 2:
        return math.nan
    b = np.array([r.beta0 for r in runs], dtype=float)
    d = np.array([r.delta_eff for r in runs], dtype=float)
    g = np.array([r.gamma_pt for r in runs], dtype=float)
    A = np.column_stack([np.ones_like(b), b])
    cd, *_ = np.linalg.lstsq(A, d, rcond=None)
    cg, *_ = np.linalg.lstsq(A, g, rcond=None)
    denom = cg[1] - cd[1]
    if abs(denom) < 1e-14:
        return math.nan
    return float((cd[0] - cg[0]) / denom)


def iteration_reliability(group: pd.DataFrame, max_residual: float,
                          max_biorth: float) -> bool:
    return (
        float(group["max_residual"].max()) <= max_residual
        and float(group["biorth_error"].max()) <= max_biorth
    )


def fingerprint(group: pd.DataFrame, levels_per_charge: int) -> dict[tuple[int, int], complex]:
    q0 = int(round(float(group["ground_charge"].iloc[0])))
    selected = group[
        group["charge"].isin([q0 - 1, q0, q0 + 1])
        & (group["ordinal"] < levels_per_charge)
    ]
    out: dict[tuple[int, int], complex] = {}
    for _, row in selected.iterrows():
        label = (int(row["charge"]) - q0, int(row["ordinal"]))
        out[label] = complex(float(row["energy_real"]), float(row["energy_imag"]))
    return out


def fingerprint_distance(a: dict[tuple[int, int], complex],
                         b: dict[tuple[int, int], complex]) -> float:
    labels = sorted(set(a).intersection(b))
    if len(labels) < 3:
        return math.nan
    av = np.asarray([a[x] for x in labels], dtype=np.complex128)
    bv = np.asarray([b[x] for x in labels], dtype=np.complex128)
    scale = max(1.0, float(np.sqrt(np.mean(np.abs(av) ** 2))),
                float(np.sqrt(np.mean(np.abs(bv) ** 2))))
    return float(np.sqrt(np.mean(np.abs(av - bv) ** 2)) / scale)


def extract_flow_scale(run: RunData, cfg: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    levels = int(cfg.get("levels_per_charge", 4))
    min_iteration = int(cfg.get("min_iteration", 4))
    tol = float(cfg.get("distance_tolerance", 0.05))
    consecutive = int(cfg.get("consecutive_same_parity", 3))
    max_res = float(cfg.get("max_residual", 1e-7))
    max_bio = float(cfg.get("max_biorth_error", 1e-5))

    groups = {int(n): g.copy() for n, g in run.flow.groupby("iteration")}
    fps = {n: fingerprint(g, levels) for n, g in groups.items()}
    distances: dict[int, float] = {}
    reliable: dict[int, bool] = {}
    detail: list[dict[str, Any]] = []
    for n in sorted(groups):
        reliable[n] = iteration_reliability(groups[n], max_res, max_bio)
        d = fingerprint_distance(fps[n], fps[n - 2]) if n - 2 in fps else math.nan
        distances[n] = d
        detail.append({
            "beta0": run.beta0,
            "iteration": n,
            "scale": float(groups[n]["scale"].iloc[0]),
            "same_parity_distance": d,
            "reliable": reliable[n],
            "max_residual": float(groups[n]["max_residual"].max()),
            "biorth_error": float(groups[n]["biorth_error"].max()),
        })

    entry = None
    for parity in (0, 1):
        ns = [n for n in sorted(groups) if n >= min_iteration and n % 2 == parity]
        for pos in range(0, len(ns) - consecutive + 1):
            window = ns[pos:pos + consecutive]
            if all(
                reliable.get(n, False)
                and np.isfinite(distances.get(n, math.nan))
                and distances[n] <= tol
                for n in window
            ):
                candidate = window[0]
                if entry is None or candidate < entry:
                    entry = candidate
                break
    if entry is None:
        return {
            "T_flow": math.nan,
            "flow_entry_iteration": math.nan,
            "flow_distance_at_entry": math.nan,
            "flow_gate": False,
        }, detail
    scale = float(groups[entry]["scale"].iloc[0])
    return {
        "T_flow": scale,
        "flow_entry_iteration": entry,
        "flow_distance_at_entry": distances[entry],
        "flow_gate": True,
    }, detail


def matrix_from_row(row: pd.Series) -> np.ndarray:
    prefix = "add" if int(row["charge"]) == int(row["ground_charge"]) + 1 else "rem"
    values = [
        complex(float(row[f"{prefix}{key}_real"]), float(row[f"{prefix}{key}_imag"]))
        for key in MATRIX_KEYS
    ]
    return np.asarray(values, dtype=np.complex128).reshape(2, 2)


def pauli_metrics(B: np.ndarray, N: np.ndarray) -> tuple[float, float, float]:
    Bnorm = float(np.linalg.norm(B))
    Nnorm = float(np.linalg.norm(N))
    alignment = float(abs(np.vdot(N, B)) / (Nnorm * Bnorm)) if Nnorm > 0 and Bnorm > 0 else math.nan
    trace_fraction = float(abs(np.trace(B)) / (math.sqrt(2) * Bnorm)) if Bnorm > 0 else math.nan
    bx = 0.5 * (B[0, 1] + B[1, 0])
    bz = 0.5 * (B[0, 0] - B[1, 1])
    mismatch = float(abs(bx - 1j * bz) / (abs(bx) + abs(bz))) if abs(bx) + abs(bz) > 0 else math.nan
    return alignment, trace_fraction, mismatch


def candidate_pairs(run: RunData, iteration: int, sector: int,
                    cfg: dict[str, Any]) -> list[PairCandidate]:
    q0_groups = run.residues[run.residues["iteration"] == iteration]
    if q0_groups.empty:
        return []
    q0 = int(round(float(q0_groups["ground_charge"].iloc[0])))
    charge = q0 + sector
    group = q0_groups[q0_groups["charge"] == charge].copy()
    if len(group) < 2:
        return []
    pool = int(cfg.get("pair_pool", 8))
    support_fraction = float(cfg.get("support_fraction", 0.05))
    group = group.sort_values(["matrix_support_abs", "matrix_rank"], ascending=[False, True]).head(pool)
    maximum = float(group["matrix_support_abs"].max())
    group = group[group["matrix_support_abs"] >= support_fraction * maximum]
    if len(group) < 2:
        return []

    N = np.asarray([
        [run.delta_eff, 1j * run.gamma_pt],
        [1j * run.gamma_pt, -run.delta_eff],
    ], dtype=np.complex128)
    out: list[PairCandidate] = []
    rows = [row for _, row in group.iterrows()]
    for first, second in itertools.combinations(rows, 2):
        z1 = complex(float(first["pole_real"]), float(first["pole_imag"]))
        z2 = complex(float(second["pole_real"]), float(second["pole_imag"]))
        if (z1.real, z1.imag) <= (z2.real, z2.imag):
            minus, plus, zminus, zplus = first, second, z1, z2
        else:
            minus, plus, zminus, zplus = second, first, z2, z1
        Wminus = matrix_from_row(minus)
        Wplus = matrix_from_row(plus)
        B = 0.5 * (Wplus - Wminus) * (zplus - zminus)
        alignment, trace_fraction, mismatch = pauli_metrics(B, N)
        out.append(PairCandidate(
            beta0=run.beta0,
            iteration=iteration,
            sector=sector,
            zminus=zminus,
            zplus=zplus,
            Wminus=Wminus,
            Wplus=Wplus,
            support=min(float(minus["matrix_support_abs"]), float(plus["matrix_support_abs"])),
            max_residual=max(float(minus["max_residual"]), float(plus["max_residual"])),
            biorth_error=max(float(minus["biorth_error"]), float(plus["biorth_error"])),
            alignment=alignment,
            trace_fraction=trace_fraction,
            mismatch=mismatch,
        ))
    return out


def candidate_reliable(c: PairCandidate, cfg: dict[str, Any]) -> bool:
    return (
        c.support >= float(cfg.get("min_support", 1e-4))
        and c.max_residual <= float(cfg.get("max_residual", 1e-7))
        and c.biorth_error <= float(cfg.get("max_biorth_error", 1e-5))
    )


def seed_key(c: PairCandidate) -> tuple[float, float, float, float]:
    return (abs(c.split), -c.support, -safe_float(c.alignment, 0.0), c.Bnorm)


def tracking_cost(prev: PairCandidate, cand: PairCandidate) -> float:
    energy_scale = max(abs(prev.center), abs(cand.center), abs(prev.split), abs(cand.split), 1e-12)
    center = abs(cand.center - prev.center) / energy_scale
    split = abs(cand.split - prev.split) / energy_scale
    bscale = max(prev.Bnorm, cand.Bnorm, 1e-14)
    residue = float(np.linalg.norm(cand.B - prev.B) / bscale)
    support_penalty = abs(math.log(max(cand.support, 1e-300) / max(prev.support, 1e-300)))
    return float(center + split + 0.5 * residue + 0.05 * support_penalty)


def track_one(runs: list[RunData], iteration: int, sector: int,
              cfg: dict[str, Any], ep: float) -> dict[float, PairCandidate]:
    candidates = {r.beta0: candidate_pairs(r, iteration, sector, cfg) for r in runs}
    available = [b for b, cs in candidates.items() if cs]
    if not available:
        return {}
    seed_beta = min(available, key=lambda b: abs(b - ep) if np.isfinite(ep) else abs(b))
    reliable_seed = [c for c in candidates[seed_beta] if candidate_reliable(c, cfg)]
    seed_pool = reliable_seed or candidates[seed_beta]
    chosen: dict[float, PairCandidate] = {seed_beta: min(seed_pool, key=seed_key)}

    lower = sorted([b for b in available if b < seed_beta], reverse=True)
    higher = sorted([b for b in available if b > seed_beta])
    prev = chosen[seed_beta]
    for beta in lower:
        pool = [c for c in candidates[beta] if candidate_reliable(c, cfg)] or candidates[beta]
        prev = min(pool, key=lambda c: tracking_cost(prev, c))
        chosen[beta] = prev
    prev = chosen[seed_beta]
    for beta in higher:
        pool = [c for c in candidates[beta] if candidate_reliable(c, cfg)] or candidates[beta]
        prev = min(pool, key=lambda c: tracking_cost(prev, c))
        chosen[beta] = prev
    return chosen


def pair_table(runs: list[RunData], cfg: dict[str, Any], ep: float) -> pd.DataFrame:
    iterations = sorted(int(x) for x in cfg.get("reference_iterations", [4, 5, 6]))
    all_iterations = sorted({int(n) for r in runs for n in r.residues["iteration"].unique()})
    wanted = sorted(set(iterations).union(all_iterations))
    rows: list[dict[str, Any]] = []
    for iteration in wanted:
        for sector in (-1, 1):
            tracked = track_one(runs, iteration, sector, cfg, ep)
            for beta, c in tracked.items():
                rows.append({
                    "beta0": beta,
                    "iteration": iteration,
                    "sector": sector,
                    "zminus_real": c.zminus.real,
                    "zminus_imag": c.zminus.imag,
                    "zplus_real": c.zplus.real,
                    "zplus_imag": c.zplus.imag,
                    "T_pair_split": abs(c.split),
                    "T_pole_abs": 0.5 * (abs(c.zminus) + abs(c.zplus)),
                    "pole_center_abs": abs(c.center),
                    "B_frobenius": c.Bnorm,
                    "jordan_alignment": c.alignment,
                    "trace_fraction": c.trace_fraction,
                    "nilpotent_mismatch": c.mismatch,
                    "support_min": c.support,
                    "max_residual": c.max_residual,
                    "biorth_error": c.biorth_error,
                    "reliable": candidate_reliable(c, cfg),
                })
    columns = [
        "beta0", "iteration", "sector", "zminus_real", "zminus_imag",
        "zplus_real", "zplus_imag", "T_pair_split", "T_pole_abs",
        "pole_center_abs", "B_frobenius", "jordan_alignment",
        "trace_fraction", "nilpotent_mismatch", "support_min",
        "max_residual", "biorth_error", "reliable",
    ]
    return pd.DataFrame(rows, columns=columns)


def interp_crossing(x1: float, y1: float, x2: float, y2: float, target: float) -> float:
    if y2 == y1:
        return 0.5 * (x1 + x2)
    t = (target - y1) / (y2 - y1)
    return float(x1 + np.clip(t, 0.0, 1.0) * (x2 - x1))


def peak_hwhm(omega: np.ndarray, signal: np.ndarray, window: float) -> tuple[float, float, float]:
    finite = np.isfinite(omega) & np.isfinite(signal)
    xf = omega[finite]
    yf = signal[finite]
    if len(xf) < 9:
        return math.nan, math.nan, math.nan
    order = np.argsort(xf)
    xf, yf = xf[order], yf[order]
    edge_count = max(2, len(xf) // 20)
    baseline = float(np.median(np.r_[yf[:edge_count], yf[-edge_count:]]))
    mask = np.abs(xf) <= window
    x = xf[mask]
    y = yf[mask]
    if len(x) < 9:
        return math.nan, math.nan, math.nan
    peak = int(np.argmax(y))
    if peak <= 1 or peak >= len(x) - 2:
        return math.nan, float(x[peak]), float(y[peak])
    height = float(y[peak] - baseline)
    if not np.isfinite(height) or height <= 0:
        return math.nan, float(x[peak]), float(y[peak])
    half = baseline + 0.5 * height

    left = math.nan
    for i in range(peak - 1, -1, -1):
        if y[i] <= half <= y[i + 1] or y[i] >= half >= y[i + 1]:
            left = interp_crossing(float(x[i]), float(y[i]), float(x[i + 1]), float(y[i + 1]), half)
            break
    right = math.nan
    for i in range(peak, len(x) - 1):
        if y[i] >= half >= y[i + 1] or y[i] <= half <= y[i + 1]:
            right = interp_crossing(float(x[i]), float(y[i]), float(x[i + 1]), float(y[i + 1]), half)
            break
    if not np.isfinite(left) or not np.isfinite(right) or right <= left:
        return math.nan, float(x[peak]), float(y[peak])
    return 0.5 * (right - left), float(x[peak]), float(y[peak])


def spectral_scales(run: RunData, cfg: dict[str, Any]) -> dict[str, Any]:
    if run.lehmann is None:
        return {
            "T_trace_HWHM": math.nan, "T_J_HWHM": math.nan,
            "trace_spectral_gate": False, "jordan_spectral_gate": False,
            "sumrule_max_error": math.nan, "trace_positive_fraction": math.nan,
        }
    df = run.lehmann
    omega = df["omega"].to_numpy(dtype=float)
    trace = df["minus_ImTrG_over_pi"].to_numpy(dtype=float)
    window = float(cfg.get("zero_window", 0.30))
    wmask = np.abs(omega) <= window
    positivity = float(np.mean(trace[wmask] >= -1e-12)) if np.any(wmask) else math.nan
    sumerr = (
        float(run.sumrule["abs_error"].max())
        if run.sumrule is not None and "abs_error" in run.sumrule else math.nan
    )
    sum_gate = np.isfinite(sumerr) and sumerr <= float(cfg.get("max_sumrule_error", 0.15))
    trace_gate = sum_gate and positivity >= float(cfg.get("min_trace_positive_fraction", 0.90))
    t_trace, trace_peak_pos, trace_peak_height = peak_hwhm(omega, trace, window)

    if "minus_ImJordanG_over_pi" in df:
        aj = df["minus_ImJordanG_over_pi"].to_numpy(dtype=float)
    else:
        g11 = df["ReG11"].to_numpy(float) + 1j * df["ImG11"].to_numpy(float)
        g12 = df["ReG12"].to_numpy(float) + 1j * df["ImG12"].to_numpy(float)
        g21 = df["ReG21"].to_numpy(float) + 1j * df["ImG21"].to_numpy(float)
        g22 = df["ReG22"].to_numpy(float) + 1j * df["ImG22"].to_numpy(float)
        N = np.asarray([
            [run.delta_eff, 1j * run.gamma_pt],
            [1j * run.gamma_pt, -run.delta_eff],
        ], dtype=np.complex128)
        nrm = float(np.linalg.norm(N))
        vals = np.empty(len(df), dtype=np.complex128)
        for i in range(len(df)):
            G = np.asarray([[g11[i], g12[i]], [g21[i], g22[i]]], dtype=np.complex128)
            vals[i] = np.vdot(N, G) / nrm if nrm > 0 else 0.0
        aj = -np.imag(vals) / np.pi
    t_j, j_peak_pos, j_peak_height = peak_hwhm(omega, np.abs(aj), window)
    return {
        "T_trace_HWHM": t_trace if trace_gate else math.nan,
        "T_trace_HWHM_raw": t_trace,
        "trace_peak_position": trace_peak_pos,
        "trace_peak_height": trace_peak_height,
        "T_J_HWHM": t_j if sum_gate else math.nan,
        "T_J_HWHM_raw": t_j,
        "jordan_peak_position": j_peak_pos,
        "jordan_peak_height_abs": j_peak_height,
        "trace_spectral_gate": bool(trace_gate and np.isfinite(t_trace)),
        "jordan_spectral_gate": bool(sum_gate and np.isfinite(t_j)),
        "sumrule_max_error": sumerr,
        "trace_positive_fraction": positivity,
    }


def onset_scale(beta: float, pairs: pd.DataFrame, run: RunData,
                cfg: dict[str, Any]) -> tuple[float, float]:
    g = pairs[(pairs["beta0"] == beta) & pairs["reliable"]].copy()
    if g.empty:
        return math.nan, math.nan
    min_align = float(cfg.get("min_jordan_alignment", 0.95))
    max_trace = float(cfg.get("max_trace_fraction", 0.10))
    consecutive = int(cfg.get("onset_consecutive_same_parity", 2))
    summary = g.groupby("iteration").agg(
        alignment=("jordan_alignment", "median"),
        trace=("trace_fraction", "median"),
        mismatch=("nilpotent_mismatch", "median"),
    ).reset_index()
    good = {
        int(row.iteration): (
            row.alignment >= min_align and row.trace <= max_trace
            and np.isfinite(row.mismatch) and row.mismatch <= 0.10
        )
        for row in summary.itertuples()
    }
    entry = None
    for parity in (0, 1):
        ns = sorted(n for n in good if n % 2 == parity)
        for pos in range(0, len(ns) - consecutive + 1):
            window = ns[pos:pos + consecutive]
            if all(good[n] for n in window):
                candidate = window[0]
                if entry is None or candidate < entry:
                    entry = candidate
                break
    if entry is None:
        return math.nan, math.nan
    flow_group = run.flow[run.flow["iteration"] == entry]
    if flow_group.empty:
        return math.nan, float(entry)
    return float(flow_group["scale"].iloc[0]), float(entry)


def aggregate_summary(runs: list[RunData], pair_df: pd.DataFrame,
                      cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    flow_rows: list[dict[str, Any]] = []
    spectral_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    ref_iterations = sorted(int(x) for x in cfg["poles"].get("reference_iterations", [4, 5, 6]))
    for run in runs:
        flow, detail = extract_flow_scale(run, cfg["flow"])
        flow_rows.extend(detail)
        spec = spectral_scales(run, cfg["spectral"])
        spectral_rows.append({"beta0": run.beta0, **spec})
        row: dict[str, Any] = {
            "run": str(run.run_dir),
            "beta0": run.beta0,
            "U": safe_float(run.summary.get("U")),
            "Delta_eff": run.delta_eff,
            "Gamma_PT": run.gamma_pt,
            "Delta_coh": safe_float(run.summary.get("Delta_coh")),
            "ep_discriminant_abs": abs(run.delta_eff ** 2 - run.gamma_pt ** 2),
            **flow, **spec,
        }
        for n in ref_iterations:
            pg = pair_df[(pair_df["beta0"] == run.beta0) & (pair_df["iteration"] == n)]
            reliable = pg[pg["reliable"]]
            use = reliable if not reliable.empty else pg
            suffix = f"n{n}"
            if use.empty:
                for name in ("T_pair_split", "T_pole_abs", "jordan_alignment", "trace_fraction", "nilpotent_mismatch", "B_frobenius"):
                    row[f"{name}_{suffix}"] = math.nan
                row[f"pair_gate_{suffix}"] = False
            else:
                for name in ("T_pair_split", "T_pole_abs", "jordan_alignment", "trace_fraction", "nilpotent_mismatch", "B_frobenius"):
                    row[f"{name}_{suffix}"] = float(use[name].median())
                row[f"pair_gate_{suffix}"] = bool(not reliable.empty and len(reliable["sector"].unique()) == 2)
        tj, nj = onset_scale(run.beta0, pair_df, run, cfg["poles"])
        row["T_matrix_Jordan_onset"] = tj
        row["matrix_Jordan_onset_iteration"] = nj
        summaries.append(row)
    return pd.DataFrame(summaries).sort_values("beta0"), pd.DataFrame(flow_rows), pd.DataFrame(spectral_rows)


def plot_scales(summary: pd.DataFrame, out_dir: Path, ep: float,
                reference_beta: float) -> None:
    if plt is None or summary.empty:
        return
    candidates = [
        ("T_flow", r"$T_{\rm flow}$"),
        ("T_pair_split_n5", r"$T_{\rm pair}^{(n=5)}$"),
        ("T_trace_HWHM", r"$T_{\rm tr}^{\rm HWHM}$"),
        ("T_J_HWHM", r"$T_{J}^{\rm HWHM}$"),
        ("T_matrix_Jordan_onset", r"$T_{J,\rm onset}$"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    beta = summary["beta0"].to_numpy(float)
    plotted = 0
    for column, label in candidates:
        if column not in summary:
            continue
        values = summary[column].to_numpy(float)
        mask = np.isfinite(values) & (values > 0)
        if not np.any(mask):
            continue
        axes[0].plot(beta[mask], values[mask], marker="o", label=label)
        ref_index = int(np.nanargmin(np.abs(beta - reference_beta)))
        ref = values[ref_index]
        if not np.isfinite(ref) or ref <= 0:
            valid = np.flatnonzero(mask)
            ref = values[valid[0]] if len(valid) else math.nan
        if np.isfinite(ref) and ref > 0:
            axes[1].plot(beta[mask], values[mask] / ref, marker="o", label=label)
        plotted += 1
    for ax in axes:
        if np.isfinite(ep):
            ax.axvline(ep, linestyle="--", linewidth=1.1, label=r"nominal $\beta_{\rm EP}$")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel(r"$\beta_0$")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("operational energy scale")
    axes[1].set_ylabel(r"scale / reference value")
    if plotted:
        handles, labels = axes[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axes[0].legend(by_label.values(), by_label.keys(), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "Operational_scales_vs_beta0.pdf")
    fig.savefig(out_dir / "Operational_scales_vs_beta0.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    if "jordan_alignment_n5" in summary:
        axes[0].plot(beta, 1.0 - summary["jordan_alignment_n5"], marker="o", label=r"$1-\mathcal{A}_J$")
        axes[0].plot(beta, summary["trace_fraction_n5"], marker="s", label="trace fraction")
        axes[0].plot(beta, summary["nilpotent_mismatch_n5"], marker="^", label="nilpotent mismatch")
        axes[0].set_yscale("log")
        axes[0].legend(fontsize=8)
    axes[0].set_xlabel(r"$\beta_0$")
    axes[0].set_ylabel("matrix diagnostic")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(beta, summary["sumrule_max_error"], marker="o", label="Lehmann sum-rule error")
    axes[1].plot(beta, summary["trace_positive_fraction"], marker="s", label="trace positivity fraction")
    axes[1].set_xlabel(r"$\beta_0$")
    axes[1].set_ylabel("spectral quality diagnostic")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)
    for ax in axes:
        if np.isfinite(ep):
            ax.axvline(ep, linestyle="--", linewidth=1.1)
    fig.tight_layout()
    fig.savefig(out_dir / "Scale_extraction_quality_vs_beta0.pdf")
    fig.savefig(out_dir / "Scale_extraction_quality_vs_beta0.png", dpi=220)
    plt.close(fig)


def write_report(summary: pd.DataFrame, out_dir: Path, ep: float) -> None:
    lines = [
        "# Operational low-energy scales from the beta0 NH-NRG scan",
        "",
        "The extraction is deliberately definition-resolved. No output column is silently identified with a thermodynamic Kondo temperature.",
        "",
        "## Definitions",
        "",
        "- `T_flow`: Wilson scale at the earliest same-parity fixed-point-entry window of the complex level flow.",
        "- `T_pair_split_n*`: half splitting of a full 2x2 impurity-supported pole pair at fixed iteration, branch tracked continuously in beta0.",
        "- `T_trace_HWHM`: kept-space trace-spectral HWHM, emitted only when the sum-rule and positivity gates pass.",
        "- `T_J_HWHM`: kept-space Jordan-projected HWHM, emitted only when the sum-rule gate passes.",
        "- `T_matrix_Jordan_onset`: Wilson scale where the branch-tracked residue first passes the matrix-Jordan gates for consecutive same-parity iterations.",
        "",
        f"Nominal compensated point inferred from the scan: `beta_EP = {ep:.9g}`" if np.isfinite(ep) else "The nominal compensated point could not be inferred.",
        "",
        "## Per-run summary",
        "",
        "| beta0 | T_flow | T_pair(n=5) | T_trace HWHM | T_J HWHM | T_J onset | pair gate | spectral gates |",
        "|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for _, row in summary.iterrows():
        def f(name: str) -> str:
            x = safe_float(row.get(name))
            return f"{x:.6g}" if np.isfinite(x) else "--"
        lines.append(
            f"| {row['beta0']:.6g} | {f('T_flow')} | {f('T_pair_split_n5')} | "
            f"{f('T_trace_HWHM')} | {f('T_J_HWHM')} | {f('T_matrix_Jordan_onset')} | "
            f"{'pass' if bool(row.get('pair_gate_n5', False)) else 'fail'} | "
            f"tr={'pass' if bool(row.get('trace_spectral_gate', False)) else 'fail'}, "
            f"J={'pass' if bool(row.get('jordan_spectral_gate', False)) else 'fail'} |"
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "These curves may be plotted together as operational crossover scales. A thermodynamic `T_K` label is reserved for a separately implemented finite-temperature susceptibility, entropy, or another explicitly justified universal criterion. The kept-space Lehmann widths are diagnostics rather than complete-basis/FDM-NRG spectra.",
        "",
    ]
    (out_dir / "BETA_SCALE_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scan_root", type=Path)
    parser.add_argument("--scan-config", type=Path,
                        default=SCRIPT_DIR.parent / "config" / "beta_scale_scan.toml")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    root = args.scan_root.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_toml(args.scan_config.resolve())
    runs = find_runs(root)
    if not runs:
        raise SystemExit(f"No complete adapter runs found below {root}")
    ep = infer_beta_ep(runs)
    pairs = pair_table(runs, cfg["poles"], ep)
    summary, flow_detail, spectral_detail = aggregate_summary(runs, pairs, cfg)

    pairs.to_csv(out / "branch_tracked_beta_pairs.csv", index=False)
    summary.to_csv(out / "beta_scale_summary.csv", index=False)
    flow_detail.to_csv(out / "flow_distance_vs_iteration.csv", index=False)
    spectral_detail.to_csv(out / "spectral_scale_diagnostics.csv", index=False)
    write_report(summary, out, ep)
    reference_beta = float(cfg.get("plot", {}).get("reference_beta", 0.0))
    plot_scales(summary, out, ep, reference_beta)
    gate = {
        "runs": len(runs),
        "beta_EP": ep,
        "flow_scales_resolved": int(summary["flow_gate"].sum()),
        "pair_n5_resolved": int(summary.get("pair_gate_n5", pd.Series(dtype=bool)).sum()),
        "trace_widths_resolved": int(summary["trace_spectral_gate"].sum()),
        "jordan_widths_resolved": int(summary["jordan_spectral_gate"].sum()),
        "thermodynamic_TK_extracted": False,
    }
    (out / "beta_scale_gate_summary.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    print(f"runs={len(runs)} beta_EP={ep:.9g}")
    print(out / "BETA_SCALE_AUDIT.md")


if __name__ == "__main__":
    main()
