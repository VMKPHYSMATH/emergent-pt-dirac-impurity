#!/usr/bin/env python3
"""Comprehensive branch-tracked Jordan and quartic-scaling audit.

This analysis is deliberately matrix-valued.  It never infers a Jordan pole
from the channel trace alone.  At each requested NRG iteration it constructs
supported two-pole candidates from the full 2x2 residue matrices, tracks one
branch continuously across U and delta_coh, and evaluates

    s = (z_+ - z_-)/2,
    y = s^2,
    B = (z_+ - z_-)(W_+ - W_-)/2.

It then performs an ensemble of zero-detuning extrapolations, power-law fits,
joint complex surface fits, iteration/convergence checks, and complex quartic
coefficient diagnostics.  The goal is to distinguish:

  * robust survival of the nilpotent Jordan matrix direction;
  * generic EP perturbation, y-y0 ~ U^p with p approximately 1;
  * the stronger quartic law, which additionally requires a stable nonzero
    complex coefficient Q/(U^2 beta0^2 F).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

KEYS = ("11", "12", "21", "22")


@dataclass(frozen=True)
class Params:
    beta0: float
    U: float
    eps_d: float
    delta_eff: float
    gamma_pt: float
    delta_coh: float
    soc_lambda: float
    F_lambda: float
    Lambda: float
    z_shift: float
    nkeep: int
    config_label: str
    model_label: str


def fnum(text: str, default=math.nan) -> float:
    try:
        return float(str(text).strip())
    except Exception:
        return default


def parse_keyvals(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        for item in raw.split(";"):
            if "=" in item:
                key, value = item.split("=", 1)
                out[key.strip()] = value.strip()
    return out


def parse_params(run: Path) -> Params:
    d = parse_keyvals(run / "RUN_SUMMARY.txt")
    m = parse_keyvals(run / "SCAN_METADATA.txt")
    return Params(
        beta0=fnum(d.get("beta0", "nan")),
        U=fnum(d.get("U", m.get("U", "nan"))),
        eps_d=fnum(d.get("eps_d", "nan")),
        delta_eff=fnum(d.get("Delta_eff", "nan")),
        gamma_pt=fnum(d.get("Gamma_PT", "nan")),
        delta_coh=fnum(d.get("Delta_coh", m.get("delta_coh", "nan"))),
        soc_lambda=fnum(d.get("soc_lambda", "0"), 0.0),
        F_lambda=fnum(d.get("F_lambda", "1"), 1.0),
        Lambda=fnum(m.get("Lambda", d.get("Lambda", "nan"))),
        z_shift=fnum(m.get("z_shift", d.get("z_shift", "nan"))),
        nkeep=int(round(fnum(m.get("nkeep", "-1"), -1))),
        config_label=m.get("config_label", run.parts[-5] if len(run.parts) >= 5 else "legacy"),
        model_label=m.get("model_label", "scalar" if abs(fnum(d.get("soc_lambda", "0"), 0.0)) < 1e-14 else "soc"),
    )


def residue_matrix(row: pd.Series) -> np.ndarray:
    prefix = "add" if int(row.charge) == int(row.ground_charge) + 1 else "rem"
    return np.array(
        [complex(row[f"{prefix}{k}_real"], row[f"{prefix}{k}_imag"]) for k in KEYS],
        dtype=np.complex128,
    ).reshape(2, 2)


def components(B: np.ndarray) -> tuple[complex, complex, complex, complex]:
    b0 = 0.5 * np.trace(B)
    bx = 0.5 * (B[0, 1] + B[1, 0])
    by = (B[1, 0] - B[0, 1]) / (2j)
    bz = 0.5 * (B[0, 0] - B[1, 1])
    return b0, bx, by, bz


def candidate_pairs(df: pd.DataFrame, p: Params, args: argparse.Namespace) -> list[dict]:
    out: list[dict] = []
    N = np.array(
        [[p.delta_eff, 1j * p.gamma_pt], [1j * p.gamma_pt, -p.delta_eff]],
        dtype=np.complex128,
    )
    nn = float(np.linalg.norm(N))
    for sector, g0 in df.groupby("sector"):
        g = g0.sort_values(["matrix_support_abs", "matrix_rank"], ascending=[False, True]).head(args.pair_pool)
        if g.empty:
            continue
        max_support = float(g.matrix_support_abs.max())
        g = g[g.matrix_support_abs >= args.support_fraction * max_support]
        rows = [row for _, row in g.iterrows()]
        for ra, rb in itertools.combinations(rows, 2):
            za = complex(ra.pole_real, ra.pole_imag)
            zb = complex(rb.pole_real, rb.pole_imag)
            if (za.real, za.imag) <= (zb.real, zb.imag):
                rm, rp, zm, zp = ra, rb, za, zb
            else:
                rm, rp, zm, zp = rb, ra, zb, za
            Wm, Wp = residue_matrix(rm), residue_matrix(rp)
            dz = zp - zm
            s = 0.5 * dz
            B = 0.5 * (Wp - Wm) * dz
            b0, bx, by, bz = components(B)
            bnorm = float(np.linalg.norm(B))
            support = min(float(rm.matrix_support_abs), float(rp.matrix_support_abs))
            alignment = float(abs(np.vdot(N, B)) / (nn * bnorm)) if nn > 0 and bnorm > 0 else math.nan
            trace_fraction = float(abs(np.trace(B)) / (math.sqrt(2) * bnorm)) if bnorm > 0 else math.nan
            out.append(
                dict(
                    sector=int(sector),
                    rank_minus=int(rm.matrix_rank),
                    rank_plus=int(rp.matrix_rank),
                    z_minus=zm,
                    z_plus=zp,
                    z_center=0.5 * (zp + zm),
                    s=s,
                    y=s * s,
                    B=B,
                    B0=b0,
                    Bx=bx,
                    By=by,
                    Bz=bz,
                    Bnorm=bnorm,
                    alignment=alignment,
                    trace_fraction=trace_fraction,
                    support=support,
                    max_residual=max(float(rm.max_residual), float(rp.max_residual)),
                    biorth_error=max(float(rm.biorth_error), float(rp.biorth_error)),
                )
            )
    return out


def orient_s(s: complex, previous: complex | None) -> complex:
    if previous is None:
        return s
    return s if abs(s - previous) <= abs(-s - previous) else -s


def normalized_matrix_distance(A: np.ndarray, B: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(A)), float(np.linalg.norm(B))
    if na == 0 or nb == 0:
        return 1.0
    a, b = A / na, B / nb
    return float(min(np.linalg.norm(a - b), np.linalg.norm(a + b)))


def seed_candidate(cands: list[dict]) -> dict | None:
    valid = [c for c in cands if c["support"] > 0 and np.isfinite(c["alignment"])]
    if not valid:
        return None
    max_support = max(c["support"] for c in valid)
    # Alignment is not used to force the result.  The seed is the closest,
    # well-supported pair at the largest causal detuning.
    return min(valid, key=lambda c: (abs(c["s"]) / (1e-14 + math.sqrt(max_support)), -c["support"]))


def next_candidate(cands: list[dict], previous: dict) -> dict | None:
    if not cands:
        return None
    zscale = max(abs(previous["z_center"]), abs(previous["s"]), 1e-5)
    sscale = max(abs(previous["s"]), 1e-6)

    def cost(c: dict) -> float:
        ss = orient_s(c["s"], previous["s"])
        rank_penalty = 0.08 * (abs(c["rank_minus"] - previous["rank_minus"]) + abs(c["rank_plus"] - previous["rank_plus"]))
        return float(
            1.0 * abs(c["z_center"] - previous["z_center"]) / zscale
            + 1.5 * abs(ss - previous["s"]) / sscale
            + 0.8 * normalized_matrix_distance(c["B"], previous["B"])
            + 0.10 * abs(math.log10((c["support"] + 1e-300) / (previous["support"] + 1e-300)))
            + rank_penalty
        )

    chosen = min(cands, key=cost).copy()
    chosen["s"] = orient_s(chosen["s"], previous["s"])
    chosen["y"] = chosen["s"] ** 2
    chosen["tracking_cost"] = cost(chosen)
    return chosen


def load_runs(root: Path, iterations: Iterable[int], args: argparse.Namespace) -> list[tuple[Path, Params, int, list[dict]]]:
    requested = set(iterations)
    runs: list[tuple[Path, Params, int, list[dict]]] = []
    for summary in root.rglob("RUN_SUMMARY.txt"):
        run = summary.parent
        csv = run / "impurity_transition_residue_matrix.csv"
        if not csv.is_file():
            continue
        p = parse_params(run)
        d = pd.read_csv(csv)
        d["sector"] = d.charge.astype(int) - d.ground_charge.astype(int)
        for iteration in sorted(requested):
            di = d[d.iteration == iteration].copy()
            if not di.empty:
                runs.append((run, p, iteration, candidate_pairs(di, p, args)))
    return runs


def track_branches(runs: list[tuple[Path, Params, int, list[dict]]], args: argparse.Namespace) -> pd.DataFrame:
    grouped: dict[tuple, list] = {}
    for run, p, iteration, cands in runs:
        key = (p.config_label, p.model_label, iteration, p.delta_coh)
        grouped.setdefault(key, []).append((p.U, run, p, cands))

    rows: list[dict] = []
    roots = sorted({(key[0], key[1], key[2]) for key in grouped})
    for config_label, model_label, iteration in roots:
        deltas = sorted(
            [key[3] for key in grouped if key[:3] == (config_label, model_label, iteration)],
            reverse=True,
        )
        sectors = sorted(
            {
                c["sector"]
                for delta in deltas
                for *_rest, cs in grouped[(config_label, model_label, iteration, delta)]
                for c in cs
            }
        )
        for sector in sectors:
            previous_delta_seed: dict | None = None
            for delta in deltas:
                items = sorted(grouped[(config_label, model_label, iteration, delta)], key=lambda item: item[0])
                previous_u: dict | None = None
                first_at_delta: dict | None = None
                for U, run, p, cands in items:
                    available = [c for c in cands if c["sector"] == sector]
                    if previous_u is None:
                        current = seed_candidate(available) if previous_delta_seed is None else next_candidate(available, previous_delta_seed)
                    else:
                        current = next_candidate(available, previous_u)
                    if current is None:
                        continue
                    if previous_u is None and previous_delta_seed is None:
                        current = current.copy()
                        current["tracking_cost"] = 0.0
                    if first_at_delta is None:
                        first_at_delta = current.copy()
                    b0, bx, by, bz = current["B0"], current["Bx"], current["By"], current["Bz"]
                    reliable = bool(
                        current["max_residual"] <= args.max_residual
                        and current["biorth_error"] <= args.max_biorth_error
                        and current["support"] >= args.min_support
                    )
                    rows.append(
                        dict(
                            run=str(run), config_label=config_label, model=model_label,
                            iteration=iteration, delta_coh=delta, U=U, sector=sector,
                            Lambda=p.Lambda, z_shift=p.z_shift, nkeep=p.nkeep,
                            beta0=p.beta0, Gamma_PT=p.gamma_pt, Delta_eff=p.delta_eff,
                            F_lambda=p.F_lambda, soc_lambda=p.soc_lambda,
                            z_center_real=current["z_center"].real,
                            z_center_imag=current["z_center"].imag,
                            s_real=current["s"].real, s_imag=current["s"].imag,
                            s_abs=abs(current["s"]), y_real=current["y"].real,
                            y_imag=current["y"].imag, y_abs=abs(current["y"]),
                            B_frobenius=current["Bnorm"], alignment=current["alignment"],
                            trace_fraction=current["trace_fraction"], support=current["support"],
                            max_residual=current["max_residual"], biorth_error=current["biorth_error"],
                            reliable=reliable, tracking_cost=current["tracking_cost"],
                            rank_minus=current["rank_minus"], rank_plus=current["rank_plus"],
                            B0_real=b0.real, B0_imag=b0.imag,
                            Bx_real=bx.real, Bx_imag=bx.imag,
                            By_real=by.real, By_imag=by.imag,
                            Bz_real=bz.real, Bz_imag=bz.imag,
                            Bx_abs=abs(bx), By_abs=abs(by), Bz_abs=abs(bz),
                            nilpotent_mismatch=abs(bx - 1j * bz) / (abs(bx) + abs(bz) + 1e-300),
                        )
                    )
                    previous_u = current
                if first_at_delta is not None:
                    previous_delta_seed = first_at_delta
    return pd.DataFrame(rows)


def poly_intercept(x: np.ndarray, values: np.ndarray, degree: int) -> tuple[complex, float]:
    degree = min(degree, len(x) - 1)
    if degree < 0:
        return complex(math.nan, math.nan), math.nan
    pr = np.polyfit(x, values.real, degree)
    pi = np.polyfit(x, values.imag, degree)
    pred = np.polyval(pr, x) + 1j * np.polyval(pi, x)
    rms = float(np.sqrt(np.mean(np.abs(values - pred) ** 2)))
    return complex(pr[-1], pi[-1]), rms


def extrapolation_specs(n_delta: int) -> list[tuple[str, int, int]]:
    specs: list[tuple[str, int, int]] = []
    if n_delta >= 2:
        specs.append(("linear_all", 1, n_delta))
        specs.append(("linear_small3", 1, min(3, n_delta)))
        specs.append(("linear_drop_largest", 1, max(2, n_delta - 1)))
    if n_delta >= 3:
        specs.append(("quadratic_all", 2, n_delta))
        specs.append(("quadratic_small4", 2, min(4, n_delta)))
    # Deduplicate specifications that collapse for a small pilot grid.
    seen = set()
    unique = []
    for spec in specs:
        key = (spec[1], spec[2])
        if key not in seen:
            seen.add(key)
            unique.append(spec)
    return unique


def extrapolate_zero_detuning(tracked: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    group_cols = ["config_label", "model", "iteration", "sector", "U"]
    for keys, g0 in tracked.groupby(group_cols):
        g = g0[g0.reliable].sort_values("delta_coh").drop_duplicates("delta_coh", keep="first")
        if len(g) < 2:
            continue
        xall = g.delta_coh.to_numpy(float)
        yall = g.y_real.to_numpy(float) + 1j * g.y_imag.to_numpy(float)
        bxall = g.Bx_real.to_numpy(float) + 1j * g.Bx_imag.to_numpy(float)
        bzall = g.Bz_real.to_numpy(float) + 1j * g.Bz_imag.to_numpy(float)
        b0all = g.B0_real.to_numpy(float) + 1j * g.B0_imag.to_numpy(float)
        byall = g.By_real.to_numpy(float) + 1j * g.By_imag.to_numpy(float)
        for method, degree, nsmall in extrapolation_specs(len(g)):
            idx = np.argsort(xall)[:nsmall]
            x = xall[idx]
            y0, yrms = poly_intercept(x, yall[idx], degree)
            bx0, bxrms = poly_intercept(x, bxall[idx], degree)
            bz0, bzrms = poly_intercept(x, bzall[idx], degree)
            b00, _ = poly_intercept(x, b0all[idx], degree)
            by0, _ = poly_intercept(x, byall[idx], degree)
            rep = g.iloc[int(np.argmin(g.delta_coh.to_numpy()))]
            rows.append(
                dict(
                    config_label=keys[0], model=keys[1], iteration=int(keys[2]),
                    sector=int(keys[3]), U=float(keys[4]), method=method,
                    degree=degree, n_delta=nsmall,
                    y0_real=y0.real, y0_imag=y0.imag, y0_abs=abs(y0), s0_abs=math.sqrt(abs(y0)),
                    y_fit_rms=yrms,
                    B0_0_real=b00.real, B0_0_imag=b00.imag,
                    Bx0_real=bx0.real, Bx0_imag=bx0.imag, Bx0_abs=abs(bx0), Bx_fit_rms=bxrms,
                    By0_real=by0.real, By0_imag=by0.imag,
                    Bz0_real=bz0.real, Bz0_imag=bz0.imag, Bz0_abs=abs(bz0), Bz_fit_rms=bzrms,
                    nilpotent_mismatch0=abs(bx0 - 1j * bz0) / (abs(bx0) + abs(bz0) + 1e-300),
                    trace_component_fraction0=abs(b00) / (math.sqrt(abs(b00)**2 + abs(bx0)**2 + abs(by0)**2 + abs(bz0)**2) + 1e-300),
                    beta0=float(rep.beta0), Gamma_PT=float(rep.Gamma_PT), F_lambda=float(rep.F_lambda),
                    soc_lambda=float(rep.soc_lambda), Lambda=float(rep.Lambda), z_shift=float(rep.z_shift),
                    nkeep=int(rep.nkeep), max_tracking_cost=float(g.tracking_cost.max()),
                    max_residual=float(g.max_residual.max()), max_biorth_error=float(g.biorth_error.max()),
                    min_support=float(g.support.min()), median_alignment=float(g.alignment.median()),
                    median_trace_fraction=float(g.trace_fraction.median()),
                )
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    enriched: list[dict] = []
    group_cols2 = ["config_label", "model", "iteration", "sector", "method"]
    for _keys, g in out.groupby(group_cols2):
        g = g.sort_values("U")
        zero = g.iloc[int(np.argmin(np.abs(g.U.to_numpy(float))))]
        ybare = complex(zero.y0_real, zero.y0_imag)
        for _, row in g.iterrows():
            y = complex(row.y0_real, row.y0_imag)
            U = float(row.U)
            dy = y - ybare
            Q = y * y - ybare * y
            denom_beta = U * U * row.beta0 * row.beta0 * row.F_lambda
            denom_gamma = U * U * row.Gamma_PT * row.Gamma_PT * row.F_lambda
            d = row.to_dict()
            d.update(
                ybare_real=ybare.real, ybare_imag=ybare.imag,
                delta_y_real=dy.real, delta_y_imag=dy.imag, delta_y_abs=abs(dy),
                Q_real=Q.real, Q_imag=Q.imag, Q_abs=abs(Q),
                C_beta0_real=(Q / denom_beta).real if denom_beta > 0 else math.nan,
                C_beta0_imag=(Q / denom_beta).imag if denom_beta > 0 else math.nan,
                C_beta0_abs=abs(Q / denom_beta) if denom_beta > 0 else math.nan,
                C_beta0_phase=float(np.angle(Q / denom_beta)) if denom_beta > 0 else math.nan,
                C_gamma_real=(Q / denom_gamma).real if denom_gamma > 0 else math.nan,
                C_gamma_imag=(Q / denom_gamma).imag if denom_gamma > 0 else math.nan,
                C_gamma_abs=abs(Q / denom_gamma) if denom_gamma > 0 else math.nan,
            )
            enriched.append(d)
    return pd.DataFrame(enriched)


def log_power_fit(g: pd.DataFrame, umax: float) -> dict:
    h = g[(g.U > 0) & (g.U <= umax) & (g.delta_y_abs > 0) & np.isfinite(g.delta_y_abs)].sort_values("U")
    if len(h) < 3:
        return dict(p=math.nan, amplitude=math.nan, r2=math.nan, n=len(h))
    x = np.log(h.U.to_numpy(float))
    y = np.log(h.delta_y_abs.to_numpy(float))
    p, loga = np.polyfit(x, y, 1)
    prediction = loga + p * x
    ssr = float(np.sum((y - prediction) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return dict(p=float(p), amplitude=float(math.exp(loga)), r2=(1 - ssr / sst if sst > 0 else math.nan), n=len(h))


def complex_power_fit(g: pd.DataFrame, umax: float) -> dict:
    h = g[(g.U > 0) & (g.U <= umax)].sort_values("U")
    if len(h) < 3:
        return dict(p=math.nan, A_real=math.nan, A_imag=math.nan, rel_rms=math.nan, n=len(h))
    U = h.U.to_numpy(float)
    dy = h.delta_y_real.to_numpy(float) + 1j * h.delta_y_imag.to_numpy(float)
    scale = max(float(np.max(np.abs(dy))), 1e-30)

    def residual(theta: np.ndarray) -> np.ndarray:
        p, ar, ai = theta
        pred = complex(ar, ai) * U ** p
        diff = (pred - dy) / scale
        return np.concatenate([diff.real, diff.imag])

    pmin, pmax = 0.2, 2.5
    mag = log_power_fit(h, umax)
    p0_raw = mag["p"] if np.isfinite(mag["p"]) else 1.0
    # A noisy or nearly flat extrapolated dataset can yield a log-fit seed
    # outside the nonlinear fit bounds.  Clip the seed strictly inside the
    # admissible interval instead of aborting the complete analysis.
    p0 = float(np.clip(p0_raw, pmin + 1e-8, pmax - 1e-8))

    # Estimate the complex amplitude from all points at fixed p0.  This is
    # substantially more stable than using only the smallest-U point.
    basis = U ** p0
    denom = float(np.vdot(basis, basis).real)
    A0 = (np.vdot(basis, dy) / denom) if denom > 0 else complex(0.0, 0.0)
    if not (np.isfinite(A0.real) and np.isfinite(A0.imag)):
        A0 = complex(0.0, 0.0)

    x0 = np.array([p0, A0.real, A0.imag], dtype=float)
    lower = np.array([pmin, -np.inf, -np.inf], dtype=float)
    upper = np.array([pmax, np.inf, np.inf], dtype=float)
    try:
        fit = least_squares(residual, x0, bounds=(lower, upper), x_scale="jac")
    except ValueError:
        # Defensive retry with a neutral exponent seed; return NaNs rather
        # than terminating every model/sector/extrapolation combination.
        p0 = 1.0
        basis = U ** p0
        denom = float(np.vdot(basis, basis).real)
        A0 = (np.vdot(basis, dy) / denom) if denom > 0 else complex(0.0, 0.0)
        x0 = np.array([p0, A0.real, A0.imag], dtype=float)
        fit = least_squares(residual, x0, bounds=(lower, upper), x_scale="jac")

    p, ar, ai = fit.x
    rel_rms = float(np.sqrt(np.mean(residual(fit.x) ** 2)))
    return dict(
        p=float(p), A_real=float(ar), A_imag=float(ai), rel_rms=rel_rms,
        n=len(h), success=bool(fit.success),
        seed_p_raw=float(p0_raw), p_at_bound=bool(abs(p - pmin) < 1e-5 or abs(p - pmax) < 1e-5),
    )


def exponent_stability(ext: pd.DataFrame, umaxes: list[float]) -> pd.DataFrame:
    rows: list[dict] = []
    keys = ["config_label", "model", "iteration", "sector", "method"]
    for group_keys, g in ext.groupby(keys):
        for umax in umaxes:
            mag = log_power_fit(g, umax)
            cpx = complex_power_fit(g, umax)
            rows.append(
                dict(
                    config_label=group_keys[0], model=group_keys[1], iteration=int(group_keys[2]),
                    sector=int(group_keys[3]), method=group_keys[4], Umax=umax,
                    p_magnitude=mag["p"], amplitude_magnitude=mag["amplitude"], r2_magnitude=mag["r2"],
                    p_complex=cpx["p"], A_real=cpx["A_real"], A_imag=cpx["A_imag"],
                    complex_rel_rms=cpx["rel_rms"], n_fit=cpx["n"],
                )
            )
    return pd.DataFrame(rows)


def joint_surface_fit(g0: pd.DataFrame, umax: float, model_variant: str) -> dict:
    g = g0[(g0.reliable) & (g0.U <= umax)].copy()
    if len(g) < 12 or g.U.nunique() < 3 or g.delta_coh.nunique() < 3:
        return dict(p=math.nan, rel_rms=math.nan, n=len(g), success=False)
    U = g.U.to_numpy(float)
    delta = g.delta_coh.to_numpy(float)
    y = g.y_real.to_numpy(float) + 1j * g.y_imag.to_numpy(float)
    uscale = max(float(np.max(U)), 1e-12)
    dscale = max(float(np.max(delta)), 1e-12)
    u = U / uscale
    d = delta / dscale
    yscale = max(float(np.max(np.abs(y))), 1e-30)

    if model_variant == "linear_delta":
        ncomplex = 3  # y00, A, a1
    elif model_variant == "quadratic_delta":
        ncomplex = 4  # + a2
    elif model_variant == "quadratic_cross":
        ncomplex = 5  # + a3 u d
    else:
        raise ValueError(model_variant)

    def unpack(theta: np.ndarray) -> tuple[float, list[complex]]:
        p = theta[0]
        coeff = [complex(theta[1 + 2 * i], theta[2 + 2 * i]) for i in range(ncomplex)]
        return p, coeff

    def prediction(theta: np.ndarray) -> np.ndarray:
        p, c = unpack(theta)
        pred = c[0] + c[1] * np.where(u > 0, u ** p, 0.0) + c[2] * d
        if ncomplex >= 4:
            pred = pred + c[3] * d * d
        if ncomplex >= 5:
            pred = pred + c[4] * u * d
        return pred

    def residual(theta: np.ndarray) -> np.ndarray:
        diff = (prediction(theta) - y) / yscale
        return np.concatenate([diff.real, diff.imag])

    y00 = y[np.argmin(U + 1e3 * delta)]
    A0 = (y[np.argmax(U)] - y00) if np.max(U) > 0 else 0j
    initial_complex = [y00, A0, 0j] + [0j] * (ncomplex - 3)
    theta0 = [1.0]
    for value in initial_complex:
        theta0 += [value.real, value.imag]
    lower = [0.2] + [-np.inf] * (2 * ncomplex)
    upper = [2.5] + [np.inf] * (2 * ncomplex)
    fit = least_squares(residual, np.array(theta0), bounds=(np.array(lower), np.array(upper)), max_nfev=4000)
    p, c = unpack(fit.x)
    return dict(
        p=float(p), y00_real=c[0].real, y00_imag=c[0].imag,
        A_real=c[1].real, A_imag=c[1].imag,
        a1_real=c[2].real, a1_imag=c[2].imag,
        a2_real=(c[3].real if ncomplex >= 4 else 0.0), a2_imag=(c[3].imag if ncomplex >= 4 else 0.0),
        a3_real=(c[4].real if ncomplex >= 5 else 0.0), a3_imag=(c[4].imag if ncomplex >= 5 else 0.0),
        rel_rms=float(np.sqrt(np.mean(residual(fit.x) ** 2))), n=len(g), success=bool(fit.success),
    )


def surface_fit_table(tracked: pd.DataFrame, umaxes: list[float]) -> pd.DataFrame:
    rows: list[dict] = []
    keys = ["config_label", "model", "iteration", "sector"]
    for group_keys, g in tracked.groupby(keys):
        for umax in umaxes:
            for variant in ("linear_delta", "quadratic_delta", "quadratic_cross"):
                result = joint_surface_fit(g, umax, variant)
                rows.append(
                    dict(config_label=group_keys[0], model=group_keys[1], iteration=int(group_keys[2]),
                         sector=int(group_keys[3]), Umax=umax, model_variant=variant, **result)
                )
    return pd.DataFrame(rows)


def circular_spread(phases: np.ndarray) -> float:
    phases = phases[np.isfinite(phases)]
    if len(phases) == 0:
        return math.nan
    R = abs(np.mean(np.exp(1j * phases)))
    return float(math.sqrt(max(0.0, -2.0 * math.log(max(R, 1e-15)))))


def quartic_plateau_metrics(ext: pd.DataFrame, umaxes: list[float]) -> pd.DataFrame:
    rows: list[dict] = []
    keys = ["config_label", "model", "iteration", "sector", "method"]
    for group_keys, g0 in ext.groupby(keys):
        for umax in umaxes:
            g = g0[(g0.U > 0) & (g0.U <= umax) & np.isfinite(g0.C_beta0_abs)].sort_values("U")
            if len(g) < 3:
                continue
            C = g.C_beta0_real.to_numpy(float) + 1j * g.C_beta0_imag.to_numpy(float)
            meanC = np.mean(C)
            complex_scatter = float(np.sqrt(np.mean(np.abs(C - meanC) ** 2)) / (abs(meanC) + 1e-300))
            logU = np.log(g.U.to_numpy(float))
            logC = np.log(np.abs(C) + 1e-300)
            slope = float(np.polyfit(logU, logC, 1)[0])
            rows.append(
                dict(
                    config_label=group_keys[0], model=group_keys[1], iteration=int(group_keys[2]),
                    sector=int(group_keys[3]), method=group_keys[4], Umax=umax, n=len(g),
                    C_mean_real=meanC.real, C_mean_imag=meanC.imag, C_mean_abs=abs(meanC),
                    complex_relative_scatter=complex_scatter,
                    phase_spread=circular_spread(np.angle(C)), log_slope_absC=slope,
                )
            )
    return pd.DataFrame(rows)


def iteration_and_sector_stability(ext: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    iteration_rows: list[dict] = []
    keys = ["config_label", "model", "sector", "method", "U"]
    for group_keys, g in ext.groupby(keys):
        if g.iteration.nunique() < 2:
            continue
        vals = g.delta_y_real.to_numpy(float) + 1j * g.delta_y_imag.to_numpy(float)
        mean = np.mean(vals)
        iteration_rows.append(
            dict(config_label=group_keys[0], model=group_keys[1], sector=int(group_keys[2]),
                 method=group_keys[3], U=float(group_keys[4]), n_iterations=len(vals),
                 relative_iteration_spread=float(np.std(np.abs(vals - mean)) / (abs(mean) + 1e-300)))
        )
    sector_rows: list[dict] = []
    keys2 = ["config_label", "model", "iteration", "method", "U"]
    for group_keys, g in ext.groupby(keys2):
        if set(g.sector.astype(int)) >= {-1, 1}:
            gm = g[g.sector == -1].iloc[0]
            gp = g[g.sector == 1].iloc[0]
            ym = complex(gm.delta_y_real, gm.delta_y_imag)
            yp = complex(gp.delta_y_real, gp.delta_y_imag)
            bm = complex(gm.Bx0_real, gm.Bx0_imag)
            bp = complex(gp.Bx0_real, gp.Bx0_imag)
            sector_rows.append(
                dict(config_label=group_keys[0], model=group_keys[1], iteration=int(group_keys[2]),
                     method=group_keys[3], U=float(group_keys[4]),
                     relative_sector_y_difference=abs(ym - yp) / (0.5 * (abs(ym) + abs(yp)) + 1e-300),
                     relative_sector_Bx_difference=abs(bm - bp) / (0.5 * (abs(bm) + abs(bp)) + 1e-300))
            )
    return pd.DataFrame(iteration_rows), pd.DataFrame(sector_rows)


def convergence_summary(exponents: pd.DataFrame) -> pd.DataFrame:
    if exponents.empty:
        return pd.DataFrame()
    subset = exponents[(exponents.method == "quadratic_small4") & (exponents.Umax <= 0.05)]
    rows: list[dict] = []
    for (model, iteration, sector, Umax), g in subset.groupby(["model", "iteration", "sector", "Umax"]):
        reference = g[g.config_label.isin(["reference", "conv_ref"])]
        pref = float(reference.p_complex.median()) if not reference.empty else math.nan
        for _, row in g.iterrows():
            rows.append(
                dict(model=model, iteration=int(iteration), sector=int(sector), Umax=Umax,
                     config_label=row.config_label, p_complex=row.p_complex,
                     p_difference_from_reference=(row.p_complex - pref if np.isfinite(pref) else math.nan))
            )
    return pd.DataFrame(rows)


def gate_summary(tracked: pd.DataFrame, ext: pd.DataFrame, exponents: pd.DataFrame, plateau: pd.DataFrame) -> dict:
    reference_labels = [label for label in ("reference", "conv_ref") if label in set(tracked.config_label)]
    reference = reference_labels[0] if reference_labels else (tracked.config_label.iloc[0] if not tracked.empty else "none")
    tr = tracked[tracked.config_label == reference]
    ex = ext[ext.config_label == reference]
    ep = exponents[(exponents.config_label == reference) & (exponents.Umax <= 0.05)] if (not exponents.empty and "config_label" in exponents.columns) else pd.DataFrame()
    qp = plateau[(plateau.config_label == reference) & (plateau.Umax <= 0.05)] if (not plateau.empty and "config_label" in plateau.columns) else pd.DataFrame()

    reliable_fraction = float(tr.reliable.mean()) if not tr.empty else 0.0
    jordan_alignment = float(ex.median_alignment.median()) if not ex.empty else math.nan
    trace_fraction = float(ex.median_trace_fraction.median()) if not ex.empty else math.nan
    mismatch = float(ex.nilpotent_mismatch0.median()) if not ex.empty else math.nan
    pvals = ep.p_complex.to_numpy(float) if (not ep.empty and "p_complex" in ep.columns) else np.array([], dtype=float)
    pvals = pvals[np.isfinite(pvals)]
    p_median = float(np.median(pvals)) if len(pvals) else math.nan
    p_spread = float(np.quantile(pvals, 0.84) - np.quantile(pvals, 0.16)) if len(pvals) >= 3 else math.nan
    square_root_gate = bool(len(pvals) >= 3 and 0.85 <= p_median <= 1.15 and p_spread <= 0.30)
    if qp.empty:
        quartic_gate = False
        plateau_slope = plateau_scatter = phase_spread = math.nan
    else:
        plateau_slope = float(qp.log_slope_absC.abs().median())
        plateau_scatter = float(qp.complex_relative_scatter.median())
        phase_spread = float(qp.phase_spread.median())
        quartic_gate = bool(plateau_slope <= 0.20 and plateau_scatter <= 0.35 and phase_spread <= 0.35)
    return dict(
        reference_config=reference,
        reliable_fraction=reliable_fraction,
        numerical_gate=bool(reliable_fraction >= 0.90),
        median_alignment=jordan_alignment,
        median_trace_fraction=trace_fraction,
        median_nilpotent_mismatch=mismatch,
        jordan_gate=bool(jordan_alignment >= 0.98 and trace_fraction <= 0.10 and mismatch <= 0.05),
        p_complex_median=p_median, p_complex_68_spread=p_spread,
        square_root_gate=square_root_gate,
        quartic_plateau_abs_slope=plateau_slope,
        quartic_complex_scatter=plateau_scatter,
        quartic_phase_spread=phase_spread,
        quartic_gate=quartic_gate,
    )


def make_plots(tracked: pd.DataFrame, ext: pd.DataFrame, exponents: pd.DataFrame,
               surface: pd.DataFrame, plateau: pd.DataFrame, out: Path) -> None:
    if plt is None or tracked.empty:
        return
    reference = "reference" if "reference" in set(tracked.config_label) else tracked.config_label.iloc[0]
    preferred = ext[(ext.config_label == reference) & (ext.method == "quadratic_small4")]
    if preferred.empty:
        preferred = ext[ext.config_label == reference]

    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    for (iteration, sector), g in preferred.groupby(["iteration", "sector"]):
        h = g[(g.U > 0) & (g.delta_y_abs > 0)].sort_values("U")
        ax.loglog(h.U, h.delta_y_abs, marker="o", label=f"n={iteration}, q={sector:+d}")
    ax.set_xlabel("U")
    ax.set_ylabel(r"$|s^2(U)-s^2(0)|_{\delta\to0}$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "Delta_s2_iteration_stability.png", dpi=180)
    fig.savefig(out / "Delta_s2_iteration_stability.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    ep = exponents[(exponents.config_label == reference) & (exponents.Umax <= 0.05)] if (not exponents.empty and "config_label" in exponents.columns) else pd.DataFrame()
    for (method, sector), g in ep.groupby(["method", "sector"]):
        g = g.sort_values("iteration")
        ax.plot(g.iteration, g.p_complex, marker="o", label=f"{method}, q={sector:+d}")
    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.set_xlabel("NRG iteration")
    ax.set_ylabel("complex-fit exponent p")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "Exponent_vs_iteration.png", dpi=180)
    fig.savefig(out / "Exponent_vs_iteration.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    j = preferred[preferred.iteration == preferred.iteration.median()]
    for sector, g in j.groupby("sector"):
        h = g[g.U > 0].sort_values("U")
        ax.loglog(h.U, h.Bx0_abs, marker="o", label=f"|Bx| q={sector:+d}")
        ax.loglog(h.U, h.Bz0_abs, marker="s", linestyle="--", label=f"|Bz| q={sector:+d}")
    ax.set_xlabel("U")
    ax.set_ylabel("zero-detuning Jordan component")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "Jordan_components_refined.png", dpi=180)
    fig.savefig(out / "Jordan_components_refined.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    cdata = preferred[(preferred.U > 0) & np.isfinite(preferred.C_beta0_real)]
    for (iteration, sector), g in cdata.groupby(["iteration", "sector"]):
        g = g.sort_values("U")
        ax.plot(g.C_beta0_real, g.C_beta0_imag, marker="o", label=f"n={iteration}, q={sector:+d}")
        if len(g):
            ax.annotate(f"U={g.U.iloc[0]:g}", (g.C_beta0_real.iloc[0], g.C_beta0_imag.iloc[0]), fontsize=7)
    ax.set_xlabel(r"Re $C_{\beta_0}$")
    ax.set_ylabel(r"Im $C_{\beta_0}$")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "Quartic_coefficient_complex_plane.png", dpi=180)
    fig.savefig(out / "Quartic_coefficient_complex_plane.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    tr = tracked[tracked.config_label == reference]
    for iteration, g in tr.groupby("iteration"):
        h = g.groupby("U", as_index=False).agg(tracking_cost=("tracking_cost", "median"), support=("support", "median"))
        ax.semilogx(h[h.U > 0].U, h[h.U > 0].tracking_cost, marker="o", label=f"cost n={iteration}")
    ax.set_xlabel("U")
    ax.set_ylabel("median branch-tracking cost")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "Branch_tracking_cost.png", dpi=180)
    fig.savefig(out / "Branch_tracking_cost.pdf")
    plt.close(fig)

    if not surface.empty:
        fig, ax = plt.subplots(figsize=(7.4, 4.9))
        sf = surface[(surface.config_label == reference) & (surface.Umax <= 0.05)]
        for (variant, sector), g in sf.groupby(["model_variant", "sector"]):
            g = g.sort_values("iteration")
            ax.plot(g.iteration, g.p, marker="o", label=f"{variant}, q={sector:+d}")
        ax.axhline(1.0, linestyle="--", linewidth=1)
        ax.set_xlabel("NRG iteration")
        ax.set_ylabel("joint surface-fit p")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(out / "Joint_surface_exponent.png", dpi=180)
        fig.savefig(out / "Joint_surface_exponent.pdf")
        plt.close(fig)


def write_report(out: Path, tracked: pd.DataFrame, ext: pd.DataFrame,
                 exponents: pd.DataFrame, surface: pd.DataFrame,
                 plateau: pd.DataFrame, gates: dict, args: argparse.Namespace) -> None:
    def status(flag: bool) -> str:
        return "PASS" if flag else "NOT PASSED"

    lines = [
        "# Comprehensive Jordan and quartic scaling audit", "",
        f"NRG iterations: **{', '.join(str(x) for x in args.iterations)}**.", "",
        "The audit tracks full 2x2 residue-matrix pole pairs. It extrapolates the analytic variable `y=s^2`, compares multiple detuning models, fits the complex interaction response, and tests the complex quartic coefficient rather than its magnitude alone.", "",
        "## Executive gates", "",
        f"- Numerical reliability: **{status(gates['numerical_gate'])}** (reliable tracked fraction `{gates['reliable_fraction']:.4f}`).",
        f"- Jordan matrix survival: **{status(gates['jordan_gate'])}** (alignment `{gates['median_alignment']:.6g}`, trace fraction `{gates['median_trace_fraction']:.6g}`, nilpotent mismatch `{gates['median_nilpotent_mismatch']:.6g}`).",
        f"- Generic square-root EP perturbation: **{status(gates['square_root_gate'])}** (median complex-fit `p={gates['p_complex_median']:.6g}`, 68% spread `{gates['p_complex_68_spread']:.6g}`).",
        f"- Full quartic coefficient plateau: **{status(gates['quartic_gate'])}** (|log slope| `{gates['quartic_plateau_abs_slope']:.6g}`, complex scatter `{gates['quartic_complex_scatter']:.6g}`, phase spread `{gates['quartic_phase_spread']:.6g}`).", "",
        "A gate not passing is not automatically a proof of absence; inspect branch continuity, convergence, and reliability tables before making a physics claim.", "",
        "## Complex exponent ensemble", "",
        "| config | model | n | q | method | Umax | p_complex | rel RMS | p_magnitude | R2 |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    ref = gates["reference_config"]
    table = exponents[(exponents.config_label == ref) & (exponents.Umax <= 0.05)].sort_values(["iteration", "sector", "method", "Umax"]) if (not exponents.empty and "config_label" in exponents.columns) else pd.DataFrame()
    for _, r in table.iterrows():
        lines.append(f"| {r.config_label} | {r.model} | {int(r.iteration)} | {int(r.sector):+d} | {r.method} | {r.Umax:g} | {r.p_complex:.6g} | {r.complex_rel_rms:.4g} | {r.p_magnitude:.6g} | {r.r2_magnitude:.4g} |")

    lines += ["", "## Joint complex surface fits", "",
              "Model: `y(U,delta)=y00+A U^p+a1 delta[+a2 delta^2+a3 U delta]`.", "",
              "| config | n | q | variant | Umax | p | rel RMS |",
              "|---|---:|---:|---|---:|---:|---:|"]
    sf = surface[(surface.config_label == ref) & (surface.Umax <= 0.05)].sort_values(["iteration", "sector", "model_variant", "Umax"]) if (not surface.empty and "config_label" in surface.columns) else pd.DataFrame()
    for _, r in sf.iterrows():
        lines.append(f"| {r.config_label} | {int(r.iteration)} | {int(r.sector):+d} | {r.model_variant} | {r.Umax:g} | {r.p:.6g} | {r.rel_rms:.4g} |")

    lines += ["", "## Interpretation", "",
              "1. Jordan survival requires high alignment, a small trace fraction, and a small `|Bx-iBz|/(|Bx|+|Bz|)` mismatch in reliable data.",
              "2. Generic second-order EP perturbation requires stable `p approximately 1` across detuning models, U windows, NRG iterations, and charge sectors.",
              "3. The stronger quartic relation requires the complex coefficient `C=Q/(U^2 beta0^2 F)` to approach a nonzero plateau with stable magnitude and phase.",
              "4. `C_gamma` is exported separately because this adapter distinguishes the raw control `beta0` from the physical local matrix element `Gamma_PT`.",
              "5. The current SOC-overlap option remains an effective control model, not the microscopic `k^2 +/- lambda k` bath.", "",
              "## Output tables", "",
              "- `tracked_pairs_all_iterations.csv`: branch-resolved pole and residue data.",
              "- `zero_detuning_ensemble.csv`: all zero-detuning extrapolation methods.",
              "- `exponent_stability.csv`: U-window, method, iteration, and sector dependence.",
              "- `joint_surface_fits.csv`: simultaneous complex U/detuning fits.",
              "- `quartic_plateau_metrics.csv`: complex plateau tests.",
              "- `iteration_stability.csv`, `sector_symmetry.csv`, `convergence_summary.csv`: robustness checks.", ""]
    (out / "COMPREHENSIVE_JORDAN_SCALING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "gate_summary.json").write_text(json.dumps(gates, indent=2), encoding="utf-8")


def synthetic_self_test() -> None:
    rng = np.random.default_rng(1234)
    U = np.array([0.0025, 0.005, 0.01, 0.02, 0.03, 0.05])
    A = 2.0e-4 + 1.0e-4j
    ptrue = 1.0
    dy = A * U ** ptrue * (1 + 2e-4 * rng.normal(size=len(U)))
    g = pd.DataFrame(dict(U=U, delta_y_real=dy.real, delta_y_imag=dy.imag, delta_y_abs=np.abs(dy)))
    result = complex_power_fit(g, 0.05)
    if not (abs(result["p"] - 1.0) < 2e-3 and result["rel_rms"] < 1e-3):
        raise SystemExit(f"SELF-TEST FAIL: {result}")
    print("SELF-TEST PASS: complex power fit recovers p=1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path, nargs="?")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--iterations", type=int, nargs="+", default=[4, 5, 6])
    parser.add_argument("--umax", type=float, nargs="+", default=[0.015, 0.020, 0.030, 0.050])
    parser.add_argument("--pair-pool", type=int, default=16)
    parser.add_argument("--support-fraction", type=float, default=0.01)
    parser.add_argument("--max-residual", type=float, default=1e-8)
    parser.add_argument("--max-biorth-error", type=float, default=1e-6)
    parser.add_argument("--min-support", type=float, default=1e-5)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        synthetic_self_test()
        return
    if args.output_root is None or args.out is None:
        parser.error("output_root and --out are required unless --self-test is used")
    args.output_root = args.output_root.resolve()
    args.out.mkdir(parents=True, exist_ok=True)

    runs = load_runs(args.output_root, args.iterations, args)
    tracked = track_branches(runs, args)
    tracked.to_csv(args.out / "tracked_pairs_all_iterations.csv", index=False)
    ext = extrapolate_zero_detuning(tracked)
    ext.to_csv(args.out / "zero_detuning_ensemble.csv", index=False)
    exponents = exponent_stability(ext, args.umax)
    exponents.to_csv(args.out / "exponent_stability.csv", index=False)
    surface = surface_fit_table(tracked, args.umax)
    surface.to_csv(args.out / "joint_surface_fits.csv", index=False)
    plateau = quartic_plateau_metrics(ext, args.umax)
    plateau.to_csv(args.out / "quartic_plateau_metrics.csv", index=False)
    iteration_stability, sector_symmetry = iteration_and_sector_stability(ext)
    iteration_stability.to_csv(args.out / "iteration_stability.csv", index=False)
    sector_symmetry.to_csv(args.out / "sector_symmetry.csv", index=False)
    convergence = convergence_summary(exponents)
    convergence.to_csv(args.out / "convergence_summary.csv", index=False)
    gates = gate_summary(tracked, ext, exponents, plateau)
    make_plots(tracked, ext, exponents, surface, plateau, args.out)
    write_report(args.out, tracked, ext, exponents, surface, plateau, gates, args)
    print(f"runs={len(runs)} tracked={len(tracked)} extrapolated={len(ext)}")
    print(args.out / "COMPREHENSIVE_JORDAN_SCALING_REPORT.md")


if __name__ == "__main__":
    main()
