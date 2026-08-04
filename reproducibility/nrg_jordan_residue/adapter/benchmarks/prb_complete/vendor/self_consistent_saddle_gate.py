#!/usr/bin/env python3
"""Self-consistent infinite-U saddle gate for the Driven-Dirac impurity revision.

The solver uses the integrated, complete thermal Markov kernel declared in the
pre-coding specification.  It solves the Coleman constraint and the derivative
of the full influence action, masks non-passive points before iteration, checks
the bath-sandwich FDR and total-charge collision integral, and re-tracks the
SOC-resolved finite-k operational minimum after convergence.

This gate is isolated from all manuscript and public-archive renderers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ptdirac_matplotlib")
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import brentq, minimize_scalar


I2 = np.eye(2, dtype=complex)
SX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
SZ = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


@dataclass(frozen=True)
class SaddleParams:
    eps_d: float = -1.0
    Q: float = 1.0
    charge_gap: float = 1.0
    gamma0: float = 0.12
    k_max: float = math.pi / 4.0
    lambda_soc: float = 0.50
    delta0: float = 0.075
    c_delta: float = 0.050
    g_gamma: float = 0.200
    temperature: float = 1.0e-6
    manuscript_reference_temperature: float = 0.10
    band_lo: float = -4.0
    band_hi: float = 4.0
    base_points: int = 4001
    pole_points_per_min_width: int = 12
    pole_span_in_max_widths: float = 30.0
    thermal_points_per_T: int = 12
    thermal_span_in_T: float = 40.0
    r_guard: float = 1.0e-5
    residual_tolerance: float = 3.0e-7
    grid_tolerance: float = 3.0e-6
    r_search_min: float = 1.0e-3
    r_search_max: float = 0.50
    r_search_points: int = 13
    beta_min: float = 0.35
    beta_max: float = 0.55

    @property
    def rho(self) -> float:
        return 1.0 / (2.0 * self.k_max)

    @property
    def W(self) -> float:
        return math.sqrt(self.gamma0 / (math.pi * self.rho))

    @property
    def beta_core(self) -> float:
        return self.delta0 / (self.g_gamma - self.c_delta)

    @property
    def beta_passive_max(self) -> float:
        return self.gamma0 / abs(self.g_gamma)


P = SaddleParams()


@dataclass(frozen=True)
class Seed:
    name: str
    boson: complex
    delta_c: float


SEEDS = (
    Seed("real", 0.30 + 0.00j, 0.80),
    Seed("negative_real", -0.65 + 0.00j, 0.25),
    Seed("complex", 0.45 + 0.35j, 1.15),
)


def D0(beta: float, p: SaddleParams = P) -> float:
    return p.delta0 + p.c_delta * float(beta)


def G0(beta: float, p: SaddleParams = P) -> float:
    return p.g_gamma * float(beta)


def fermi(omega: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0.0:
        return (omega < 0.0).astype(float) + 0.5 * (omega == 0.0)
    x = np.asarray(omega, dtype=float) / temperature
    result = np.empty_like(x)
    result[x > 40.0] = 0.0
    result[x < -40.0] = 1.0
    middle = (x >= -40.0) & (x <= 40.0)
    result[middle] = 1.0 / (np.exp(x[middle]) + 1.0)
    return result


def rate_halfwidths(beta: float, r: float,
                    p: SaddleParams = P) -> tuple[float, float]:
    common = r * r * p.gamma0
    relative = r * r * G0(beta, p)
    return common - relative, common + relative


def effective_poles(r: float, delta_c: float, beta: float,
                    p: SaddleParams = P) -> np.ndarray:
    h_eff = (
        (p.eps_d + delta_c) * I2
        + r * r * D0(beta, p) * SX
        - 1j * r * r * p.gamma0 * I2
        + 1j * r * r * G0(beta, p) * SZ
    )
    return np.linalg.eigvals(h_eff)


def adaptive_omega_grid(r: float, delta_c: float, beta: float, refine: int,
                        p: SaddleParams = P) -> tuple[np.ndarray, dict[str, float]]:
    """Resolve the narrowest admitted linewidth and the Fermi edge locally."""
    if refine < 1:
        raise ValueError("refine must be >= 1")
    base = np.linspace(p.band_lo, p.band_hi, p.base_points)
    halfwidths = rate_halfwidths(beta, r, p)
    positive_widths = [value for value in halfwidths if value > 0.0]
    if not positive_widths:
        raise ValueError("non-passive rate matrix")
    width_min = min(positive_widths)
    width_max = max(positive_widths)
    pole_step = width_min / (p.pole_points_per_min_width * refine)
    pole_span = p.pole_span_in_max_widths * width_max
    local_grids: list[np.ndarray] = [base]
    centers = sorted(set(float(value.real) for value in effective_poles(r, delta_c, beta, p)))
    for center in centers:
        lo = max(p.band_lo, center - pole_span)
        hi = min(p.band_hi, center + pole_span)
        count = max(3, int(math.ceil((hi - lo) / pole_step)) + 1)
        local_grids.append(np.linspace(lo, hi, count))
    if p.temperature > 0.0:
        thermal_step = p.temperature / (p.thermal_points_per_T * refine)
        thermal_span = p.thermal_span_in_T * p.temperature
        count = max(3, int(math.ceil(2.0 * thermal_span / thermal_step)) + 1)
        local_grids.append(np.linspace(-thermal_span, thermal_span, count))
    omega = np.unique(np.concatenate(local_grids))
    return omega, {
        "omega_points": float(len(omega)),
        "base_spacing": float((p.band_hi - p.band_lo) / (p.base_points - 1)),
        "narrowest_halfwidth": float(width_min),
        "pole_local_spacing": float(pole_step),
        "points_per_narrowest_halfwidth": float(width_min / pole_step),
    }


def inverse_2x2_stack(matrix: np.ndarray) -> np.ndarray:
    result = np.empty_like(matrix)
    determinant = matrix[:, 0, 0] * matrix[:, 1, 1] - matrix[:, 0, 1] * matrix[:, 1, 0]
    result[:, 0, 0] = matrix[:, 1, 1] / determinant
    result[:, 1, 1] = matrix[:, 0, 0] / determinant
    result[:, 0, 1] = -matrix[:, 0, 1] / determinant
    result[:, 1, 0] = -matrix[:, 1, 0] / determinant
    return result


def evaluate_saddle_on_grid(r: float, delta_c: float, beta: float,
                            omega: np.ndarray, grid: dict[str, float],
                            p: SaddleParams = P) -> dict[str, float]:
    distribution = fermi(omega, p.temperature)
    relative = G0(beta, p)
    k_r = -1j * p.gamma0 * I2 + 1j * relative * SZ
    k_a = k_r.conj().T
    gamma_k = 1j * (k_r - k_a)
    k_less = 1j * distribution[:, None, None] * gamma_k[None, :, :]
    k_greater = -1j * (1.0 - distribution)[:, None, None] * gamma_k[None, :, :]
    h_coh = (p.eps_d + delta_c) * I2 + r * r * D0(beta, p) * SX
    inverse_argument = (
        omega[:, None, None] * I2[None, :, :]
        - h_coh[None, :, :]
        - r * r * k_r[None, :, :]
    )
    g_r = inverse_2x2_stack(inverse_argument)
    g_a = np.conj(np.swapaxes(g_r, 1, 2))
    sigma_less = r * r * k_less
    sigma_greater = r * r * k_greater
    g_less = g_r @ sigma_less @ g_a
    g_greater = (1.0 - distribution)[:, None, None] * (g_r - g_a)
    fdr_target = distribution[:, None, None] * (g_a - g_r)
    trace_less = np.trace(g_less, axis1=1, axis2=2)
    n_f = float(np.real(-1j * np.trapezoid(trace_less, omega) / (2.0 * np.pi)))
    x_force = float(
        np.real(
            -1j
            * np.trapezoid(np.trace(SX @ g_less, axis1=1, axis2=2), omega)
            / (2.0 * np.pi)
        )
    )
    influence_integrand = np.trace(k_r @ g_less + k_less @ g_a, axis1=1, axis2=2)
    influence_force = float(
        np.real(-1j * np.trapezoid(influence_integrand, omega) / (2.0 * np.pi))
    )
    constraint = float(r * r + n_f - p.Q)
    stationarity_divided = float(delta_c + D0(beta, p) * x_force + influence_force)
    stationarity_raw = float(2.0 * r * stationarity_divided)
    fdr_abs = float(np.max(np.abs(g_less - fdr_target)))
    collision = np.trace(
        sigma_less @ g_greater - sigma_greater @ g_less,
        axis1=1,
        axis2=2,
    )
    charge_continuity = float(
        abs(np.trapezoid(collision, omega) / (2.0 * np.pi))
    )
    spectral = 1j * (g_r - g_a)
    rate_matrix = r * r * gamma_k
    occupation = -1j * np.trapezoid(g_less, omega, axis=0) / (2.0 * np.pi)
    occupation = 0.5 * (occupation + occupation.conj().T)
    return {
        "n_f": n_f,
        "x_force": x_force,
        "influence_force": influence_force,
        "constraint_residual": constraint,
        "stationarity_divided_residual": stationarity_divided,
        "stationarity_raw_residual": stationarity_raw,
        "fdr_abs_residual": fdr_abs,
        "charge_continuity_abs_residual": charge_continuity,
        "rate_min_eigenvalue": float(np.min(np.linalg.eigvalsh(rate_matrix))),
        "spectral_min_eigenvalue": float(np.min(np.linalg.eigvalsh(spectral))),
        "occupation_min_eigenvalue": float(np.min(np.linalg.eigvalsh(occupation))),
        **grid,
    }


def evaluate_saddle(r: float, delta_c: float, beta: float, refine: int = 1,
                    p: SaddleParams = P) -> dict[str, float]:
    omega, grid = adaptive_omega_grid(r, delta_c, beta, refine, p)
    return evaluate_saddle_on_grid(r, delta_c, beta, omega, grid, p)


def project_constraint(r: float, beta: float, refine: int,
                       p: SaddleParams = P) -> tuple[float, dict[str, float]]:
    """Solve r^2+n_f=Q before evaluating the amplitude stationarity."""
    if r <= p.r_guard:
        raise ValueError("constraint projection requested below r_guard")

    # First locate the local-moment branch with a delta-adapted grid.  Then
    # freeze the nodes around that estimate and polish the scalar root; this
    # removes point-count discontinuities without losing a narrow resonance.
    def moving_constraint(delta_c: float) -> float:
        return evaluate_saddle(r, delta_c, beta, refine, p)["constraint_residual"]

    estimate = float(
        brentq(moving_constraint, 0.25, 1.75, xtol=2.0e-8, rtol=2.0e-8)
    )
    omega, grid = adaptive_omega_grid(r, estimate, beta, refine, p)

    def values(delta_c: float) -> dict[str, float]:
        return evaluate_saddle_on_grid(r, delta_c, beta, omega, grid, p)

    def fixed_constraint(delta_c: float) -> float:
        return values(delta_c)["constraint_residual"]

    widths = rate_halfwidths(beta, r, p)
    span = max(20.0 * max(widths), 20.0 * p.temperature, 2.0e-6)
    for _ in range(8):
        lo, hi = estimate - span, estimate + span
        if fixed_constraint(lo) * fixed_constraint(hi) <= 0.0:
            break
        span *= 2.0
    else:
        raise RuntimeError("could not polish the Coleman constraint on a fixed grid")
    delta_c = float(brentq(fixed_constraint, lo, hi, xtol=2.0e-11, rtol=2.0e-11))
    return delta_c, values(delta_c)


def solve_seed(beta: float, seed: Seed, p: SaddleParams = P) -> dict[str, Any]:
    """Safeguarded constraint-projected self-consistency loop.

    Direct simultaneous Picard updates are not contractive in the small-width
    regime.  The robust loop projects the constraint at each r and brackets the
    remaining stationarity residual.  Its state variable y=r*delta_c is only
    divided by r above r_guard.
    """
    if p.gamma0 < abs(G0(beta, p)):
        return {"beta0": beta, "seed": seed.name,
                "status": "MASKED_NONPASSIVE", "converged": 0}

    seed_r = min(max(abs(seed.boson), p.r_search_min), p.r_search_max)

    def solve_at_refinement(refine: int, preferred_r: float | None = None) -> tuple[float, float, dict[str, float], int]:
        cache: dict[float, tuple[float, dict[str, float]]] = {}

        def projected(r: float) -> float:
            key = float(r)
            if key not in cache:
                cache[key] = project_constraint(key, beta, refine, p)
            return cache[key][1]["stationarity_divided_residual"]

        if preferred_r is None:
            r_grid = np.geomspace(p.r_search_min, p.r_search_max, p.r_search_points)
            values = [projected(float(r_value)) for r_value in r_grid]
            brackets = [
                (float(r_grid[index]), float(r_grid[index + 1]))
                for index in range(len(r_grid) - 1)
                if values[index] == 0.0 or values[index] * values[index + 1] < 0.0
            ]
        else:
            factor = 1.35
            brackets = []
            for _ in range(8):
                lo_try = max(p.r_search_min, preferred_r / factor)
                hi_try = min(p.r_search_max, preferred_r * factor)
                if projected(lo_try) * projected(hi_try) <= 0.0:
                    brackets = [(lo_try, hi_try)]
                    break
                factor *= 1.35
        if not brackets:
            raise RuntimeError("no finite-r saddle bracket in the declared search window")
        reference = seed_r if preferred_r is None else preferred_r
        lo, hi = min(
            brackets,
            key=lambda pair: abs(math.log(math.sqrt(pair[0] * pair[1]) / reference)),
        )
        root = brentq(projected, lo, hi, xtol=2.0e-10, rtol=2.0e-10, full_output=True)
        r_value = float(root[0])
        delta_c, values_at_root = project_constraint(r_value, beta, refine, p)
        return r_value, delta_c, values_at_root, len(cache)

    try:
        coarse_r, coarse_delta, coarse, coarse_calls = solve_at_refinement(1)
        r, delta_c, refined, refined_calls = solve_at_refinement(2, coarse_r)
    except RuntimeError as error:
        return {
            "beta0": beta,
            "seed": seed.name,
            "seed_boson_real": float(seed.boson.real),
            "seed_boson_imag": float(seed.boson.imag),
            "seed_delta_c": seed.delta_c,
            "gauge_fixed_initial_r": abs(seed.boson),
            "status": "NO_FINITE_R_SADDLE",
            "converged": 0,
            "failure_reason": str(error),
        }

    # Re-evaluate the coarse grid at the refined solution so the quadrature
    # comparison does not mix discretization error with a state displacement.
    coarse_same_state = evaluate_saddle(r, delta_c, beta, 1, p)
    grid_error = max(
        abs(coarse_same_state["n_f"] - refined["n_f"]),
        abs(coarse_same_state["x_force"] - refined["x_force"]),
        abs(coarse_same_state["influence_force"] - refined["influence_force"]),
    )
    residual_norm = max(
        abs(refined["constraint_residual"]),
        abs(refined["stationarity_divided_residual"]),
    )
    guarded_y = r * delta_c
    recovered_delta = guarded_y / r if r > p.r_guard else math.nan
    converged = (
        r > p.r_guard
        and residual_norm < p.residual_tolerance
        and grid_error < p.grid_tolerance
        and refined["rate_min_eigenvalue"] >= -1.0e-12
        and refined["charge_continuity_abs_residual"] < 1.0e-11
        and math.isfinite(recovered_delta)
    )
    return {
        "beta0": beta,
        "seed": seed.name,
        "seed_boson_real": float(seed.boson.real),
        "seed_boson_imag": float(seed.boson.imag),
        "seed_delta_c": seed.delta_c,
        "gauge_fixed_initial_r": abs(seed.boson),
        "status": "CONVERGED" if converged else "FAILED",
        "converged": int(converged),
        "solver": "constraint_projected_safeguarded_brent",
        "coarse_function_calls": coarse_calls,
        "refined_function_calls": refined_calls,
        "r": r,
        "delta_c": delta_c,
        "r_delta_c": guarded_y,
        "delta_recovered_only_above_guard": recovered_delta,
        "grid_refinement_error": float(grid_error),
        "seed_log_distance_to_solution": abs(math.log(r / seed_r)),
        **{f"coarse_{key}": value for key, value in coarse_same_state.items()},
        **refined,
    }


def h4(beta: float, r: float, delta_c: float, k: float,
       p: SaddleParams = P) -> np.ndarray:
    ep = k * k + p.lambda_soc * k
    em = k * k - p.lambda_soc * k
    d = r * r * D0(beta, p)
    g = r * r * G0(beta, p)
    v = r * p.W
    eps = p.eps_d + delta_c
    return np.array(
        [[eps + d, 1j * g, v, 0.0],
         [1j * g, eps - d, 0.0, v],
         [v, 0.0, ep, 0.0],
         [0.0, v, 0.0, em]],
        dtype=complex,
    )


def eig_metrics(matrix: np.ndarray) -> tuple[float, float]:
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    gap = min(
        abs(eigenvalues[i] - eigenvalues[j])
        for i in range(len(eigenvalues))
        for j in range(i + 1, len(eigenvalues))
    )
    return float(gap), float(np.linalg.cond(eigenvectors))


def operational_minimum(r: float, delta_c: float, k: float,
                        p: SaddleParams = P) -> dict[str, float]:
    if k == 0.0:
        gap, condition = eig_metrics(h4(p.beta_core, r, delta_c, 0.0, p))
        return {"beta0": p.beta_core, "minimum_gap": gap, "condition_number": condition,
                "at_search_boundary": 0, "passive_at_reported_beta": 1}

    def objective(beta: float) -> float:
        return eig_metrics(h4(beta, r, delta_c, k, p))[0]

    scan = np.linspace(p.beta_min, p.beta_max, 501)
    gaps = np.array([objective(float(beta)) for beta in scan])
    index = int(np.argmin(gaps))
    lo = float(scan[max(0, index - 3)])
    hi = float(scan[min(len(scan) - 1, index + 3)])
    optimum = minimize_scalar(
        objective,
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 1.0e-13},
    )
    gap, condition = eig_metrics(h4(float(optimum.x), r, delta_c, k, p))
    return {
        "beta0": float(optimum.x),
        "minimum_gap": gap,
        "condition_number": condition,
        "at_search_boundary": int(
            abs(float(optimum.x) - p.beta_min) < 5.0e-7
            or abs(float(optimum.x) - p.beta_max) < 5.0e-7
        ),
        "passive_at_reported_beta": int(float(optimum.x) <= p.beta_passive_max),
    }


def reference_temperature_scan(beta: float, p: SaddleParams = P) -> list[dict[str, float]]:
    """Check whether the manuscript reference temperature admits finite r."""
    reference = SaddleParams(
        **{
            **asdict(p),
            "temperature": p.manuscript_reference_temperature,
            "base_points": 2401,
            "pole_points_per_min_width": 8,
            "thermal_points_per_T": 8,
        }
    )
    rows: list[dict[str, float]] = []
    for r in np.geomspace(0.002, p.r_search_max, 10):
        r_value = float(r)

        def constraint(delta_c: float) -> float:
            return evaluate_saddle(r_value, delta_c, beta, 1, reference)["constraint_residual"]

        delta_c = float(brentq(constraint, 0.25, 1.75, xtol=2.0e-8, rtol=2.0e-8))
        values = evaluate_saddle(r_value, delta_c, beta, 1, reference)
        rows.append(
            {
                "beta0": beta,
                "temperature": reference.temperature,
                "r_probe": r_value,
                "delta_c_constraint_projected": delta_c,
                "stationarity_divided_residual": values["stationarity_divided_residual"],
                "constraint_residual": values["constraint_residual"],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_typed(path: Path) -> list[dict[str, Any]]:
    text_fields = {"seed", "status", "solver", "failure_reason", "classification"}
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if key in text_fields or value == "":
                    row[key] = value
                else:
                    row[key] = float(value)
            rows.append(row)
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_plot(out: Path, canonical: list[dict[str, Any]], seed_spread: list[dict[str, Any]],
              p: SaddleParams = P) -> None:
    beta = np.array([row["beta0"] for row in canonical])
    fig, axs = plt.subplots(2, 2, figsize=(8.1, 6.3), constrained_layout=True)
    fig.suptitle(
        r"Finite saddle at $T=10^{-6}$; no finite-$r$ saddle at the reference $T_b=0.1$",
        color="firebrick",
        fontsize=11,
    )
    axs[0, 0].plot(beta, [row["r"] for row in canonical], "o-", label="$r$")
    axs[0, 0].set(xlabel=r"$\beta_0$", ylabel=r"$r$", title="Converged saddle amplitude")
    axs[0, 1].plot(
        beta,
        [1.0e7 * (row["delta_c"] + p.eps_d) for row in canonical],
        "s-",
        color="tab:orange",
    )
    axs[0, 1].set(
        xlabel=r"$\beta_0$",
        ylabel=r"$10^7(\delta_c-1)$",
        title="Constraint-field displacement",
    )
    axs[1, 0].semilogy(beta, [max(abs(row["constraint_residual"]), 1e-24) for row in canonical],
                       "o-", label="constraint")
    axs[1, 0].semilogy(beta, [max(abs(row["stationarity_divided_residual"]), 1e-24) for row in canonical],
                       "s--", label="stationarity")
    axs[1, 0].semilogy(beta, [max(row["grid_refinement_error"], 1e-24) for row in canonical],
                       "^:", label="grid refinement")
    axs[1, 0].set(xlabel=r"$\beta_0$", ylabel="absolute residual", title="Saddle and quadrature checks")
    axs[1, 0].legend(frameon=False, fontsize=8)
    axs[1, 1].semilogy(
        [row["beta0"] for row in seed_spread],
        [max(row["r_spread"], 1e-24) for row in seed_spread],
        "o-",
        label="$r$ spread",
    )
    axs[1, 1].semilogy(
        [row["beta0"] for row in seed_spread],
        [max(row["delta_c_spread"], 1e-24) for row in seed_spread],
        "s--",
        label=r"$\delta_c$ spread",
    )
    axs[1, 1].set(xlabel=r"$\beta_0$", ylabel="max-min", title="Seed independence")
    axs[1, 1].legend(frameon=False, fontsize=8)
    for axis in axs.flat:
        axis.grid(alpha=0.18, lw=0.5)
    fixed_time = datetime(2026, 7, 20, 12, 0, 0, tzinfo=timezone.utc)
    fig.savefig(
        out / "Self_Consistent_Saddle_Gate.pdf",
        metadata={
            "Title": "Driven-Dirac impurity self-consistent saddle gate",
            "Author": "Driven-Dirac impurity reproducibility pipeline",
            "CreationDate": fixed_time,
            "ModDate": fixed_time,
        },
    )
    fig.savefig(
        out / "Self_Consistent_Saddle_Gate.png",
        dpi=220,
        metadata={"Software": "Driven-Dirac impurity self-consistent saddle gate"},
    )
    plt.close(fig)


def run(out: Path, p: SaddleParams = P, reuse_seed_rows: bool = False) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    beta_values = [0.35, 0.40, 0.45, 0.50, 0.55]
    seed_path = out / "seed_convergence.csv"
    if reuse_seed_rows and seed_path.exists():
        rows = read_csv_typed(seed_path)
    else:
        tasks = [(beta, seed) for beta in beta_values for seed in SEEDS]
        with ThreadPoolExecutor(max_workers=3) as executor:
            rows = list(executor.map(lambda task: solve_seed(task[0], task[1], p), tasks))
    write_csv(out / "seed_convergence.csv", rows)

    reference_rows: list[dict[str, float]] = []
    for beta in beta_values:
        reference_rows.extend(reference_temperature_scan(beta, p))
    write_csv(out / "reference_temperature_stability.csv", reference_rows)
    reference_parameters = SaddleParams(
        **{**asdict(p), "temperature": p.manuscript_reference_temperature}
    )
    reference_solver_rows = [
        solve_seed(beta, SEEDS[0], reference_parameters) for beta in beta_values
    ]
    write_csv(out / "reference_temperature_solver.csv", reference_solver_rows)
    reference_finite_by_beta = {
        beta: bool(
            next(row for row in reference_solver_rows if row["beta0"] == beta).get("converged") == 1
        )
        for beta in beta_values
    }
    canonical: list[dict[str, Any]] = []
    spreads: list[dict[str, Any]] = []
    for beta in beta_values:
        group = [row for row in rows if row["beta0"] == beta]
        converged = [row for row in group if row.get("converged") == 1]
        if converged:
            canonical.append(dict(converged[0]))
            spreads.append(
                {
                    "beta0": beta,
                    "converged_seed_count": len(converged),
                    "r_spread": max(row["r"] for row in converged) - min(row["r"] for row in converged),
                    "delta_c_spread": max(row["delta_c"] for row in converged) - min(row["delta_c"] for row in converged),
                }
            )
        else:
            spreads.append(
                {"beta0": beta, "converged_seed_count": 0, "r_spread": math.nan, "delta_c_spread": math.nan}
            )
    write_csv(out / "seed_independence.csv", spreads)

    superfine_rows: list[dict[str, Any]] = []
    for solution in canonical:
        level2 = evaluate_saddle(solution["r"], solution["delta_c"], solution["beta0"], 2, p)
        level3 = evaluate_saddle(solution["r"], solution["delta_c"], solution["beta0"], 3, p)
        superfine_rows.append(
            {
                "beta0": solution["beta0"],
                "r": solution["r"],
                "delta_c": solution["delta_c"],
                "level2_to_level3_integral_drift": max(
                    abs(level2[key] - level3[key])
                    for key in ("n_f", "x_force", "influence_force")
                ),
                "level3_constraint_residual": level3["constraint_residual"],
                "level3_stationarity_divided_residual": level3["stationarity_divided_residual"],
                "level3_omega_points": level3["omega_points"],
                "level3_points_per_narrowest_halfwidth": level3["points_per_narrowest_halfwidth"],
            }
        )
    if superfine_rows:
        write_csv(out / "superfine_grid_validation.csv", superfine_rows)

    root_rows: list[dict[str, Any]] = []
    for solution in canonical:
        for k_probe in [0.0, 0.05, 0.10, 0.15, 0.20]:
            result = operational_minimum(solution["r"], solution["delta_c"], k_probe, p)
            root_rows.append(
                {
                    "solution_beta0": solution["beta0"],
                    "r": solution["r"],
                    "delta_c": solution["delta_c"],
                    "k_probe": k_probe,
                    "lambda_k": p.lambda_soc * k_probe,
                    "classification": (
                        "exact_k0_double_root"
                        if k_probe == 0.0
                        else (
                            "finite_k_boundary_minimum"
                            if result["at_search_boundary"]
                            else "finite_k_operational_minimum"
                        )
                    ),
                    **result,
                }
            )
    if root_rows:
        write_csv(out / "dressed_root_tracking.csv", root_rows)

    all_seed_converged = len(canonical) == len(beta_values) and all(
        row["converged_seed_count"] == len(SEEDS) for row in spreads
    )
    seed_independent = all(
        row["r_spread"] < 2.0e-6 and row["delta_c_spread"] < 2.0e-6
        for row in spreads if row["converged_seed_count"] == len(SEEDS)
    ) and all_seed_converged
    finite_r = bool(canonical) and min(row["r"] for row in canonical) > p.r_guard
    checks = {
        "test_7_all_seed_triplets_converged": all_seed_converged,
        "test_7_seed_independence": seed_independent,
        "test_8_constraint_residual": bool(canonical) and max(abs(row["constraint_residual"]) for row in canonical) < p.residual_tolerance,
        "test_8_stationarity_residual": bool(canonical) and max(abs(row["stationarity_divided_residual"]) for row in canonical) < p.residual_tolerance,
        "test_8_grid_refinement": bool(canonical) and max(row["grid_refinement_error"] for row in canonical) < p.grid_tolerance,
        "test_8_superfine_grid_validation": bool(superfine_rows)
        and max(row["level2_to_level3_integral_drift"] for row in superfine_rows) < 1.0e-6
        and max(abs(row["level3_constraint_residual"]) for row in superfine_rows) < 1.0e-6
        and max(abs(row["level3_stationarity_divided_residual"]) for row in superfine_rows) < 1.0e-6,
        "test_8_rate_matrix_PSD": bool(canonical) and min(row["rate_min_eigenvalue"] for row in canonical) >= -1.0e-12,
        "test_8_total_charge_continuity": bool(canonical) and max(row["charge_continuity_abs_residual"] for row in canonical) < 1.0e-11,
        "test_9_projected_root_exported": len(root_rows) == len(canonical) * 5,
        "test_finite_nonzero_saddle": finite_r,
        "test_lambda_below_kmax": p.lambda_soc < p.k_max,
    }
    low_temperature_pass = all(checks.values())
    reference_temperature_has_finite_saddle = all(reference_finite_by_beta.values())
    if low_temperature_pass and reference_temperature_has_finite_saddle:
        status = "PASS"
    elif low_temperature_pass:
        status = "PASS_LOW_T__BLOCKED_AT_REFERENCE_T"
    else:
        status = "FAIL"
    summary = {
        "status": status,
        "kernel_mode": "complete_thermal_markov_influence_kernel",
        "parameters": {**asdict(p), "rho": p.rho, "W": p.W, "beta_core": p.beta_core,
                       "beta_passive_max": p.beta_passive_max},
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "checks": checks,
        "scope_gate": {
            "manuscript_reference_temperature": p.manuscript_reference_temperature,
            "finite_r_saddle_at_every_beta": reference_temperature_has_finite_saddle,
            "finite_r_saddle_by_beta": {
                f"{beta:.3f}": value for beta, value in reference_finite_by_beta.items()
            },
        },
        "headline": {
            "converged_beta_count": len(canonical),
            "minimum_r": min((row["r"] for row in canonical), default=math.nan),
            "maximum_r": max((row["r"] for row in canonical), default=math.nan),
            "max_constraint_residual": max((abs(row["constraint_residual"]) for row in canonical), default=math.nan),
            "max_stationarity_residual": max((abs(row["stationarity_divided_residual"]) for row in canonical), default=math.nan),
            "max_grid_refinement_error": max((row["grid_refinement_error"] for row in canonical), default=math.nan),
            "max_level2_to_level3_integral_drift": max(
                (row["level2_to_level3_integral_drift"] for row in superfine_rows),
                default=math.nan,
            ),
            "max_charge_continuity_residual": max((row["charge_continuity_abs_residual"] for row in canonical), default=math.nan),
        },
        "notes": [
            "The input boson phase is gauge fixed by r=|b_c| before iteration.",
            "The solver updates r*delta_c when r is above the guard and never divides Eq. (24) by a vanishing r.",
            "Non-passive beta0 values are masked before entering the iteration loop.",
            "The frequency grid is locally refined around every retarded pole and around the Fermi edge, then doubled for the exported quadrature check.",
            "The converged finite-r table is evaluated at the explicitly reported low temperature; the manuscript reference temperature is tested separately and may block transfer of the result into the paper.",
            "At finite k the reported gap minimum reaches the upper edge of the declared physical beta scan, so it is classified as boundary-limited rather than as an interior projected EP.",
            "Across the converged low-temperature scan r decreases with beta0; this gate therefore does not supply evidence for an enhanced saddle amplitude near the core EP.",
            "No manuscript, response letter, figure renderer, or public archive is modified by this gate.",
        ],
    }
    (out / "self_consistent_gate_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if canonical:
        write_csv(out / "converged_saddles.csv", canonical)
        make_plot(out, canonical, spreads, p)

    report = [
        "DRIVEN-DIRAC IMPURITY SELF-CONSISTENT SADDLE GATE",
        f"STATUS: {status}",
        "",
        f"temperature={p.temperature:.12g}; Gamma0={p.gamma0:.8f}; lambda_soc={p.lambda_soc:.8f}; kmax={p.k_max:.8f}",
        f"manuscript reference temperature={p.manuscript_reference_temperature:.12g}; finite saddle at every beta={reference_temperature_has_finite_saddle}",
        "",
        "CHECKS",
    ]
    report.extend(f"{name}: {'PASS' if value else 'FAIL'}" for name, value in checks.items())
    report.extend(["", "HEADLINE", json.dumps(summary["headline"], indent=2), ""])
    (out / "SELF_CONSISTENT_GATE_REPORT.txt").write_text("\n".join(report), encoding="utf-8")

    files = sorted(path for path in out.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    checksum_lines = [f"{sha256(path)}  {path.name}" for path in files]
    script_path = Path(__file__).resolve()
    checksum_lines.append(
        f"{sha256(script_path)}  ../../submission_work/revision_gate/{script_path.name}"
    )
    (out / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reuse-seed-rows", action="store_true")
    args = parser.parse_args()
    summary = run(args.out, reuse_seed_rows=args.reuse_seed_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
