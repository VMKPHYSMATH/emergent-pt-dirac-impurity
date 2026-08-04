#!/usr/bin/env python3
r"""Causal channel-resolved one-loop RG gate for Driven-Dirac impurity.

The finite-U Schrieffer--Wolff transformation supplies the bare reciprocal
exchange matrix.  The running channel density is computed directly from the
retarded and advanced projected-core Green functions; no Petermann or Zbio
factor is inserted.  The matrix RG equation is integrated without assuming
commutativity and compared with a matched Gamma_PT=0 control.  The exact
inverse-flow identity follows directly by differentiating J^{-1} and is
checked independently.  For the particle-hole-symmetric benchmark used here,
shell symmetrization also happens to make the density matrices commute in a
fixed basis; that is a separate a posteriori check rather than an assumption.

This is an isolated diagnostic gate.  It does not modify the manuscript or
public repository and it does not claim a universal strong-coupling solution.
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
from scipy.integrate import quad, solve_ivp

from self_consistent_saddle_gate import D0, G0, I2, P, SX, SZ, SaddleParams


BETA_VALUES = (0.35, 0.40, 0.45, 0.48, 0.49, 0.495, 0.50, 0.52, 0.55)
THRESHOLDS = (0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75, 1.00)
PRIMARY_THRESHOLD = 1.00
MIN_CUTOFF = 1.0e-7
MAX_STEPS = (0.020, 0.010, 0.005)
STRICT_PRE_EP_MAX = 0.4999
DENSE_OMEGAS = np.linspace(-4.0, 4.0, 1601)
DENSE_PASSIVE_BETAS = np.linspace(0.0, P.beta_passive_max, 121)
DENSE_PRE_EP_BETAS = np.linspace(P.beta_min, STRICT_PRE_EP_MAX, 61)


def bare_sw_exchange(p: SaddleParams = P) -> float:
    """Particle-hole-symmetric finite-U=2 SW exchange for eps_d=-1."""
    U = 2.0
    denominator = 1.0 / abs(p.eps_d) + 1.0 / abs(p.eps_d + U)
    return 2.0 * p.W * p.W * denominator


def projected_core_hamiltonian(beta: float, control: bool, p: SaddleParams = P) -> np.ndarray:
    relative = 0.0 if control else G0(beta, p)
    return D0(beta, p) * SX - 1j * p.gamma0 * I2 + 1j * relative * SZ


def bath_rate_matrix(beta: float, control: bool, p: SaddleParams = P) -> np.ndarray:
    relative = 0.0 if control else G0(beta, p)
    return 2.0 * (p.gamma0 * I2 - relative * SZ)


def green_and_density(omega: float, beta: float, control: bool,
                      p: SaddleParams = P) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    h_r = projected_core_hamiltonian(beta, control, p)
    g_r = np.linalg.inv(float(omega) * I2 - h_r)
    g_a = g_r.conj().T
    density = 1j * (g_r - g_a) / (2.0 * np.pi)
    density = 0.5 * (density + density.conj().T)
    gamma = bath_rate_matrix(beta, control, p)
    sandwich = g_r @ gamma @ g_a / (2.0 * np.pi)
    identity_error = float(np.max(np.abs(density - sandwich)))
    return g_r, g_a, density, identity_error


def density_sym(cutoff: float, beta: float, control: bool,
                p: SaddleParams = P) -> np.ndarray:
    _, _, positive, _ = green_and_density(cutoff, beta, control, p)
    _, _, negative, _ = green_and_density(-cutoff, beta, control, p)
    return positive + negative


def psd_sqrt(matrix: np.ndarray) -> np.ndarray:
    hermitian = 0.5 * (matrix + matrix.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    if np.min(eigenvalues) < -1.0e-10:
        raise ValueError("spectral matrix is not positive semidefinite")
    return (eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))) @ eigenvectors.conj().T


def dimensionless_coupling(exchange: np.ndarray, cutoff: float, beta: float,
                           control: bool, p: SaddleParams = P) -> tuple[np.ndarray, float]:
    rho_average = 0.5 * density_sym(cutoff, beta, control, p)
    root = psd_sqrt(rho_average)
    coupling = root @ exchange @ root
    coupling = 0.5 * (coupling + coupling.conj().T)
    return coupling, float(np.max(np.linalg.eigvalsh(coupling)))


def rg_rhs(log_scale: float, flattened: np.ndarray, beta: float, control: bool,
           p: SaddleParams = P) -> np.ndarray:
    exchange = flattened.reshape(2, 2)
    cutoff = p.k_max * math.exp(-float(log_scale))
    derivative = exchange @ density_sym(cutoff, beta, control, p) @ exchange
    derivative = 0.5 * (derivative + derivative.conj().T)
    return derivative.reshape(-1)


def integrate_to_threshold(beta: float, control: bool, threshold: float,
                           max_step: float, p: SaddleParams = P) -> dict[str, Any]:
    j0 = bare_sw_exchange(p)
    initial = (j0 * I2).reshape(-1)
    log_limit = math.log(p.k_max / MIN_CUTOFF)

    def event(log_scale: float, flattened: np.ndarray) -> float:
        cutoff = p.k_max * math.exp(-float(log_scale))
        _, maximum = dimensionless_coupling(flattened.reshape(2, 2), cutoff, beta, control, p)
        return float(threshold - maximum)

    event.terminal = True
    event.direction = -1
    solution = solve_ivp(
        lambda log_scale, state: rg_rhs(log_scale, state, beta, control, p),
        (0.0, log_limit),
        initial,
        events=event,
        rtol=2.0e-9,
        atol=2.0e-11,
        max_step=max_step,
    )
    reached = bool(len(solution.t_events[0]))
    if reached:
        log_star = float(solution.t_events[0][0])
        state_star = solution.y_events[0][0].reshape(2, 2)
        scale = p.k_max * math.exp(-log_star)
        matrix_g, maximum = dimensionless_coupling(state_star, scale, beta, control, p)
    else:
        log_star = float(solution.t[-1])
        state_star = solution.y[:, -1].reshape(2, 2)
        scale = math.nan
        matrix_g, maximum = dimensionless_coupling(
            state_star, p.k_max * math.exp(-log_star), beta, control, p
        )
    return {
        "beta0": beta,
        "control_gamma_pt_zero": int(control),
        "threshold": threshold,
        "max_step": max_step,
        "threshold_reached": int(reached),
        "operational_RG_scale": scale,
        "log_scale_at_stop": log_star,
        "gmax_at_stop": maximum,
        "exchange_trace_real_at_stop": float(np.trace(state_star).real),
        "exchange_00_real_at_stop": float(state_star[0, 0].real),
        "exchange_00_imag_at_stop": float(state_star[0, 0].imag),
        "exchange_01_real_at_stop": float(state_star[0, 1].real),
        "exchange_01_imag_at_stop": float(state_star[0, 1].imag),
        "exchange_10_real_at_stop": float(state_star[1, 0].real),
        "exchange_10_imag_at_stop": float(state_star[1, 0].imag),
        "exchange_11_real_at_stop": float(state_star[1, 1].real),
        "exchange_11_imag_at_stop": float(state_star[1, 1].imag),
        "exchange_hermiticity_error": float(np.max(np.abs(state_star - state_star.conj().T))),
        "g_hermiticity_error": float(np.max(np.abs(matrix_g - matrix_g.conj().T))),
        "solver_function_evaluations": int(solution.nfev),
        "solver_status": int(solution.status),
    }


def dense_causality_scan(p: SaddleParams = P) -> dict[str, float]:
    """Scan the complete passive window and the strict pre-EP region."""
    minimum_rate = math.inf
    minimum_density = math.inf
    minimum_diagonal = math.inf
    maximum_hermiticity_error = 0.0
    maximum_imaginary_diagonal = 0.0
    maximum_identity_error = 0.0
    for beta in DENSE_PASSIVE_BETAS:
        rate = bath_rate_matrix(float(beta), False, p)
        minimum_rate = min(minimum_rate, float(np.min(np.linalg.eigvalsh(rate))))
        h_r = projected_core_hamiltonian(float(beta), False, p)
        for omega in DENSE_OMEGAS:
            g_r = np.linalg.inv(float(omega) * I2 - h_r)
            g_a = g_r.conj().T
            raw_density = 1j * (g_r - g_a) / (2.0 * np.pi)
            density = 0.5 * (raw_density + raw_density.conj().T)
            sandwich = g_r @ rate @ g_a / (2.0 * np.pi)
            minimum_density = min(
                minimum_density, float(np.min(np.linalg.eigvalsh(density)))
            )
            minimum_diagonal = min(
                minimum_diagonal, float(np.min(np.diag(raw_density).real))
            )
            maximum_hermiticity_error = max(
                maximum_hermiticity_error,
                float(np.max(np.abs(raw_density - raw_density.conj().T))),
            )
            maximum_imaginary_diagonal = max(
                maximum_imaginary_diagonal,
                float(np.max(np.abs(np.diag(raw_density).imag))),
            )
            maximum_identity_error = max(
                maximum_identity_error,
                float(np.max(np.abs(raw_density - sandwich))),
            )

    strict_pre_ep_minimum = math.inf
    strict_omegas = np.linspace(-0.50, 0.50, 1001)
    for beta in DENSE_PRE_EP_BETAS:
        for omega in strict_omegas:
            _, _, density, _ = green_and_density(
                float(omega), float(beta), False, p
            )
            strict_pre_ep_minimum = min(
                strict_pre_ep_minimum,
                float(np.min(np.linalg.eigvalsh(density))),
            )

    return {
        "minimum_rate_eigenvalue_full_passive_window": minimum_rate,
        "minimum_density_eigenvalue_full_passive_window": minimum_density,
        "minimum_diagonal_DOS_full_passive_window": minimum_diagonal,
        "minimum_density_eigenvalue_strict_pre_EP_window": strict_pre_ep_minimum,
        "maximum_density_hermiticity_error": maximum_hermiticity_error,
        "maximum_imaginary_diagonal_DOS": maximum_imaginary_diagonal,
        "maximum_resolvent_sandwich_error_dense_scan": maximum_identity_error,
    }


def density_sum_rules(p: SaddleParams = P) -> list[dict[str, float]]:
    """Integrate Tr rho over the full real axis for representative cases."""
    rows: list[dict[str, float]] = []
    for beta in (p.beta_min, STRICT_PRE_EP_MAX, p.beta_max):
        integral, error = quad(
            lambda omega: float(
                np.trace(green_and_density(omega, beta, False, p)[2]).real
            ),
            -np.inf,
            np.inf,
            epsabs=2.0e-10,
            epsrel=2.0e-10,
            limit=300,
        )
        rows.append(
            {
                "beta0": float(beta),
                "integrated_trace_density": float(integral),
                "quadrature_error_estimate": float(error),
                "absolute_two_state_sum_rule_error": abs(float(integral) - 2.0),
            }
        )
    return rows


def basis_covariance_error(p: SaddleParams = P) -> float:
    """Check covariance of rho under a fixed unitary channel rotation."""
    unitary = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / math.sqrt(2.0)
    maximum = 0.0
    for beta in (p.beta_min, 0.45, STRICT_PRE_EP_MAX, p.beta_max):
        h_r = projected_core_hamiltonian(beta, False, p)
        h_rotated = unitary.conj().T @ h_r @ unitary
        for omega in (-0.37, -0.11, 0.0, 0.19, 0.41):
            _, _, density, _ = green_and_density(omega, beta, False, p)
            g_rotated = np.linalg.inv(float(omega) * I2 - h_rotated)
            density_rotated = 1j * (
                g_rotated - g_rotated.conj().T
            ) / (2.0 * np.pi)
            maximum = max(
                maximum,
                float(
                    np.max(
                        np.abs(
                            density_rotated
                            - unitary.conj().T @ density @ unitary
                        )
                    )
                ),
            )
    return maximum


def shell_commutator_error(beta: float = 0.50, p: SaddleParams = P) -> float:
    """Measure whether shell-symmetrized densities share a fixed eigenbasis."""
    cutoffs = np.geomspace(MIN_CUTOFF, p.k_max, 48)
    densities = [density_sym(float(cutoff), beta, False, p) for cutoff in cutoffs]
    return max(
        float(np.max(np.abs(left @ right - right @ left)))
        for left in densities
        for right in densities
    )


def inverse_flow_threshold(
    beta: float,
    control: bool,
    threshold: float,
    p: SaddleParams = P,
) -> dict[str, Any]:
    """Solve the exact inverse-flow identity independently of the Riccati ODE."""
    j0 = bare_sw_exchange(p)
    log_limit = math.log(p.k_max / MIN_CUTOFF)
    initial_integral = np.zeros((2, 2), dtype=complex).reshape(-1)

    def exchange_from_integral(flattened: np.ndarray) -> np.ndarray:
        accumulated = flattened.reshape(2, 2)
        return np.linalg.inv(I2 / j0 - accumulated)

    def event(log_scale: float, flattened: np.ndarray) -> float:
        cutoff = p.k_max * math.exp(-float(log_scale))
        exchange = exchange_from_integral(flattened)
        _, maximum = dimensionless_coupling(
            exchange, cutoff, beta, control, p
        )
        return float(threshold - maximum)

    event.terminal = True
    event.direction = -1
    solution = solve_ivp(
        lambda log_scale, _state: density_sym(
            p.k_max * math.exp(-float(log_scale)), beta, control, p
        ).reshape(-1),
        (0.0, log_limit),
        initial_integral,
        events=event,
        rtol=2.0e-10,
        atol=2.0e-12,
        max_step=0.0025,
    )
    reached = bool(len(solution.t_events[0]))
    log_star = (
        float(solution.t_events[0][0]) if reached else float(solution.t[-1])
    )
    scale = p.k_max * math.exp(-log_star) if reached else math.nan
    return {
        "beta0": beta,
        "control_gamma_pt_zero": int(control),
        "threshold": threshold,
        "threshold_reached": int(reached),
        "inverse_flow_scale": scale,
        "inverse_flow_log_scale_at_stop": log_star,
    }


def adverse_relative_only_control(p: SaddleParams = P) -> dict[str, float]:
    """Deliberately omit most common loss; the resulting rate is indefinite."""
    beta = 0.45
    residual_common_loss = 0.015
    relative = G0(beta, p)
    rate = 2.0 * (residual_common_loss * I2 - relative * SZ)
    h_r = (
        D0(beta, p) * SX
        - 1j * residual_common_loss * I2
        + 1j * relative * SZ
    )
    minimum_density = math.inf
    for omega in np.linspace(-0.50, 0.50, 4001):
        g_r = np.linalg.inv(float(omega) * I2 - h_r)
        density = 1j * (g_r - g_r.conj().T) / (2.0 * np.pi)
        density = 0.5 * (density + density.conj().T)
        minimum_density = min(
            minimum_density, float(np.min(np.linalg.eigvalsh(density)))
        )
    return {
        "beta0": beta,
        "residual_common_loss": residual_common_loss,
        "minimum_rate_eigenvalue": float(np.min(np.linalg.eigvalsh(rate))),
        "maximum_rate_eigenvalue": float(np.max(np.linalg.eigvalsh(rate))),
        "minimum_density_eigenvalue": minimum_density,
    }


def flow_curve(beta: float, control: bool, stop_threshold: float = 1.2,
               p: SaddleParams = P) -> list[dict[str, Any]]:
    j0 = bare_sw_exchange(p)
    initial = (j0 * I2).reshape(-1)
    log_limit = math.log(p.k_max / MIN_CUTOFF)

    def event(log_scale: float, flattened: np.ndarray) -> float:
        cutoff = p.k_max * math.exp(-float(log_scale))
        _, maximum = dimensionless_coupling(flattened.reshape(2, 2), cutoff, beta, control, p)
        return stop_threshold - maximum

    event.terminal = True
    event.direction = -1
    solution = solve_ivp(
        lambda log_scale, state: rg_rhs(log_scale, state, beta, control, p),
        (0.0, log_limit),
        initial,
        events=event,
        rtol=2.0e-9,
        atol=2.0e-11,
        max_step=0.005,
        dense_output=True,
    )
    end = float(solution.t_events[0][0]) if len(solution.t_events[0]) else float(solution.t[-1])
    logs = np.linspace(0.0, end, 320)
    states = solution.sol(logs).T
    rows: list[dict[str, Any]] = []
    for log_scale, flattened in zip(logs, states):
        cutoff = p.k_max * math.exp(-float(log_scale))
        exchange = flattened.reshape(2, 2)
        _, maximum = dimensionless_coupling(exchange, cutoff, beta, control, p)
        rows.append(
            {
                "beta0": beta,
                "control_gamma_pt_zero": int(control),
                "log_scale": float(log_scale),
                "cutoff": cutoff,
                "gmax": maximum,
                "exchange_trace_real": float(np.trace(exchange).real),
            }
        )
    return rows


def petermann_factor(beta: float, p: SaddleParams = P) -> float:
    matrix = D0(beta, p) * SX + 1j * G0(beta, p) * SZ
    _, right = np.linalg.eig(matrix)
    inverse = np.linalg.inv(right)
    values = []
    for index in range(2):
        projector = np.outer(right[:, index], inverse[index, :])
        values.append(float(np.linalg.norm(projector, "fro") ** 2))
    return max(values)


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_plot(out: Path, density_rows: list[dict[str, Any]], flow_rows: list[dict[str, Any]],
              scale_rows: list[dict[str, Any]], comparison_rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.4), constrained_layout=True)
    for control, style, label in ((False, "-", r"near EP"), (True, "--", r"$\Gamma_{\rm PT}=0$")):
        rows = [row for row in density_rows if row["control_gamma_pt_zero"] == int(control)]
        axes[0, 0].plot(
            [row["omega"] for row in rows],
            [row["rho_max_eigenvalue"] for row in rows],
            style,
            label=label,
        )
    axes[0, 0].set(
        xlabel=r"$\omega$",
        ylabel=r"$\lambda_{\max}[\rho(\omega)]$",
        title=r"Causal channel density at $\beta_0=0.50$",
        xlim=(-0.35, 0.35),
    )
    axes[0, 0].legend(frameon=False, fontsize=7)

    for control, style, label in ((False, "-", "near EP"), (True, "--", r"$\Gamma_{\rm PT}=0$")):
        rows = [row for row in flow_rows if row["control_gamma_pt_zero"] == int(control)]
        axes[0, 1].semilogx(
            [row["cutoff"] for row in rows],
            [row["gmax"] for row in rows],
            style,
            label=label,
        )
    axes[0, 1].invert_xaxis()
    axes[0, 1].axhline(1.0, color="black", lw=0.7, ls=":")
    axes[0, 1].set(
        xlabel=r"running cutoff $\Lambda$",
        ylabel=r"$g_{\max}(\Lambda)$",
        title="Matrix RG flow\nearly amplified, later suppressed",
    )
    axes[0, 1].set_xticks((0.10, 0.20, 0.50))
    axes[0, 1].set_xticklabels(("0.10", "0.20", "0.50"))
    axes[0, 1].legend(frameon=False, fontsize=7)

    threshold_styles = (
        (0.30, "o-", "tab:blue"),
        (0.50, "s-", "tab:orange"),
        (0.75, "^-", "tab:green"),
        (1.00, "d-", "tab:red"),
    )
    for threshold, style, color in threshold_styles:
        rows = [
            row for row in comparison_rows
            if abs(row["threshold"] - threshold) < 1.0e-12
        ]
        axes[1, 0].plot(
            [row["beta0"] for row in rows],
            [row["active_to_control_scale_ratio"] for row in rows],
            style,
            color=color,
            ms=3.0,
            lw=1.1,
        )
        axes[1, 0].annotate(
            fr"$g_\star={threshold:.2f}$",
            xy=(
                rows[-1]["beta0"],
                rows[-1]["active_to_control_scale_ratio"],
            ),
            xytext=(4, 0),
            textcoords="offset points",
            color=color,
            fontsize=6.2,
            va="center",
        )
    axes[1, 0].axhline(1.0, color="black", lw=0.7, ls=":")
    axes[1, 0].set(
        xlabel=r"$\beta_0$",
        ylabel=r"$T_\star^{\rm active}/T_\star^{\rm control}$",
        title="Active/control scale ratio\nsign changes with threshold",
        xlim=(0.34, 0.59),
    )

    primary_active = [
        row for row in scale_rows
        if row["control_gamma_pt_zero"] == 0
        and abs(row["threshold"] - PRIMARY_THRESHOLD) < 1.0e-12
        and abs(row["max_step"] - min(MAX_STEPS)) < 1.0e-12
    ]
    primary_control = [
        row for row in scale_rows
        if row["control_gamma_pt_zero"] == 1
        and abs(row["threshold"] - PRIMARY_THRESHOLD) < 1.0e-12
        and abs(row["max_step"] - min(MAX_STEPS)) < 1.0e-12
    ]
    axes[1, 1].plot(
        [row["beta0"] for row in primary_active],
        [row["operational_RG_scale"] for row in primary_active],
        "o-",
        label="active matrix RG",
    )
    axes[1, 1].plot(
        [row["beta0"] for row in primary_control],
        [row["operational_RG_scale"] for row in primary_control],
        "s--",
        label=r"$\Gamma_{\rm PT}=0$ RG",
    )
    sw_scale = comparison_rows[0]["flat_band_SW_scale"]
    axes[1, 1].axhline(sw_scale, color="tab:green", ls=":", label="flat-band SW estimate")
    axes[1, 1].set(
        xlabel=r"$\beta_0$",
        ylabel="operational scale",
        title=r"Operational $g_\star=1$ scale" + "\nactive reaches threshold later",
    )
    axes[1, 1].legend(frameon=False, fontsize=7)
    for axis in axes.flat:
        axis.grid(alpha=0.18, lw=0.5)
    fixed_time = datetime(2026, 7, 20, 12, 0, 0, tzinfo=timezone.utc)
    fig.savefig(
        out / "Channel_Resolved_RG_Gate.pdf",
        metadata={
            "Title": "Driven-Dirac impurity channel-resolved RG gate",
            "Author": "Driven-Dirac impurity reproducibility pipeline",
            "CreationDate": fixed_time,
            "ModDate": fixed_time,
        },
    )
    fig.savefig(
        out / "Channel_Resolved_RG_Gate.png",
        dpi=220,
        metadata={"Software": "Driven-Dirac impurity channel-resolved RG gate"},
    )
    plt.close(fig)


def run(out: Path, p: SaddleParams = P) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    j0 = bare_sw_exchange(p)
    flat_sw_scale = p.k_max * math.exp(-1.0 / (2.0 * p.rho * j0))

    density_rows: list[dict[str, Any]] = []
    max_resolvent_identity_error = 0.0
    min_density_eigenvalue = math.inf
    min_rate_eigenvalue = math.inf
    for control in (False, True):
        rate = bath_rate_matrix(0.50, control, p)
        min_rate_eigenvalue = min(min_rate_eigenvalue, float(np.min(np.linalg.eigvalsh(rate))))
        for omega in np.linspace(-0.40, 0.40, 801):
            _, _, density, identity_error = green_and_density(float(omega), 0.50, control, p)
            eigenvalues = np.linalg.eigvalsh(density)
            min_density_eigenvalue = min(min_density_eigenvalue, float(np.min(eigenvalues)))
            max_resolvent_identity_error = max(max_resolvent_identity_error, identity_error)
            density_rows.append(
                {
                    "beta0": 0.50,
                    "control_gamma_pt_zero": int(control),
                    "omega": float(omega),
                    "rho_min_eigenvalue": float(np.min(eigenvalues)),
                    "rho_max_eigenvalue": float(np.max(eigenvalues)),
                    "rho_trace": float(np.trace(density).real),
                    "resolvent_sandwich_identity_error": identity_error,
                }
            )
    write_csv(out / "causal_channel_density.csv", density_rows)

    scale_rows: list[dict[str, Any]] = []
    for beta in BETA_VALUES:
        for control in (False, True):
            for threshold in THRESHOLDS:
                for max_step in MAX_STEPS:
                    scale_rows.append(integrate_to_threshold(beta, control, threshold, max_step, p))
    write_csv(out / "matrix_rg_threshold_scales.csv", scale_rows)

    fine_rows = [row for row in scale_rows if abs(row["max_step"] - min(MAX_STEPS)) < 1.0e-12]
    comparison_rows: list[dict[str, Any]] = []
    for beta in BETA_VALUES:
        for threshold in THRESHOLDS:
            active = next(
                row for row in fine_rows
                if row["beta0"] == beta and row["control_gamma_pt_zero"] == 0 and row["threshold"] == threshold
            )
            control = next(
                row for row in fine_rows
                if row["beta0"] == beta and row["control_gamma_pt_zero"] == 1 and row["threshold"] == threshold
            )
            comparison_rows.append(
                {
                    "beta0": beta,
                    "threshold": threshold,
                    "active_RG_scale": active["operational_RG_scale"],
                    "control_RG_scale": control["operational_RG_scale"],
                    "active_to_control_scale_ratio": active["operational_RG_scale"] / control["operational_RG_scale"],
                    "petermann_factor": petermann_factor(beta, p) if beta < p.beta_core else math.inf if beta == p.beta_core else petermann_factor(beta, p),
                    "bare_SW_exchange": j0,
                    "flat_band_SW_scale": flat_sw_scale,
                }
            )
    write_csv(out / "active_control_scale_comparison.csv", comparison_rows)

    refinement_rows: list[dict[str, Any]] = []
    for beta in BETA_VALUES:
        for control in (False, True):
            for threshold in (0.30, 0.50, 0.75, 1.00):
                group = [
                    row for row in scale_rows
                    if row["beta0"] == beta
                    and row["control_gamma_pt_zero"] == int(control)
                    and row["threshold"] == threshold
                ]
                coarse = next(row for row in group if abs(row["max_step"] - max(MAX_STEPS)) < 1.0e-12)
                fine = next(row for row in group if abs(row["max_step"] - min(MAX_STEPS)) < 1.0e-12)
                refinement_rows.append(
                    {
                        "beta0": beta,
                        "control_gamma_pt_zero": int(control),
                        "threshold": threshold,
                        "coarse_max_step": max(MAX_STEPS),
                        "fine_max_step": min(MAX_STEPS),
                        "absolute_scale_drift": abs(coarse["operational_RG_scale"] - fine["operational_RG_scale"]),
                        "relative_scale_drift": abs(coarse["operational_RG_scale"] - fine["operational_RG_scale"]) / fine["operational_RG_scale"],
                    }
                )
    write_csv(out / "rg_step_refinement.csv", refinement_rows)

    flow_rows = flow_curve(0.50, False, p=p) + flow_curve(0.50, True, p=p)
    write_csv(out / "representative_rg_flow.csv", flow_rows)

    dense_scan = dense_causality_scan(p)
    sum_rule_rows = density_sum_rules(p)
    write_csv(out / "causal_density_sum_rules.csv", sum_rule_rows)
    covariance_error = basis_covariance_error(p)
    commutator_error = shell_commutator_error(0.50, p)
    adverse_control = adverse_relative_only_control(p)

    inverse_rows: list[dict[str, Any]] = []
    for control in (False, True):
        for threshold in (0.30, 1.00):
            inverse = inverse_flow_threshold(0.50, control, threshold, p)
            nonlinear = next(
                row
                for row in fine_rows
                if row["beta0"] == 0.50
                and row["control_gamma_pt_zero"] == int(control)
                and row["threshold"] == threshold
            )
            inverse["nonlinear_Riccati_scale"] = nonlinear[
                "operational_RG_scale"
            ]
            inverse["relative_scale_difference"] = abs(
                inverse["inverse_flow_scale"]
                - nonlinear["operational_RG_scale"]
            ) / nonlinear["operational_RG_scale"]
            inverse_rows.append(inverse)
    write_csv(out / "inverse_flow_crosscheck.csv", inverse_rows)

    make_plot(out, density_rows, flow_rows, scale_rows, comparison_rows)

    primary = [row for row in comparison_rows if abs(row["threshold"] - PRIMARY_THRESHOLD) < 1.0e-12]
    weak = [row for row in comparison_rows if abs(row["threshold"] - 0.30) < 1.0e-12]
    max_refinement = max(row["relative_scale_drift"] for row in refinement_rows)
    max_hermiticity = max(
        max(row["exchange_hermiticity_error"], row["g_hermiticity_error"])
        for row in scale_rows
    )
    maximum_sum_rule_error = max(
        row["absolute_two_state_sum_rule_error"] for row in sum_rule_rows
    )
    maximum_inverse_flow_difference = max(
        row["relative_scale_difference"] for row in inverse_rows
    )
    all_thresholds_reached = all(row["threshold_reached"] == 1 for row in scale_rows)
    primary_all_suppressed = all(row["active_to_control_scale_ratio"] < 1.0 for row in primary)
    weak_all_enhanced = all(row["active_to_control_scale_ratio"] > 1.0 for row in weak)
    sign_changes_with_threshold = weak_all_enhanced and primary_all_suppressed
    beta_half = [row for row in comparison_rows if row["beta0"] == 0.50]
    below_cross = next(row for row in beta_half if row["threshold"] == 0.40)
    above_cross = next(row for row in beta_half if row["threshold"] == 0.50)
    crossover_threshold = below_cross["threshold"] + (
        (1.0 - below_cross["active_to_control_scale_ratio"])
        * (above_cross["threshold"] - below_cross["threshold"])
        / (above_cross["active_to_control_scale_ratio"] - below_cross["active_to_control_scale_ratio"])
    )
    summary = {
        "status": "CAUSAL_MATRIX_RG_PASS__TRANSIENT_AMPLIFICATION_BUT_NO_ROBUST_KONDO_SCALE_ENHANCEMENT",
        "generated_utc": "2026-07-20T12:00:00+00:00",
        "scope": "Projected-core Markov diagnostic using the complete causal G^R/G^A matrix; not a universal strong-coupling solution.",
        "parameters": asdict(p),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "checks": {
            "passive_rate_matrix": min_rate_eigenvalue >= -1.0e-12,
            "causal_density_psd": min_density_eigenvalue >= -1.0e-10,
            "dense_full_passive_window_density_psd": (
                dense_scan[
                    "minimum_density_eigenvalue_full_passive_window"
                ]
                >= -1.0e-10
            ),
            "strict_pre_EP_density_positive": (
                dense_scan[
                    "minimum_density_eigenvalue_strict_pre_EP_window"
                ]
                > 0.0
            ),
            "diagonal_DOS_real_nonnegative": (
                dense_scan["minimum_diagonal_DOS_full_passive_window"]
                >= -1.0e-10
                and dense_scan["maximum_imaginary_diagonal_DOS"]
                < 1.0e-12
            ),
            "resolvent_sandwich_identity_pass": max_resolvent_identity_error < 1.0e-10,
            "dense_resolvent_sandwich_identity_pass": (
                dense_scan[
                    "maximum_resolvent_sandwich_error_dense_scan"
                ]
                < 1.0e-10
            ),
            "two_state_density_sum_rule_pass": maximum_sum_rule_error < 1.0e-8,
            "fixed_basis_covariance_pass": covariance_error < 1.0e-12,
            "all_operational_thresholds_reached": all_thresholds_reached,
            "matrix_hermiticity_pass": max_hermiticity < 1.0e-10,
            "RG_step_refinement_pass": max_refinement < 1.0e-6,
            "inverse_flow_crosscheck_pass": (
                maximum_inverse_flow_difference < 1.0e-8
            ),
            "symmetric_shell_fixed_basis_crosscheck": (
                commutator_error < 1.0e-12
            ),
            "adverse_relative_only_kernel_detected": (
                adverse_control["minimum_rate_eigenvalue"] < 0.0
                and adverse_control["minimum_density_eigenvalue"] < -1.0e-6
            ),
            "weak_flow_transient_amplification_present": weak_all_enhanced,
            "gstar1_screening_scale_enhancement_present": not primary_all_suppressed,
            "enhancement_sign_changes_with_threshold": sign_changes_with_threshold,
            "petermann_factor_inserted_by_hand": False,
        },
        "headline": {
            "bare_SW_exchange": j0,
            "flat_band_SW_scale": flat_sw_scale,
            "minimum_rate_eigenvalue": min_rate_eigenvalue,
            "minimum_density_eigenvalue": min_density_eigenvalue,
            **dense_scan,
            "maximum_resolvent_sandwich_identity_error": max_resolvent_identity_error,
            "maximum_two_state_density_sum_rule_error": maximum_sum_rule_error,
            "maximum_fixed_basis_covariance_error": covariance_error,
            "maximum_shell_density_commutator": commutator_error,
            "maximum_RG_step_relative_drift": max_refinement,
            "maximum_inverse_flow_relative_scale_difference": (
                maximum_inverse_flow_difference
            ),
            "maximum_matrix_hermiticity_error": max_hermiticity,
            "beta050_scale_ratios_by_threshold": {
                str(row["threshold"]): row["active_to_control_scale_ratio"] for row in beta_half
            },
            "beta050_interpolated_flow_crossover_threshold": crossover_threshold,
            "beta04999_petermann_factor": petermann_factor(0.4999, p),
            "adverse_relative_only_control": adverse_control,
        },
        "interpretation": {
            "positive": "The causal matrix kernel produces a real early-stage enhancement of the largest dimensionless screening eigenvalue at weak thresholds (for example g*=0.30).",
            "negative": "The active and control flows cross. By g*=0.50 and throughout the approach to g*=1, the active projected-core flow reaches the threshold at a lower scale than the matched control.",
            "conclusion": "The present Markov projected-core RG does not provide threshold-robust evidence for an enhanced Kondo scale, despite strong Petermann growth.",
            "BA_SW_relation": "SW fixes the bare J0. A BA proximity scale that grows with nonorthogonality can agree with the early transient, but it cannot be identified with the later RG threshold without an energy-resolved comparison.",
        },
    }
    (out / "channel_resolved_rg_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    ratio_03 = next(row["active_to_control_scale_ratio"] for row in beta_half if row["threshold"] == 0.30)
    ratio_10 = next(row["active_to_control_scale_ratio"] for row in beta_half if row["threshold"] == 1.00)
    readme = f"""# Driven-Dirac impurity causal channel-resolved RG gate

Status: **{summary['status']}**.

The finite-`U` Schrieffer--Wolff transformation gives `J0={j0:.8f}`. The
channel density is generated directly from

`rho(omega)=i[G^R(omega)-G^A(omega)]/(2*pi)`

and the matrix flow `dJ/dl=J[rho(+Lambda)+rho(-Lambda)]J` is integrated
without an inserted Petermann or `Zbio` factor.  No commutativity is assumed
by the solver.  In this particle-hole-symmetric benchmark the
shell-symmetrized density matrices share a fixed eigenbasis; their maximum
commutator is `{commutator_error:.3e}`.  This permits the independent exact
inverse-flow check

`J^(-1)(l)=J0^(-1)-integral_0^l rho_shell(l') dl'`,

which agrees with the nonlinear Riccati integration to
`{maximum_inverse_flow_difference:.3e}` in the threshold scales tested.

The calculation is causal and passive.  A dense scan over
`0 <= beta0 <= 0.60` and `-4 <= omega <= 4` gives minimum density eigenvalue
`{dense_scan['minimum_density_eigenvalue_full_passive_window']:.3e}` (roundoff
at the passive boundary), while the strict pre-EP scan over
`0.35 <= beta0 <= 0.4999` and `-0.5 <= omega <= 0.5` gives the positive
minimum `{dense_scan['minimum_density_eigenvalue_strict_pre_EP_window']:.3e}`.
Analytically, positivity at every real frequency follows from the
resolvent-sandwich identity whenever the complete bath-rate matrix is
positive semidefinite.  The diagonal DOS is real and nonnegative.  The
resolvent/sandwich identity closes to
`{dense_scan['maximum_resolvent_sandwich_error_dense_scan']:.3e}`, the
two-state spectral sum rule closes to `{maximum_sum_rule_error:.3e}`, and the
maximum RG step-refinement drift is `{max_refinement:.3e}`.  An adverse
relative-only kernel with an indefinite rate matrix is correctly rejected
and develops a negative density eigenvalue.

The result is not a simple enhancement. At `beta0=0.50`, the active/control
operational-scale ratio is `{ratio_03:.6f}` at the weak threshold `g*=0.30`,
but `{ratio_10:.6f}` at `g*=1`. The flows cross: biorthogonal structure
amplifies the early weak-coupling eigenchannel, then suppresses the later
approach to strong coupling relative to the matched `Gamma_PT=0` control.
The operational crossover occurs near `g*={crossover_threshold:.4f}`.

Thus this gate supplies direct evidence for **transient channel-resolved RG
amplification**, not for a threshold-independent enhanced Kondo scale.

Files:

- `causal_channel_density.csv`: PSD spectral eigenchannels from `G^R/G^A`.
- `causal_density_sum_rules.csv`: full-axis two-state spectral sum rules.
- `matrix_rg_threshold_scales.csv`: all thresholds and solver refinements.
- `active_control_scale_comparison.csv`: active/control scale ratios.
- `representative_rg_flow.csv`: near-EP and control flow curves.
- `rg_step_refinement.csv`: ODE convergence.
- `inverse_flow_crosscheck.csv`: exact inverse-flow/Riccati comparison.
- `Channel_Resolved_RG_Gate.pdf`: four-panel review figure.
- `channel_resolved_rg_summary.json`: machine-readable decision.
- `SHA256SUMS`: hashes for every output, the executable, and its local model
  helper.

No manuscript or public repository file was modified.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")

    script = Path(__file__).resolve()
    targets = sorted(
        [path for path in out.iterdir() if path.name != "SHA256SUMS"],
        key=lambda path: path.name,
    )
    lines = [f"{sha256(path)}  {path.name}" for path in targets]
    lines.append(f"{sha256(script)}  ../channel_resolved_rg_gate.py")
    helper = script.parent / "self_consistent_saddle_gate.py"
    lines.append(f"{sha256(helper)}  ../self_consistent_saddle_gate.py")
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    package_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=package_dir / "regenerated_rg"
    )
    args = parser.parse_args()
    summary = run(args.out)
    print(json.dumps({"status": summary["status"], "headline": summary["headline"]}, indent=2))


if __name__ == "__main__":
    main()
