#!/usr/bin/env python3
"""Certified full-k nonlinear pole EP for a rotating-drive impurity kernel.

This script retains all radial momenta and all open helicity cuts.  A remote
Hermitian auxiliary level driven with phi(t)=Omega t is eliminated exactly in
the rotating frame.  Its retarded contribution to the bath inverse is

    B(z) I + A(z) sigma_z,

where the static spin-even value B(0) is absorbed into the definition of the
bath chemical potential.  The remaining functions are

    B(z) = V^2 (z-eO)/[(z-eO)^2-Omega^2] - B_raw(0),
    A(z) = -V^2 Omega/[(z-eO)^2-Omega^2].

The exact impurity self-energy is

    Sigma(z) = W^2 int_0^K k dk/(2 pi)
               [z-B(z)-k^2-lambda k sigma_x-A(z) sigma_z]^{-1}.

The script continues every open positive-energy bath channel to the outgoing
resonance sheet, solves F=F_z=0 for a real impurity gate, checks rank and
F_zz, computes the local discriminant winding, and explicitly tracks the
exchange of the two poles around the certified point.

The resulting nonlinear pole EP is distinct from the constant nilpotent
Markov-core EP.  No Jordan-twist or Yang--Baxter conclusion is inferred for
this curved, frequency-dependent kernel.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ptdirac_two_cut_matplotlib")

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.optimize import linear_sum_assignment, root, root_scalar


I2 = np.eye(2, dtype=complex)
SX = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
SZ = np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


@dataclass(frozen=True)
class Parameters:
    cutoff: float = math.pi / 4.0
    soc: float = 0.10
    impurity_hybridization: float = 0.10
    auxiliary_energy: float = 120.0
    drive_frequency: float = 60.0
    bath_temperature: float = 0.10
    derivative_step: float = 2.0e-6
    gate_halfwidth: float = 1.0e-4
    auxiliary_vertex_halfwidth: float = 1.0e-2
    contour_samples: int = 512
    pole_samples: int = 256
    certification_tolerance: float = 1.0e-10


P = Parameters()


def encode_complex(value: complex) -> list[float]:
    value = complex(value)
    return [float(value.real), float(value.imag)]


def auxiliary_terms(
    z: complex, auxiliary_vertex: float, p: Parameters = P
) -> tuple[complex, complex]:
    """Return recentered spin-even B(z) and spin-odd A(z)."""

    denominator = (z - p.auxiliary_energy) ** 2 - p.drive_frequency**2
    raw_even = auxiliary_vertex**2 * (z - p.auxiliary_energy) / denominator
    raw_even_zero = (
        -auxiliary_vertex**2
        * p.auxiliary_energy
        / (p.auxiliary_energy**2 - p.drive_frequency**2)
    )
    odd = -auxiliary_vertex**2 * p.drive_frequency / denominator
    return raw_even - raw_even_zero, odd


def radial_roots(q: complex, odd: complex, p: Parameters = P) -> tuple[complex, complex]:
    radical = np.sqrt(p.soc**4 + 4.0 * p.soc**2 * q + 4.0 * odd**2 + 0.0j)
    return (
        q + p.soc**2 / 2.0 + radical / 2.0,
        q + p.soc**2 / 2.0 - radical / 2.0,
    )


def analytic_physical_self_energy_from_q(
    q: complex, odd: complex, p: Parameters = P
) -> np.ndarray:
    """Exact physical-sheet radial integral at effective energy q."""

    x1, x2 = radial_roots(q, odd, p)
    log1 = np.log((p.cutoff**2 - x1) / (-x1))
    log2 = np.log((p.cutoff**2 - x2) / (-x2))
    j0 = 0.5 * (
        (q - x1) * log1 / (x1 - x2)
        + (q - x2) * log2 / (x2 - x1)
    )
    jz = odd * (log1 - log2) / (2.0 * (x1 - x2))

    def radial_j(root_value: complex) -> complex:
        return -np.arctanh(p.cutoff / np.sqrt(root_value)) / np.sqrt(root_value)

    jx = p.soc * (
        x1 * radial_j(x1) - x2 * radial_j(x2)
    ) / (x1 - x2)
    prefactor = p.impurity_hybridization**2 / (2.0 * math.pi)
    return prefactor * (j0 * I2 + jx * SX + jz * SZ)


def analytic_physical_self_energy(
    z: complex, auxiliary_vertex: float, p: Parameters = P
) -> np.ndarray:
    even, odd = auxiliary_terms(z, auxiliary_vertex, p)
    return analytic_physical_self_energy_from_q(z - even, odd, p)


def quadrature_physical_self_energy(
    z: complex, auxiliary_vertex: float, p: Parameters = P
) -> np.ndarray:
    even, odd = auxiliary_terms(z, auxiliary_vertex, p)

    def integrand(momentum: float) -> np.ndarray:
        bath = (
            (momentum**2 + even) * I2
            + p.soc * momentum * SX
            + odd * SZ
        )
        return (
            p.impurity_hybridization**2
            * momentum
            / (2.0 * math.pi)
            * np.linalg.inv(z * I2 - bath)
        )

    answer = np.zeros((2, 2), dtype=complex)
    for row in range(2):
        for column in range(2):
            real = quad(
                lambda momentum: float(integrand(momentum)[row, column].real),
                0.0,
                p.cutoff,
                epsabs=2.0e-12,
                epsrel=2.0e-12,
                limit=300,
            )[0]
            imag = quad(
                lambda momentum: float(integrand(momentum)[row, column].imag),
                0.0,
                p.cutoff,
                epsabs=2.0e-12,
                epsrel=2.0e-12,
                limit=300,
            )[0]
            answer[row, column] = real + 1.0j * imag
    return answer


def open_channels(
    real_energy: float, auxiliary_vertex: float, p: Parameters = P
) -> list[dict[str, float | int]]:
    """Identify every radial root on an open real-energy bath cut."""

    even, odd = auxiliary_terms(real_energy, auxiliary_vertex, p)
    q = float((real_energy - even).real)
    odd_real = float(odd.real)
    answer: list[dict[str, float | int]] = []
    for index, root_value in enumerate(radial_roots(q + 0.0j, odd_real, p)):
        root_real = float(root_value.real)
        if abs(root_value.imag) > 1.0e-8:
            continue
        if not 1.0e-12 < root_real < p.cutoff**2 - 1.0e-12:
            continue
        momentum = math.sqrt(root_real)
        energy_offset = q - root_real
        radius = math.sqrt(p.soc**2 * root_real + odd_real**2)
        if abs(abs(energy_offset) - radius) > 1.0e-7 * (1.0 + radius):
            continue
        velocity = 2.0 * momentum + p.soc**2 * momentum / energy_offset
        answer.append(
            {
                "root_index": index,
                "velocity_sign": 1 if velocity > 0.0 else -1,
                "branch_sign": 1 if energy_offset > 0.0 else -1,
                "momentum_squared": root_real,
                "momentum": momentum,
                "velocity": velocity,
                "soc_to_quadratic_ratio": p.soc / momentum,
            }
        )
    return answer


def resonance_self_energy(
    z: complex, auxiliary_vertex: float, p: Parameters = P
) -> np.ndarray:
    """Outgoing continuation across every channel open at Re z."""

    even, odd = auxiliary_terms(z, auxiliary_vertex, p)
    q = z - even
    roots = radial_roots(q, odd, p)
    density = np.zeros((2, 2), dtype=complex)
    for channel in open_channels(float(z.real), auxiliary_vertex, p):
        root_value = roots[int(channel["root_index"])]
        momentum = np.sqrt(root_value)
        energy_offset = q - root_value
        projector = 0.5 * (
            I2 + (p.soc * momentum * SX + odd * SZ) / energy_offset
        )
        velocity = (
            2.0 * momentum + p.soc**2 * momentum / energy_offset
        )
        density += (
            momentum
            * projector
            / (2.0 * math.pi * velocity * int(channel["velocity_sign"]))
        )
    return analytic_physical_self_energy_from_q(q, odd, p) - (
        2.0j * math.pi * p.impurity_hybridization**2 * density
    )


def pauli_components(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            0.5 * np.trace(matrix),
            0.5 * (matrix[0, 1] + matrix[1, 0]),
            0.5 * (matrix[0, 0] - matrix[1, 1]),
        ],
        dtype=complex,
    )


def matrix_shift_identity_residual(p: Parameters = P) -> float:
    """Roundoff gate for the exact g0/gx SOC conversion identities."""

    maximum = 0.0
    for momentum in np.linspace(0.017, p.cutoff - 0.017, 31):
        z = 0.021 + 0.037j
        a = 1.0 / (z - momentum**2 - p.soc * momentum)
        b = 1.0 / (z - momentum**2 + p.soc * momentum)
        g0 = 0.5 * (a + b)
        gx = 0.5 * (a - b)
        scalar_left = 0.5 * (a**2 + b**2)
        scalar_right = g0**2 + gx**2
        scalar = abs(scalar_left - scalar_right) / max(
            1.0, abs(scalar_left), abs(scalar_right)
        )
        matrix = g0 * I2 + gx * SX
        sandwich_left = matrix @ SZ @ matrix
        sandwich_right = (a * b) * SZ
        sandwich = np.linalg.norm(sandwich_left - sandwich_right) / max(
            1.0, np.linalg.norm(sandwich_left), np.linalg.norm(sandwich_right)
        )
        maximum = max(maximum, scalar, sandwich)
    return float(maximum)


def upper_threshold(
    auxiliary_vertex: float, p: Parameters = P
) -> float:
    """Smallest positive driven k=0 threshold."""

    def equation(energy: float) -> float:
        even, odd = auxiliary_terms(energy, auxiliary_vertex, p)
        return float(energy - even.real - abs(odd.real))

    grid = np.linspace(-0.08, 0.08, 2001)
    values = np.asarray([equation(value) for value in grid])
    changes = np.flatnonzero(values[:-1] * values[1:] <= 0.0)
    roots: list[float] = []
    for index in changes:
        answer = root_scalar(
            equation,
            bracket=(float(grid[index]), float(grid[index + 1])),
            method="brentq",
            xtol=1.0e-14,
        )
        if answer.root > 0.0:
            roots.append(float(answer.root))
    if not roots:
        raise RuntimeError("No positive driven k=0 threshold was found")
    return min(roots)


def derivative(function, z: complex, step: float) -> np.ndarray | complex:
    return (
        function(z - 2.0 * step)
        - 8.0 * function(z - step)
        + 8.0 * function(z + step)
        - function(z + 2.0 * step)
    ) / (12.0 * step)


def reduced_data(
    coordinates: np.ndarray, p: Parameters = P, step: float | None = None
) -> dict[str, complex | float]:
    z = complex(coordinates[0], coordinates[1])
    auxiliary_vertex = float(coordinates[2])
    h = p.derivative_step if step is None else step
    components = pauli_components(resonance_self_energy(z, auxiliary_vertex, p))
    component_derivatives = derivative(
        lambda argument: pauli_components(
            resonance_self_energy(argument, auxiliary_vertex, p)
        ),
        z,
        h,
    )
    s0, sx, sz = components
    ds0, dsx, dsz = component_derivatives
    a = (sx * dsx + sz * dsz) / (1.0 - ds0)
    discriminant = a**2 - sx**2 - sz**2
    epsilon = z - s0 - a
    return {
        "z": z,
        "auxiliary_vertex": auxiliary_vertex,
        "a": a,
        "discriminant": discriminant,
        "epsilon": epsilon,
    }


def reduced_residual(
    coordinates: np.ndarray, p: Parameters = P, step: float | None = None
) -> np.ndarray:
    data = reduced_data(coordinates, p, step)
    discriminant = complex(data["discriminant"])
    epsilon = complex(data["epsilon"])
    return np.asarray(
        [discriminant.real, discriminant.imag, epsilon.imag], dtype=float
    )


def determinant(
    z: complex, epsilon: float, auxiliary_vertex: float, p: Parameters = P
) -> complex:
    return complex(
        np.linalg.det(
            (z - epsilon) * I2
            - resonance_self_energy(z, auxiliary_vertex, p)
        )
    )


def solve_ep(p: Parameters = P) -> dict[str, object]:
    initial = np.asarray([0.02036, -0.00293, 1.8332], dtype=float)
    step_results = []
    solution = initial
    for step in [2.0e-4, 1.0e-4, 5.0e-5, 2.0e-5, 1.0e-5, 5.0e-6, 2.0e-6]:
        result = root(
            lambda coordinates: reduced_residual(coordinates, p, step),
            solution,
            tol=1.0e-12,
        )
        solution = result.x
        data = reduced_data(solution, p, step)
        epsilon = complex(data["epsilon"])
        z = complex(data["z"])
        value = determinant(z, epsilon.real, solution[2], p)
        first = derivative(
            lambda argument: determinant(argument, epsilon.real, solution[2], p),
            z,
            step,
        )
        step_results.append(
            {
                "step": step,
                "coordinates": solution.tolist(),
                "epsilon": encode_complex(epsilon),
                "reduced_residual_norm": float(
                    np.linalg.norm(reduced_residual(solution, p, step))
                ),
                "F_modulus": float(abs(value)),
                "F_z_modulus": float(abs(first)),
            }
        )

    result = root(
        lambda coordinates: reduced_residual(coordinates, p, p.derivative_step),
        solution,
        tol=1.0e-12,
    )
    solution = result.x
    data = reduced_data(solution, p, p.derivative_step)
    z = complex(data["z"])
    epsilon = complex(data["epsilon"])
    auxiliary_vertex = float(data["auxiliary_vertex"])
    f_value = determinant(z, epsilon.real, auxiliary_vertex, p)
    f_z = derivative(
        lambda argument: determinant(argument, epsilon.real, auxiliary_vertex, p),
        z,
        p.derivative_step,
    )
    f_zz = derivative(
        lambda argument: derivative(
            lambda inner: determinant(inner, epsilon.real, auxiliary_vertex, p),
            argument,
            p.derivative_step,
        ),
        z,
        p.derivative_step,
    )
    kernel = (z - epsilon.real) * I2 - resonance_self_energy(
        z, auxiliary_vertex, p
    )
    singular_values = np.linalg.svd(kernel, compute_uv=False)
    certified = bool(
        result.success
        and np.linalg.norm(reduced_residual(solution, p, p.derivative_step))
        < p.certification_tolerance
        and abs(f_value) < p.certification_tolerance
        and abs(f_z) < p.certification_tolerance
        and singular_values[-1] < p.certification_tolerance
        and singular_values[0] > 100.0 * p.certification_tolerance
        and abs(f_zz) > 100.0 * p.certification_tolerance
    )
    return {
        "certified": certified,
        "coordinates": solution.tolist(),
        "z": encode_complex(z),
        "epsilon": encode_complex(epsilon),
        "auxiliary_vertex": auxiliary_vertex,
        "reduced_residual": reduced_residual(
            solution, p, p.derivative_step
        ).tolist(),
        "F": encode_complex(f_value),
        "F_z": encode_complex(f_z),
        "F_zz": encode_complex(f_zz),
        "kernel_singular_values": singular_values.tolist(),
        "open_channels": open_channels(z.real, auxiliary_vertex, p),
        "derivative_step_stability": step_results,
    }


def solve_critical_point(
    epsilon: float,
    auxiliary_vertex: float,
    initial: complex,
    p: Parameters = P,
) -> tuple[bool, complex]:
    def f_z(value: complex) -> complex:
        return complex(
            derivative(
                lambda argument: determinant(
                    argument, epsilon, auxiliary_vertex, p
                ),
                value,
                p.derivative_step,
            )
        )

    result = root(
        lambda coordinates: [
            f_z(complex(coordinates[0], coordinates[1])).real,
            f_z(complex(coordinates[0], coordinates[1])).imag,
        ],
        [initial.real, initial.imag],
        tol=1.0e-11,
    )
    return bool(result.success), complex(result.x[0], result.x[1])


def local_certificate(ep: dict[str, object], p: Parameters = P) -> dict[str, object]:
    z_ep = complex(*ep["z"])
    epsilon_ep = float(ep["epsilon"][0])
    auxiliary_vertex_ep = float(ep["auxiliary_vertex"])
    theta = np.linspace(0.0, 2.0 * math.pi, p.contour_samples + 1)
    critical_points = []
    discriminants = []
    minimum_curvature = math.inf
    current = z_ep
    critical_success = True

    def f_z(value: complex, epsilon: float, auxiliary_vertex: float) -> complex:
        return complex(
            derivative(
                lambda argument: determinant(
                    argument, epsilon, auxiliary_vertex, p
                ),
                value,
                p.derivative_step,
            )
        )

    def f_zz(value: complex, epsilon: float, auxiliary_vertex: float) -> complex:
        return complex(
            derivative(
                lambda argument: f_z(argument, epsilon, auxiliary_vertex),
                value,
                p.derivative_step,
            )
        )

    for angle in theta:
        epsilon = epsilon_ep + p.gate_halfwidth * math.cos(angle)
        auxiliary_vertex = (
            auxiliary_vertex_ep
            + p.auxiliary_vertex_halfwidth * math.sin(angle)
        )
        success, current = solve_critical_point(
            epsilon, auxiliary_vertex, current, p
        )
        critical_success = critical_success and success
        critical_points.append(current)
        discriminants.append(
            determinant(current, epsilon, auxiliary_vertex, p)
        )
        minimum_curvature = min(
            minimum_curvature, abs(f_zz(current, epsilon, auxiliary_vertex))
        )

    discriminants_array = np.asarray(discriminants)
    phase_increments = np.angle(
        discriminants_array[1:] / discriminants_array[:-1]
    )
    winding = float(np.sum(phase_increments) / (2.0 * math.pi))

    pole_theta = np.linspace(0.0, 2.0 * math.pi, p.pole_samples + 1)
    tracked_poles: list[np.ndarray] = []
    current_critical = z_ep
    pole_failures = 0
    maximum_pole_residual = 0.0
    maximum_assignment_ratio = 0.0
    for angle in pole_theta:
        epsilon = epsilon_ep + p.gate_halfwidth * math.cos(angle)
        auxiliary_vertex = (
            auxiliary_vertex_ep
            + p.auxiliary_vertex_halfwidth * math.sin(angle)
        )
        success, current_critical = solve_critical_point(
            epsilon, auxiliary_vertex, current_critical, p
        )
        if not success:
            pole_failures += 1
        local_discriminant = determinant(
            current_critical, epsilon, auxiliary_vertex, p
        )
        curvature = f_zz(current_critical, epsilon, auxiliary_vertex)
        local_split = np.sqrt(-2.0 * local_discriminant / curvature)
        poles = []
        for guess in [current_critical + local_split, current_critical - local_split]:
            result = root(
                lambda coordinates: [
                    determinant(
                        complex(coordinates[0], coordinates[1]),
                        epsilon,
                        auxiliary_vertex,
                        p,
                    ).real,
                    determinant(
                        complex(coordinates[0], coordinates[1]),
                        epsilon,
                        auxiliary_vertex,
                        p,
                    ).imag,
                ],
                [guess.real, guess.imag],
                tol=1.0e-11,
            )
            if not result.success:
                pole_failures += 1
            pole = complex(result.x[0], result.x[1])
            maximum_pole_residual = max(
                maximum_pole_residual,
                abs(determinant(pole, epsilon, auxiliary_vertex, p)),
            )
            poles.append(pole)
        poles_array = np.asarray(poles)
        if tracked_poles:
            cost_matrix = abs(
                tracked_poles[-1][:, None] - poles_array[None, :]
            )
            rows, columns = linear_sum_assignment(cost_matrix)
            ordered = np.empty(2, dtype=complex)
            ordered[rows] = poles_array[columns]
            assignment_cost = float(np.max(cost_matrix[rows, columns]))
            local_half_separation = 0.5 * min(
                abs(tracked_poles[-1][1] - tracked_poles[-1][0]),
                abs(ordered[1] - ordered[0]),
            )
            maximum_assignment_ratio = max(
                maximum_assignment_ratio,
                assignment_cost / local_half_separation,
            )
        else:
            ordered = poles_array
        tracked_poles.append(ordered)

    tracked_array = np.asarray(tracked_poles)
    rows, columns = linear_sum_assignment(
        abs(tracked_array[-1][:, None] - tracked_array[0][None, :])
    )
    permutation = np.empty(2, dtype=int)
    permutation[rows] = columns
    separation = tracked_array[:, 1] - tracked_array[:, 0]
    half_winding = float(
        np.sum(np.angle(separation[1:] / separation[:-1]))
        / (2.0 * math.pi)
    )
    certified = bool(
        critical_success
        and abs(abs(winding) - 1.0) < 1.0e-9
        and np.min(abs(discriminants_array)) > 0.0
        and np.max(abs(phase_increments)) < math.pi / 2.0
        and minimum_curvature > 100.0 * p.certification_tolerance
        and pole_failures == 0
        and maximum_assignment_ratio < 1.0
        and permutation.tolist() == [1, 0]
        and abs(abs(half_winding) - 0.5) < 1.0e-9
    )
    return {
        "certified": certified,
        "theta": theta,
        "critical_points": np.asarray(critical_points),
        "discriminants": discriminants_array,
        "winding": winding,
        "minimum_boundary_modulus": float(np.min(abs(discriminants_array))),
        "maximum_phase_increment": float(np.max(abs(phase_increments))),
        "minimum_F_zz_modulus": float(minimum_curvature),
        "pole_theta": pole_theta,
        "tracked_poles": tracked_array,
        "endpoint_permutation": permutation.tolist(),
        "half_winding": half_winding,
        "minimum_pole_separation": float(np.min(abs(separation))),
        "maximum_pole_residual": float(maximum_pole_residual),
        "maximum_assignment_to_half_separation_ratio": float(
            maximum_assignment_ratio
        ),
        "pole_failures": pole_failures,
    }


def validation(ep: dict[str, object], p: Parameters = P) -> dict[str, object]:
    z_ep = complex(*ep["z"])
    auxiliary_vertex = float(ep["auxiliary_vertex"])
    quadrature_point = z_ep.real + 0.01j
    quadrature_residual = float(
        np.linalg.norm(
            analytic_physical_self_energy(
                quadrature_point, auxiliary_vertex, p
            )
            - quadrature_physical_self_energy(
                quadrature_point, auxiliary_vertex, p
            )
        )
    )
    eta = 1.0e-8
    upper = analytic_physical_self_energy(
        z_ep.real + 1.0j * eta, auxiliary_vertex, p
    )
    lower_continued = resonance_self_energy(
        z_ep.real - 1.0j * eta, auxiliary_vertex, p
    )
    sheet_residual = float(np.linalg.norm(upper - lower_continued))
    rate_matrix = (upper.conj().T - upper) / (2.0j)
    rate_eigenvalues = np.linalg.eigvalsh(rate_matrix)
    mean_rate = float(np.mean(rate_eigenvalues))
    threshold = upper_threshold(auxiliary_vertex, p)
    even, odd = auxiliary_terms(z_ep, auxiliary_vertex, p)
    return {
        "quadrature_point": encode_complex(quadrature_point),
        "quadrature_residual": quadrature_residual,
        "sheet_eta": eta,
        "sheet_continuity_residual": sheet_residual,
        "rate_matrix_eigenvalues": rate_eigenvalues.tolist(),
        "mean_rate": mean_rate,
        "passivity_margin": float(np.min(rate_eigenvalues) / mean_rate),
        "upper_threshold": threshold,
        "threshold_gap_in_widths": float(
            (z_ep.real - threshold) / abs(z_ep.imag)
        ),
        "matrix_shift_identity_relative_residual": (
            matrix_shift_identity_residual(p)
        ),
        "auxiliary_even_at_pole": encode_complex(even),
        "auxiliary_odd_at_pole": encode_complex(odd),
        "off_shell_ratio": auxiliary_vertex
        / (p.auxiliary_energy - p.drive_frequency),
        "drive_to_cutoff_ratio": p.drive_frequency / p.cutoff,
        "thermal_suppression_exponent": (
            p.auxiliary_energy - p.drive_frequency
        )
        / p.bath_temperature,
    }


def nested_contour_checks(
    ep: dict[str, object],
    reference: dict[str, object],
    p: Parameters = P,
) -> list[dict[str, object]]:
    """Repeat the certificate on half- and double-size control loops."""

    answer: list[dict[str, object]] = []
    for factor in (0.5, 1.0, 2.0):
        if factor == 1.0:
            result = reference
        else:
            scaled = replace(
                p,
                gate_halfwidth=factor * p.gate_halfwidth,
                auxiliary_vertex_halfwidth=(
                    factor * p.auxiliary_vertex_halfwidth
                ),
            )
            result = local_certificate(ep, scaled)
        answer.append(
            {
                "radius_factor": factor,
                "certified": result["certified"],
                "winding": result["winding"],
                "minimum_boundary_modulus": result[
                    "minimum_boundary_modulus"
                ],
                "maximum_phase_increment": result[
                    "maximum_phase_increment"
                ],
                "minimum_F_zz_modulus": result[
                    "minimum_F_zz_modulus"
                ],
                "endpoint_permutation": result["endpoint_permutation"],
                "half_winding": result["half_winding"],
                "maximum_assignment_to_half_separation_ratio": result[
                    "maximum_assignment_to_half_separation_ratio"
                ],
            }
        )
    return answer


def write_outputs(
    output: Path,
    ep: dict[str, object],
    local: dict[str, object],
    nested: list[dict[str, object]],
    checks: dict[str, object],
    p: Parameters = P,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "local_discriminant_contour.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["theta", "critical_re", "critical_im", "D_re", "D_im"]
        )
        for angle, critical, value in zip(
            local["theta"], local["critical_points"], local["discriminants"]
        ):
            writer.writerow(
                [angle, critical.real, critical.imag, value.real, value.imag]
            )
    with (output / "pole_exchange_contour.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["theta", "pole_1_re", "pole_1_im", "pole_2_re", "pole_2_im"])
        for angle, poles in zip(local["pole_theta"], local["tracked_poles"]):
            writer.writerow([angle, poles[0].real, poles[0].imag, poles[1].real, poles[1].imag])

    discriminants = local["discriminants"]
    poles = local["tracked_poles"]
    z_ep = complex(*ep["z"])
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), constrained_layout=True)
    scale = np.max(abs(discriminants))
    axes[0].plot(
        discriminants.real / scale,
        discriminants.imag / scale,
        lw=1.8,
    )
    axes[0].scatter([0.0], [0.0], marker="x", s=45, color="black", zorder=3)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel(r"$\operatorname{Re}{\cal D}_{\rm loc}/\max|{\cal D}_{\rm loc}|$")
    axes[0].set_ylabel(r"$\operatorname{Im}{\cal D}_{\rm loc}/\max|{\cal D}_{\rm loc}|$")
    axes[0].set_title(f"Local degree: winding {local['winding']:.0f}")
    axes[0].grid(alpha=0.25)

    axes[1].plot(poles[:, 0].real, poles[:, 0].imag, lw=1.8, label="tracked branch 1")
    axes[1].plot(poles[:, 1].real, poles[:, 1].imag, lw=1.8, label="tracked branch 2")
    axes[1].scatter([z_ep.real], [z_ep.imag], marker="*", s=90, color="black", label="EP")
    axes[1].set_xlabel(r"$\operatorname{Re}z$")
    axes[1].set_ylabel(r"$\operatorname{Im}z$")
    axes[1].set_title("Pole exchange on one control loop")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(alpha=0.25)
    fig.savefig(output / "full_k_two_cut_ep_certificate.pdf")
    fig.savefig(output / "full_k_two_cut_ep_certificate.png", dpi=220)
    plt.close(fig)

    serializable_local = {
        key: value
        for key, value in local.items()
        if key not in {"theta", "critical_points", "discriminants", "pole_theta", "tracked_poles"}
    }
    summary = {
        "status": "PASS" if ep["certified"] and local["certified"] else "FAIL",
        "parameters": asdict(p),
        "ep": ep,
        "local_topology": serializable_local,
        "nested_contour_checks": nested,
        "validation": checks,
        "scope": (
            "Exact for the stated full-k retarded kernel and rotating-frame auxiliary closure. "
            "The nonlinear pole EP is not identified with the constant nilpotent Markov-core EP, "
            "and no full-k Yang-Baxter statement follows."
        ),
    }
    (output / "full_k_two_cut_ep_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    report = f"""# Full-k two-cut nonlinear pole EP certificate

## Certified witness

- `lambda = {p.soc:.2f}`, `W = {p.impurity_hybridization:.2f}`,
  `K = pi/4`.
- `epsilon_O = {p.auxiliary_energy:.0f}`, `Omega = {p.drive_frequency:.0f}`.
- Auxiliary vertex `V = {ep['auxiliary_vertex']:.12f}`.
- Pole `z_EP = {ep['z'][0]:.12f} {ep['z'][1]:+.12f} i`.
- Real impurity gate `epsilon = {ep['epsilon'][0]:.12f}`.

## Algebraic gates

- `|F| = {abs(complex(*ep['F'])):.3e}`.
- `|F_z| = {abs(complex(*ep['F_z'])):.3e}`.
- `F_zz = {ep['F_zz'][0]:.12f} {ep['F_zz'][1]:+.12f} i`.
- Kernel singular values: `{ep['kernel_singular_values'][0]:.3e}` and
  `{ep['kernel_singular_values'][1]:.3e}`.
- Local-discriminant winding: `{local['winding']:.12f}`.
- Endpoint pole permutation: `{local['endpoint_permutation']}`.
- Pole half-winding: `{local['half_winding']:.12f}`.
- Maximum tracking-step/half-separation ratio:
  `{local['maximum_assignment_to_half_separation_ratio']:.6f}`.

## Causal and microscopic gates

- Analytic integral versus adaptive quadrature:
  `{checks['quadrature_residual']:.3e}`.
- Two-cut sheet-continuity residual:
  `{checks['sheet_continuity_residual']:.3e}`.
- Physical rate eigenvalues:
  `{checks['rate_matrix_eigenvalues'][0]:.8f}` and
  `{checks['rate_matrix_eigenvalues'][1]:.8f}`.
- Passivity margin `Gamma_min/Gamma_bar =
  {checks['passivity_margin']:.6f}`.
- Threshold gap in pole widths:
  `{checks['threshold_gap_in_widths']:.6f}`.
- Exact SOC matrix-shift identity residual:
  `{checks['matrix_shift_identity_relative_residual']:.3e}`.
- Off-shell ratio `V/(epsilon_O-Omega) = {checks['off_shell_ratio']:.5f}`.
- `Omega/K = {checks['drive_to_cutoff_ratio']:.3f}`.
- Empty-auxiliary thermal exponent `(epsilon_O-Omega)/T_b =
  {checks['thermal_suppression_exponent']:.0f}`.

The exact frequency dependence of both the spin-even and spin-odd auxiliary
terms is retained.  The constant value of the spin-even principal-value term
at the Fermi energy is absorbed into the bath chemical potential; its
remaining value at the pole is reported in the JSON output.

## Interpretation

This is a genuine nonlinear continuum pole EP: `F=F_z=0`, the nonlinear
kernel has geometric multiplicity one, `F_zz` is nonzero, the local degree is
nonzero, and the two poles exchange on a closed control loop.  It establishes
a controlled microscopic possibility for the full curved momentum-summed
kernel.

It is **not** the nilpotent constant-matrix EP used by the Jordan-twist
construction.  No full-k YBE, RLL, RTT, or many-body Jordan amplification is
inferred from this certificate.
"""
    (output / "FULL_K_TWO_CUT_EP_CERTIFICATE.md").write_text(
        report, encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("two_cut_output"))
    parser.add_argument(
        "--soc",
        type=float,
        default=0.05,
        help=(
            "SOC value for the standalone certificate.  The default is the "
            "better-conditioned representative point of the continued family."
        ),
    )
    arguments = parser.parse_args()
    parameters = replace(P, soc=float(arguments.soc))
    ep = solve_ep(parameters)
    local = local_certificate(ep, parameters)
    nested = nested_contour_checks(ep, local, parameters)
    checks = validation(ep, parameters)
    write_outputs(
        arguments.out.resolve(), ep, local, nested, checks, parameters
    )
    print(
        json.dumps(
            {
                "ep_certified": ep["certified"],
                "topology_certified": local["certified"],
                "z": ep["z"],
                "epsilon": ep["epsilon"],
                "auxiliary_vertex": ep["auxiliary_vertex"],
                "winding": local["winding"],
                "endpoint_permutation": local["endpoint_permutation"],
                "sheet_residual": checks["sheet_continuity_residual"],
                "rate_eigenvalues": checks["rate_matrix_eigenvalues"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
