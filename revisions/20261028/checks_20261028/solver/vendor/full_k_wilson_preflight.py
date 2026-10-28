#!/usr/bin/env python3
"""Full-k matrix-hybridization Wilson-chain preflight.

This is deliberately a *bath mapping* test, not an interacting NRG result.
It reconstructs the static rotating-frame microscopic bath whose Schur
complement is used by ``full_k_two_cut_ep_certificate.py``; checks that the
static four-orbital bath reproduces the analytic 2x2 hybridization function;
performs a logarithmic matrix-star discretization; and maps the star to a
block-Wilson chain by block Lanczos.

The output is a gate for the proposed finite-U calculation.  It must pass
before the existing scalar-hybridization NH-NRG adapter is extended.  No EP
or finite-U claim is inferred from this script alone.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Settings:
    soc: float = 0.05
    impurity_hybridization: float = 0.10
    auxiliary_vertex: float = 1.2134220096136235
    quadrature_nodes: int = 8192
    wilson_lambda: float = 2.5
    wilson_intervals: int = 34
    z_shifts: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)
    lanczos_blocks: int = 28
    rank_tolerance: float = 1.0e-13
    low_bands_only: bool = False


def load_certificate_module(path: Path):
    spec = importlib.util.spec_from_file_location("full_k_certificate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load certificate source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def hermitian_sqrt(matrix: np.ndarray, inverse: bool = False) -> np.ndarray:
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.conj().T))
    cutoff = max(float(np.max(np.abs(values))), 1.0) * 1.0e-14
    if np.min(values) < -cutoff:
        raise RuntimeError(f"matrix is not positive semidefinite: {values}")
    values = np.maximum(values, 0.0)
    if inverse:
        factors = np.asarray([1.0 / math.sqrt(x) if x > cutoff else 0.0 for x in values])
    else:
        factors = np.sqrt(values)
    return (vectors * factors) @ vectors.conj().T


def microscopic_bath_matrix(momentum: float, auxiliary_vertex: float, p) -> np.ndarray:
    """Static 4x4 rotating-frame bath at fixed radial momentum.

    Ordering is (c_+, c_-, O_+, O_-).  Eliminating the O sector gives the
    recentered B(z) I + A(z) sigma_z used by the full-k certificate.
    """

    raw_even_zero = (
        -auxiliary_vertex**2
        * p.auxiliary_energy
        / (p.auxiliary_energy**2 - p.drive_frequency**2)
    )
    hcc = (
        (momentum**2 - raw_even_zero) * np.eye(2, dtype=complex)
        + p.soc * momentum * np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    )
    hoo = np.diag(
        [p.auxiliary_energy - p.drive_frequency,
         p.auxiliary_energy + p.drive_frequency]
    ).astype(complex)
    coupling = auxiliary_vertex * np.eye(2, dtype=complex)
    return np.block([[hcc, coupling], [coupling, hoo]])


def microscopic_hybridization(
    z: complex, auxiliary_vertex: float, p, nodes: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    answer = np.zeros((2, 2), dtype=complex)
    selector = np.zeros((2, 4), dtype=complex)
    selector[:, :2] = np.eye(2)
    for momentum, radial_weight in zip(nodes, weights, strict=True):
        bath = microscopic_bath_matrix(float(momentum), auxiliary_vertex, p)
        resolvent = np.linalg.inv(z * np.eye(4, dtype=complex) - bath)
        answer += (
            p.impurity_hybridization**2
            * radial_weight
            * (selector @ resolvent @ selector.conj().T)
        )
    return answer


def radial_quadrature(cutoff: float, count: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.legendre.leggauss(count)
    momentum = 0.5 * cutoff * (x + 1.0)
    dk = 0.5 * cutoff * w
    radial_weight = momentum * dk / (2.0 * math.pi)
    return momentum, radial_weight


def microscopic_modes(p, settings: Settings) -> tuple[np.ndarray, np.ndarray]:
    """Return bath eigenenergies and 2-component impurity coupling vectors."""

    nodes, weights = radial_quadrature(p.cutoff, settings.quadrature_nodes)
    energies: list[float] = []
    couplings: list[np.ndarray] = []
    for momentum, radial_weight in zip(nodes, weights, strict=True):
        bath = microscopic_bath_matrix(float(momentum), settings.auxiliary_vertex, p)
        values, vectors = np.linalg.eigh(bath)
        conduction = vectors[:2, :]
        scale = settings.impurity_hybridization * math.sqrt(float(radial_weight))
        selected = range(2) if settings.low_bands_only else range(len(values))
        for index in selected:
            value = values[index]
            energies.append(float(value))
            couplings.append(scale * conduction[:, index])
    return np.asarray(energies), np.asarray(couplings).T


def logarithmic_boundaries(maximum: float, lam: float, z_shift: float, count: int) -> np.ndarray:
    if maximum <= 0.0:
        return np.asarray([0.0])
    values = [maximum * (1.0 + 1.0e-12), maximum * lam ** (-z_shift)]
    values.extend(maximum * lam ** (-(n + z_shift)) for n in range(1, count))
    values.append(0.0)
    return np.asarray(values, dtype=float)


def interval_index(magnitude: float, boundaries: np.ndarray) -> int:
    if magnitude <= 0.0:
        return len(boundaries) - 2
    for index in range(len(boundaries) - 1):
        if boundaries[index] >= magnitude > boundaries[index + 1]:
            return index
    return len(boundaries) - 2


def compress_bin(energies: np.ndarray, couplings: np.ndarray, tolerance: float):
    s0 = couplings @ couplings.conj().T
    s1 = (couplings * energies[np.newaxis, :]) @ couplings.conj().T
    values, vectors = np.linalg.eigh(0.5 * (s0 + s0.conj().T))
    maximum = max(float(np.max(values)), 1.0)
    keep = values > tolerance * maximum
    if not np.any(keep):
        return []
    root = vectors[:, keep] * np.sqrt(values[keep])
    pinv = np.linalg.pinv(root)
    onsite = pinv @ s1 @ pinv.conj().T
    onsite = 0.5 * (onsite + onsite.conj().T)
    eps, rotation = np.linalg.eigh(onsite)
    transformed = root @ rotation
    return [(float(eps[i]), transformed[:, i]) for i in range(len(eps))]


def logarithmic_star(
    energies: np.ndarray,
    couplings: np.ndarray,
    lam: float,
    z_shift: float,
    intervals: int,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    positive_max = float(np.max(energies[energies > 0.0]))
    negative_max = float(np.max(-energies[energies < 0.0])) if np.any(energies < 0.0) else 0.0
    boundaries = {
        1: logarithmic_boundaries(positive_max, lam, z_shift, intervals),
        -1: logarithmic_boundaries(negative_max, lam, z_shift, intervals),
    }
    grouped: dict[tuple[int, int], list[int]] = {}
    for index, energy in enumerate(energies):
        sign = 1 if energy >= 0.0 else -1
        key = (sign, interval_index(abs(float(energy)), boundaries[sign]))
        grouped.setdefault(key, []).append(index)

    modes: list[tuple[float, np.ndarray]] = []
    for key in sorted(grouped):
        indices = np.asarray(grouped[key], dtype=int)
        modes.extend(compress_bin(energies[indices], couplings[:, indices], tolerance))
    star_energies = np.asarray([item[0] for item in modes], dtype=float)
    star_couplings = np.asarray([item[1] for item in modes], dtype=complex).T
    return star_energies, star_couplings


def star_hybridization(z: complex, energies: np.ndarray, couplings: np.ndarray) -> np.ndarray:
    return (couplings / (z - energies)[np.newaxis, :]) @ couplings.conj().T


def block_lanczos(
    energies: np.ndarray, couplings: np.ndarray, blocks: int
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], float]:
    """Map a matrix star to a two-orbital block chain."""

    s0 = couplings @ couplings.conj().T
    v0 = hermitian_sqrt(s0)
    q = couplings.conj().T @ hermitian_sqrt(s0, inverse=True)
    # q is already orthonormal and its gauge is tied to v0 through
    # V Q_0 = sqrt(V V^dag).  A standalone QR would rotate Q_0 without the
    # compensating inverse rotation of v0 and therefore change the star.
    basis = [q]
    onsite: list[np.ndarray] = []
    hopping: list[np.ndarray] = []
    for _ in range(min(blocks, max(1, len(energies) // 2))):
        qn = basis[-1]
        hqn = energies[:, np.newaxis] * qn
        an = qn.conj().T @ hqn
        an = 0.5 * (an + an.conj().T)
        onsite.append(an)
        residual = hqn - qn @ an
        # Two passes of full reorthogonalization keep the block Krylov basis
        # stable across the large auxiliary/conduction scale separation.
        for _pass in range(2):
            for old in basis[:-1]:
                residual -= old @ (old.conj().T @ residual)
            residual -= qn @ (qn.conj().T @ residual)
        qnext, rnext = np.linalg.qr(residual)
        singular = np.linalg.svd(rnext[:2, :2], compute_uv=False)
        if singular[-1] < 1.0e-12 * max(singular[0], 1.0):
            break
        qnext = qnext[:, :2]
        rnext = rnext[:2, :2]
        # The chain hopping from block n to n+1 is Q_n^dag H Q_{n+1}=R^dag.
        hopping.append(rnext.conj().T)
        basis.append(qnext)
    orthogonality = np.linalg.norm(
        np.concatenate(basis, axis=1).conj().T @ np.concatenate(basis, axis=1)
        - np.eye(2 * len(basis))
    )
    return v0, onsite, hopping, float(orthogonality)


def chain_hybridization(
    z: complex, v0: np.ndarray, onsite: list[np.ndarray], hopping: list[np.ndarray]
) -> np.ndarray:
    identity = np.eye(2, dtype=complex)
    tail = np.linalg.inv(z * identity - onsite[-1])
    for index in range(len(onsite) - 2, -1, -1):
        hop = hopping[index]
        tail = np.linalg.inv(z * identity - onsite[index] - hop @ tail @ hop.conj().T)
    return v0 @ tail @ v0.conj().T


def relative_error(reference: np.ndarray, value: np.ndarray) -> float:
    return float(np.linalg.norm(value - reference) / max(np.linalg.norm(reference), 1.0e-30))


def write_chain_csv(path: Path, onsite: list[np.ndarray], hopping: list[np.ndarray]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "block", "a11_re", "a11_im", "a12_re", "a12_im",
            "a21_re", "a21_im", "a22_re", "a22_im",
            "b11_re", "b11_im", "b12_re", "b12_im",
            "b21_re", "b21_im", "b22_re", "b22_im", "b_frobenius",
        ])
        zero = np.zeros((2, 2), dtype=complex)
        for index, block in enumerate(onsite):
            hop = hopping[index] if index < len(hopping) else zero
            row = [index]
            for value in block.reshape(-1):
                row.extend([float(value.real), float(value.imag)])
            for value in hop.reshape(-1):
                row.extend([float(value.real), float(value.imag)])
            row.append(float(np.linalg.norm(hop)))
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--certificate-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--nodes", type=int, default=Settings.quadrature_nodes)
    parser.add_argument("--lambda", dest="wilson_lambda", type=float, default=Settings.wilson_lambda)
    parser.add_argument("--intervals", type=int, default=Settings.wilson_intervals)
    parser.add_argument("--blocks", type=int, default=Settings.lanczos_blocks)
    parser.add_argument("--z-count", type=int, default=len(Settings.z_shifts))
    parser.add_argument("--low-bands-only", action="store_true")
    parser.add_argument("--emit-all-z-chains", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    module = load_certificate_module(args.certificate_source)
    z_shifts = tuple((index + 0.5) / args.z_count for index in range(args.z_count))
    settings = replace(
        Settings(),
        quadrature_nodes=args.nodes,
        wilson_lambda=args.wilson_lambda,
        wilson_intervals=args.intervals,
        lanczos_blocks=args.blocks,
        z_shifts=z_shifts,
        low_bands_only=args.low_bands_only,
    )
    p = replace(
        module.P,
        soc=settings.soc,
        impurity_hybridization=settings.impurity_hybridization,
    )
    nodes, weights = radial_quadrature(p.cutoff, settings.quadrature_nodes)

    upper_points = [
        0.012 + 0.003j,
        0.012 + 0.010j,
        0.030 + 0.020j,
        -0.001 + 0.005j,
    ]
    embedding_errors = []
    for z in upper_points:
        exact = module.analytic_physical_self_energy(z, settings.auxiliary_vertex, p)
        static = microscopic_hybridization(z, settings.auxiliary_vertex, p, nodes, weights)
        embedding_errors.append(relative_error(exact, static))

    energies, couplings = microscopic_modes(p, settings)
    star_reports = []
    representative = None
    all_stars = []
    star_values_by_point: list[list[np.ndarray]] = [[] for _ in upper_points]
    for z_shift in settings.z_shifts:
        star_e, star_v = logarithmic_star(
            energies,
            couplings,
            settings.wilson_lambda,
            z_shift,
            settings.wilson_intervals,
            settings.rank_tolerance,
        )
        errors = []
        for z in upper_points:
            exact = module.analytic_physical_self_energy(z, settings.auxiliary_vertex, p)
            value = star_hybridization(z, star_e, star_v)
            errors.append(relative_error(exact, value))
            star_values_by_point[len(errors) - 1].append(value)
        star_reports.append(
            {
                "z_shift": z_shift,
                "star_modes": int(len(star_e)),
                "errors": errors,
                "maximum_error": max(errors),
            }
        )
        all_stars.append((z_shift, star_e, star_v))
        if representative is None or abs(z_shift - 0.5) < abs(representative[0] - 0.5):
            representative = (z_shift, star_e, star_v)

    if representative is None:
        raise RuntimeError("z=0.5 representative star was not generated")
    _, star_e, star_v = representative
    v0, onsite, hopping, orthogonality = block_lanczos(
        star_e, star_v, settings.lanczos_blocks
    )
    chain_errors = []
    chain_star_errors = []
    for z in upper_points:
        exact = module.analytic_physical_self_energy(z, settings.auxiliary_vertex, p)
        chain = chain_hybridization(z, v0, onsite, hopping)
        star = star_hybridization(z, star_e, star_v)
        chain_errors.append(relative_error(exact, chain))
        chain_star_errors.append(relative_error(star, chain))

    hopping_norms = [float(np.linalg.norm(block)) for block in hopping]
    ratios = [
        hopping_norms[i + 1] / hopping_norms[i]
        for i in range(len(hopping_norms) - 1)
        if hopping_norms[i] > 0.0
    ]
    averaged_star_errors = []
    for index, z in enumerate(upper_points):
        exact = module.analytic_physical_self_energy(z, settings.auxiliary_vertex, p)
        averaged_star_errors.append(
            relative_error(exact, sum(star_values_by_point[index]) / len(star_values_by_point[index]))
        )
    summary = {
        "status": "PASS" if max(embedding_errors) < 2.0e-9 and max(chain_star_errors) < 2.0e-8 else "FAIL",
        "scope": (
            "Bath-mapping preflight only. This does not compute an interacting "
            "NRG Green function and does not certify an EP."
        ),
        "settings": asdict(settings),
        "static_embedding": {
            "upper_half_plane_points": [[z.real, z.imag] for z in upper_points],
            "relative_errors": embedding_errors,
            "maximum_relative_error": max(embedding_errors),
        },
        "microscopic_star": {
            "mode_count": int(len(energies)),
            "minimum_energy": float(np.min(energies)),
            "maximum_energy": float(np.max(energies)),
            "z_shift_reports": star_reports,
            "z_averaged_errors": averaged_star_errors,
            "maximum_z_averaged_error": max(averaged_star_errors),
        },
        "block_chain": {
            "blocks": len(onsite),
            "orthogonality_error": orthogonality,
            "chain_to_exact_errors": chain_errors,
            "chain_to_star_errors": chain_star_errors,
            "maximum_chain_to_star_error": max(chain_star_errors),
            "hopping_norms": hopping_norms,
            "median_successive_hopping_ratio": float(np.median(ratios)) if ratios else math.nan,
            "impurity_to_chain_matrix": [
                [[float(value.real), float(value.imag)] for value in row] for row in v0
            ],
        },
    }
    (args.out / "full_k_wilson_preflight_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    write_chain_csv(args.out / "full_k_block_wilson_chain.csv", onsite, hopping)
    if args.emit_all_z_chains:
        manifest = []
        for index, (z_shift, shifted_e, shifted_v) in enumerate(all_stars):
            shifted_v0, shifted_onsite, shifted_hopping, shifted_orthogonality = block_lanczos(
                shifted_e, shifted_v, settings.lanczos_blocks
            )
            relative = Path(f"z_{index:02d}")
            destination = args.out / relative
            destination.mkdir(parents=True, exist_ok=True)
            chain_name = "full_k_block_wilson_chain.csv"
            summary_name = "full_k_wilson_preflight_summary.json"
            write_chain_csv(destination / chain_name, shifted_onsite, shifted_hopping)
            shifted_summary = {
                "status": "PASS",
                "scope": "Single-z block-Wilson chain for z-averaged complete-basis response.",
                "settings": asdict(settings),
                "z_shift": z_shift,
                "block_chain": {
                    "blocks": len(shifted_onsite),
                    "orthogonality_error": shifted_orthogonality,
                    "impurity_to_chain_matrix": [
                        [[float(value.real), float(value.imag)] for value in row]
                        for row in shifted_v0
                    ],
                },
            }
            (destination / summary_name).write_text(json.dumps(shifted_summary, indent=2) + "\n")
            manifest.append({
                "z_shift": z_shift,
                "chain": str(relative / chain_name),
                "summary": str(relative / summary_name),
            })
        (args.out / "z_chain_manifest.json").write_text(
            json.dumps({"chains": manifest}, indent=2) + "\n"
        )
    print(json.dumps(summary, indent=2))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
