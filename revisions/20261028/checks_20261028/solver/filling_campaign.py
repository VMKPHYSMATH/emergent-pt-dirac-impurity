#!/usr/bin/env python3
"""Interaction/filling grids and finite-chain self-energy EP continuation.

Python 3.10+, numpy/scipy/matplotlib. All energies are in the v5 model units.
The Hermitian radial model defines the equilibrium reference state explicitly.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time
import csv
import platform
import scipy

import numpy as np
from scipy.optimize import root, least_squares, linear_sum_assignment

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor"))
import full_k_two_cut_ep_certificate as fullk
import full_k_wilson_preflight as mapping
import exact_block_chain_response as exact
import charge_solver as charge


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Reject non-finite data rather than emit invalid JSON disguised as results.
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def parameters(model, width=None):
    return replace(fullk.P, soc=model["soc"], cutoff=model["cutoff"],
                   auxiliary_energy=model["auxiliary_energy"],
                   drive_frequency=model["drive_frequency"],
                   impurity_hybridization=width or model["W"])


def make_chains(config, mu, cache, archived=False):
    if archived:
        manifest = json.loads((ROOT / "reference/submitted_chains/z_chain_manifest.json").read_text())
        chains = []
        for item in manifest["chains"]:
            directory = ROOT / "reference/submitted_chains"
            onsite, hopping, coupling = exact.read_chain(directory / item["chain"], directory / item["summary"])
            chains.append(([(a - mu * np.eye(2)) for a in onsite], hopping, coupling,
                           {"source": "archived submitted low-band chain", "z_shift": item["z_shift"]}))
        return chains
    options = {"model": config["model"], "mapping": config["mapping"], "mu": mu}
    key = signature(options)
    cache_path = cache / f"chain_{key}.npz"
    report_path = cache / f"chain_{key}.json"
    if cache_path.exists() and report_path.exists():
        data = np.load(cache_path)
        reports = json.loads(report_path.read_text())["reports"]
        return [(list(data[f"a{i}"]), list(data[f"b{i}"]), data[f"v{i}"], r)
                for i, r in enumerate(reports)]
    model, opts = config["model"], config["mapping"]
    p = parameters(model)
    settings = replace(mapping.Settings(), soc=p.soc, impurity_hybridization=p.impurity_hybridization,
                       auxiliary_vertex=model["VO"], quadrature_nodes=opts["quadrature_nodes"],
                       low_bands_only=opts["low_bands_only"])
    energies, couplings = mapping.microscopic_modes(p, settings)
    # Logarithmic bins must be centered at the actual Fermi energy. Merely
    # shifting a previously discretized chain misses the new low-energy window.
    energies = energies - mu
    chains, reports, arrays = [], [], {}
    points = [complex(-.004, .003), complex(0, .003), complex(.004, .003), complex(.012 - mu, .003)]
    for i, z_shift in enumerate(opts["z_shifts"]):
        e, v = mapping.logarithmic_star(energies, couplings, opts["Lambda"], z_shift,
                                        opts["intervals"], opts["rank_tolerance"])
        v0, onsite, hopping, orthogonality = mapping.block_lanczos(e, v, opts["blocks"])
        if orthogonality > 1e-8:
            raise RuntimeError("block-Lanczos orthogonality gate failed")
        errors = []
        for z in points:
            physical = fullk.analytic_physical_self_energy(z + mu, model["VO"], p)
            star = mapping.star_hybridization(z, e, v)
            chain_value = mapping.chain_hybridization(z, v0, onsite, hopping)
            errors.append({"z_relative_to_mu": [z.real, z.imag],
                           "star_to_continuum_relative_error": mapping.relative_error(physical, star),
                           "chain_to_continuum_relative_error": mapping.relative_error(physical, chain_value)})
        report = {"source": "new microscopic Hermitian matrix chain", "mu": mu,
                  "Lambda": opts["Lambda"], "z_shift": z_shift, "blocks": len(onsite),
                  "low_bands_only": opts["low_bands_only"],
                  "orthogonality_error": orthogonality, "mapping_checks": errors}
        chains.append((onsite, hopping, v0, report))
        reports.append(report)
        arrays.update({f"a{i}": np.asarray(onsite), f"b{i}": np.asarray(hopping), f"v{i}": v0})
    cache.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **arrays)
    write_json(report_path, {"options": options, "reports": reports})
    return chains


def build_states(config, chains, epsilon, width, interaction, sites):
    states = []
    mu = config.get("current_mu", 0.)
    for onsite, hopping, coupling, report in chains:
        if sites > len(onsite):
            raise ValueError(f"requested {sites} sites, mapping has only {len(onsite)}")
        h = exact.build_one_body(onsite, hopping, coupling, epsilon - mu, width,
                                 config["model"]["W"], sites)
        states.append(charge.solve(h, interaction, config["solver"]["max_sector_dimension"],
                                   config["solver"]["ground_tolerance"]))
    return states


def averaged_response(states, z):
    values = [charge.response(s, z) for s in states]
    return tuple(sum(v[j] for v in values) / len(values) for j in range(3))


def interaction_sigma(states, z):
    if states[0].interaction == 0:
        return np.zeros((2, 2), dtype=complex)
    green, composite, sumrule = averaged_response(states, z)
    if np.linalg.norm(sumrule - np.eye(2)) > 1e-9:
        raise RuntimeError("z-averaged anticommutator sum rule failed")
    if np.linalg.cond(green) > 1e12:
        raise RuntimeError("ill-conditioned z-averaged impurity Green function")
    return np.linalg.solve(green.T, (states[0].interaction * composite).T).T


def fillings(states):
    values = [charge.observables(s) for s in states]
    fields = ["nd", "double_occupancy", "local_moment_squared", "single_occupancy_probability",
              "empty_probability", "double_minus_gaussian_reference"]
    result = {f: float(np.mean([v[f] for v in values])) for f in fields}
    result.update({"nd_z_min": min(v["nd"] for v in values), "nd_z_max": max(v["nd"] for v in values),
                   "z_members": values,
                   "scope": "arithmetic z-average of finite-chain equilibrium reference occupations"})
    return result


def h_derivatives(function, z, h):
    fm2, fm1, f0, fp1, fp2 = [function(z + i*h) for i in (-2, -1, 0, 1, 2)]
    first = (fm2 - 8*fm1 + 8*fp1 - fp2) / (12*h)
    second = (-fp2 + 16*fp1 - 30*f0 + 16*fm1 - fm2) / (12*h*h)
    return f0, first, second


def kernel(config, states, z, epsilon, width, interaction_mode="dynamic"):
    import resonance_sheets
    mu, model = config["current_mu"], config["model"]
    if interaction_mode == "dynamic":
        sigma = interaction_sigma(states, z)
    elif interaction_mode == "static_hf":
        sigma = sum(charge.hartree_fock(s) for s in states) / len(states)
    else:
        raise ValueError(interaction_mode)
    return ((z + mu - epsilon) * np.eye(2)
            - resonance_sheets.bath(config,z+mu,model["VO"],width) - sigma)


def physical_diagnostics(config, states, epsilon, width):
    eta = config["solver"]["real_axis_eta"]
    rows = []
    for omega in [-.004, 0., .004, .012 - config["current_mu"]]:
        z = complex(omega, eta)
        bath = fullk.analytic_physical_self_energy(z + config["current_mu"], config["model"]["VO"], parameters(config["model"], width))
        sigma = interaction_sigma(states, z)
        effective = (z + config["current_mu"] - epsilon) * np.eye(2) - bath - sigma
        green = np.linalg.inv(effective)
        bath_rate = 1j * (bath - bath.conj().T)
        total_rate = 1j * (bath + sigma - (bath + sigma).conj().T)
        spectral = -(green - green.conj().T) / (2j * math.pi)
        rows.append({"omega_relative_to_mu": omega, "eta": eta,
                     "bath_rate_min_eigenvalue": float(np.linalg.eigvalsh(bath_rate).min()),
                     "embedded_total_rate_min_eigenvalue": float(np.linalg.eigvalsh(total_rate).min()),
                     "embedded_spectral_min_eigenvalue": float(np.linalg.eigvalsh(spectral).min())})
    return rows


def run_filling(config, chains, job):
    states = build_states(config, chains, job["epsilon_d"], job["W"], job["U"], job["sites"])
    result = {"status": "FINITE_CHAIN_OCCUPATIONS_COMPUTED", "job": job, "filling": fillings(states),
              "mapping": [c[3] for c in chains], "temperature": 0.,
              "truncated_chain_checks": truncated_checks(config, chains, job["sites"], job["W"])}
    try:
        result["physical_axis_checks"] = physical_diagnostics(config, states, job["epsilon_d"], job["W"])
        result["physical_axis_checks_pass"] = physical_checks_pass(result["physical_axis_checks"])
    except charge.SectorTooLarge as e:
        result["status"] = "FINITE_CHAIN_OCCUPATIONS_COMPUTED_RESPONSE_BLOCKED"
        result["response_block_reason"] = str(e)
    if job["U"] == 0:
        reference = [charge.gaussian_observables(s.one_body) for s in states]
        result["u0_gaussian_agreement"] = max(abs(charge.observables(s)["nd"] - r["nd"])
                                                 for s, r in zip(states, reference))
    return result


def truncated_checks(config, chains, sites, width):
    rows = []
    mu = config["current_mu"]
    for onsite, hopping, coupling, report in chains:
        points = []
        for z in [complex(0, .003), complex(.012-mu, .003)]:
            physical = fullk.analytic_physical_self_energy(z+mu, config["model"]["VO"], parameters(config["model"], width))
            value = mapping.chain_hybridization(z, coupling*(width/config["model"]["W"]),
                                               onsite[:sites], hopping[:sites-1])
            points.append({"z_relative_to_mu": [z.real, z.imag],
                           "truncated_chain_to_continuum_relative_error": mapping.relative_error(physical, value)})
        rows.append({"z_shift": report["z_shift"], "sites": sites, "points": points})
    return rows


def physical_checks_pass(rows):
    return bool(all(min(r["bath_rate_min_eigenvalue"], r["embedded_total_rate_min_eigenvalue"],
                        r["embedded_spectral_min_eigenvalue"]) >= -1e-9 for r in rows))


def run_kept(config, chains, job, nkeep):
    import kept_filling
    values, flows = [], []
    for onsite, hopping, coupling, report in chains:
        state, flow = kept_filling.run(onsite, hopping, coupling*(job["W"]/config["model"]["W"]),
            job["epsilon_d"]-job["mu"], job["U"], job["sites"], nkeep)
        values.append(kept_filling.observables(state))
        flows.append(flow)
    fields = ["nd", "double_occupancy", "local_moment_squared", "single_occupancy_probability",
              "empty_probability", "double_minus_gaussian_reference"]
    fill = {f: float(np.mean([v[f] for v in values])) for f in fields}
    fill.update(nd_z_min=min(v["nd"] for v in values), nd_z_max=max(v["nd"] for v in values), z_members=values)
    return {"status": "KEPT_SPACE_FILLING_PILOT_REQUIRES_CONVERGENCE", "job": job, "nkeep": nkeep,
            "filling": fill, "iteration_flows": flows,
            "truncated_chain_checks": truncated_checks(config, chains, job["sites"], job["W"]),
            "scope": "finite-chain iterative kept-space occupations; no FDM density matrix or complete-basis spectrum"}


def continue_ep(config, chains, job, seed=None):
    eps, width = job["epsilon_d"], job["W"]
    mu = job["mu"]
    if seed:
        z, eps, width = complex(*seed["z_relative_to_mu"]), seed["epsilon_d"], seed["W"]
    else:
        z = complex(.011972834346 - mu, -.002807216274)
    h = config["ep"]["derivative_step"]
    scale = max(config["model"]["W"]**2 / 4, 1e-4)
    history = []
    for outer in range(config["ep"]["outer_iterations"]):
        states = build_states(config, chains, eps, width, job["U"], job["sites"])
        def equations(x):
            zz, gate, w = complex(*x[:2]), x[2], x[3]
            fun = lambda value: complex(np.linalg.det(kernel(config, states, value, gate, w)))
            f, fz, _ = h_derivatives(fun, zz, h)
            return [f.real/scale**2, f.imag/scale**2, fz.real/scale, fz.imag/scale]
        solved = root(equations, [z.real, z.imag, eps, width], tol=1e-10)
        new_z, new_eps, new_w = complex(*solved.x[:2]), float(solved.x[2]), float(solved.x[3])
        if new_w < 0:
            # A scalar hybridization sign is a bath basis phase; Sigma_bath
            # and local observables depend on W^2.
            new_w = abs(new_w)
        if not (config["ep"]["W_min"] <= new_w <= config["ep"]["W_max"] and new_z.imag < -1e-7):
            bounded = least_squares(equations, [z.real, z.imag, eps, width],
                bounds=([-.05-mu, -.05, -.1, config["ep"]["W_min"]],
                        [.1-mu, -1e-7, .1, config["ep"]["W_max"]]),
                xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=400)
            if np.linalg.norm(bounded.fun) > 1e-5:
                return {"status": "EP_SEARCH_DID_NOT_CONVERGE", "job": job, "history": history,
                        "attempted_root": {"z": [new_z.real, new_z.imag], "epsilon_d": new_eps, "W": new_w},
                        "bounded_solver_residual": float(np.linalg.norm(bounded.fun)),
                        "scope": "no center found by this search; does not prove absence of an EP"}
            solved = bounded
            new_z, new_eps, new_w = complex(*solved.x[:2]), float(solved.x[2]), float(solved.x[3])
        displacement = max(abs(new_z-z), abs(new_eps-eps), abs(new_w-width))
        history.append({"iteration": outer, "root_success": bool(solved.success), "displacement": displacement,
                        "z_relative_to_mu": [new_z.real, new_z.imag], "epsilon_d": new_eps, "W": new_w})
        z, eps, width = new_z, new_eps, new_w
        if displacement < config["ep"]["center_tolerance"]:
            break
    # Rebuild at the returned controls. A frozen interaction self-energy
    # at a preceding iterate is not accepted as a self-consistent center.
    states = build_states(config, chains, eps, width, job["U"], job["sites"])
    fun = lambda value: complex(np.linalg.det(kernel(config, states, value, eps, width)))
    f, fz, fzz = h_derivatives(fun, z, h)
    _, fz_half, fzz_half = h_derivatives(fun, z, h/2)
    singular = np.linalg.svd(kernel(config, states, z, eps, width), compute_uv=False)
    channels = fullk.open_channels(z.real + mu, config["model"]["VO"], parameters(config["model"], width))
    checks = {"F_modulus": abs(f), "F_z_modulus": abs(fz), "F_zz_modulus": abs(fzz),
              "derivative_halving_F_z_difference": abs(fz-fz_half),
              "derivative_halving_F_zz_relative_difference": abs(fzz-fzz_half)/max(abs(fzz), 1e-20),
              "kernel_singular_values": singular.tolist(), "open_channel_count": len(channels)}
    pass_center = (abs(f) < 1e-9 and abs(fz) < 1e-6 and abs(fzz) > 1e-4
                   and singular[1] < 1e-6 and singular[0] > 1e-5
                   and len(channels) == config["ep"]["required_open_channels"]
                   and history[-1]["displacement"] < config["ep"]["center_tolerance"]
                   and checks["derivative_halving_F_zz_relative_difference"] < .05)
    axis_checks = physical_diagnostics(config, states, eps, width)
    pass_center = pass_center and physical_checks_pass(axis_checks)
    result = {"status": "EP_CENTER_PASS_REQUIRES_TOPOLOGY" if pass_center else "EP_CENTER_NOT_CONVERGED",
              "job": job, "history": history, "center_checks": checks,
              "center": {"z_relative_to_mu": [z.real, z.imag], "z_absolute": [z.real+mu, z.imag],
                         "epsilon_d": eps, "W": width}, "filling": fillings(states),
              "physical_axis_checks": axis_checks, "physical_axis_checks_pass": physical_checks_pass(axis_checks),
              "truncated_chain_checks": truncated_checks(config, chains, job["sites"], width),
              "scope": "finite-chain interaction self-energy embedded in analytic continuum retarded kernel"}
    sigma = interaction_sigma(states, z)
    hf = sum(charge.hartree_fock(s) for s in states) / len(states)
    result["interaction_at_pole"] = {"Sigma_U": charge.encode_matrix(sigma),
        "static_HF": charge.encode_matrix(hf), "dynamic_minus_HF_norm": float(np.linalg.norm(sigma-hf)),
        "F_with_static_HF_at_dynamic_center": abs(np.linalg.det(kernel(config, states, z, eps, width, "static_hf"))),
        "interpretation": "HF uses the same correlated finite-chain density; it is not a self-consistent HF solution"}
    if pass_center and config["ep"]["topology_samples"]:
        result["topology"] = contour(config, chains, job, result["center"])
        result["status"] = "FINITE_CHAIN_EP_TOPOLOGY_PASS_REQUIRES_CONVERGENCE" if result["topology"]["pass"] else "EP_TOPOLOGY_FAILED"
    return result


def contour(config, chains, job, center):
    import resonance_sheets
    opts = config["ep"]
    zcrit = complex(*center["z_relative_to_mu"])
    h = opts["derivative_step"]
    rows, tracked, discriminants = [], [], []
    success, assignment_ratio, pole_error = True, 0., 0.
    for angle in np.linspace(0., 2*math.pi, opts["topology_samples"]+1):
        eps = center["epsilon_d"] + opts["gate_halfwidth"] * math.cos(angle)
        width = center["W"] + opts["W_halfwidth"] * math.sin(angle)
        # Recompute the physical reference at every contour point.
        states = build_states(config, chains, eps, width, job["U"], job["sites"])
        fun = lambda value: complex(np.linalg.det(kernel(config, states, value, eps, width)))
        critical = root(lambda x: [h_derivatives(fun, complex(*x), h)[1].real,
                                    h_derivatives(fun, complex(*x), h)[1].imag], [zcrit.real, zcrit.imag], tol=1e-10)
        zcrit = complex(*critical.x)
        d, dz, dzz = h_derivatives(fun, zcrit, h)
        success = success and (bool(critical.success) or abs(dz) < 1e-7)
        split = np.sqrt(-2*d/dzz)
        poles = []
        for guess in [zcrit+split, zcrit-split]:
            solved = root(lambda x: [fun(complex(*x)).real, fun(complex(*x)).imag], [guess.real, guess.imag], tol=1e-10)
            pole = complex(*solved.x)
            err = abs(fun(pole))
            success = success and (bool(solved.success) or err < 1e-10)
            pole_error = max(pole_error, err)
            poles.append(pole)
        poles = np.asarray(poles)
        if tracked:
            _, columns = linear_sum_assignment(abs(tracked[-1][:, None]-poles[None, :]))
            poles = poles[columns]
            assignment_ratio = max(assignment_ratio, float(max(abs(poles-tracked[-1])) / max(abs(poles[0]-poles[1])/2, 1e-20)))
        tracked.append(poles)
        discriminants.append(d)
        rows.append({"angle": float(angle), "epsilon_d": eps, "W": width, "F_z_modulus": abs(dz),
                     "F_zz_modulus": abs(dzz), "discriminant": [d.real, d.imag],
                     "nd": fillings(states)["nd"],
                     "open_channels": len(fullk.open_channels(zcrit.real+job["mu"], config["model"]["VO"], parameters(config["model"], width)))})
    d = np.asarray(discriminants)
    phase = np.angle(d[1:]/d[:-1])
    winding = float(np.sum(phase)/(2*math.pi))
    poles = np.asarray(tracked)
    separation = poles[:, 0]-poles[:, 1]
    half = float(np.sum(np.angle(separation[1:]/separation[:-1]))/(2*math.pi))
    _, permutation = linear_sum_assignment(abs(poles[0][:, None]-poles[-1][None, :]))
    passed = (success and abs(abs(winding)-1) < .05 and abs(abs(half)-.5) < .05
              and permutation.tolist() == [1, 0] and max(abs(phase)) < math.pi/2
              and min(abs(d)) > 1e-12 and assignment_ratio < .5 and pole_error < 1e-9
              and min(r["F_zz_modulus"] for r in rows) > 1e-4
              and max(r["F_z_modulus"] for r in rows) < 1e-6
              and all(resonance_sheets.channel_gate(config,r["open_channels"]) for r in rows))
    return {"pass": bool(passed), "winding": winding, "half_winding": half,
            "endpoint_permutation": permutation.tolist(), "maximum_phase_step": float(max(abs(phase))),
            "minimum_boundary_modulus": float(min(abs(d))), "maximum_pole_residual": pole_error,
            "maximum_assignment_to_half_separation_ratio": assignment_ratio, "contour": rows,
            "scope": "local winding/degree and pole-exchange check, not a global zero-count theorem"}


def jobs(config, mode):
    if mode == "submitted":
        center = json.loads((ROOT / "reference/submitted_center.json").read_text())["predicted_center"]
        for sites, u in itertools.product(config["grid"]["sites"], [0., .0025]):
            yield {"U": u, "mu": 0., "epsilon_d": center["epsilon_d"] if u else .015170349783,
                   "W": center["W"] if u else .1, "sites": sites, "archived": True}
        return
    grid = config["grid"]
    offsets = [None] if mode == "ep" else grid["gate_offsets_in_Gamma_reference"]
    for mu, sites, u, offset in itertools.product(grid["mu"], grid["sites"], grid["U"], offsets):
        eps = .015170349783 if offset is None else mu-u/2+offset*config["model"]["W"]**2/4
        yield {"U": u, "mu": mu, "epsilon_d": eps, "W": config["model"]["W"],
               "sites": sites, "gate_offset_in_Gamma_reference": offset,
               "archived": bool(grid.get("use_submitted_chains", False))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pilot.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=["filling", "ep", "submitted"], default="filling")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--backend", choices=["exact", "kept"], default="exact")
    parser.add_argument("--nkeep", type=int, default=128)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-jobs", type=int)
    args = parser.parse_args()
    if args.backend == "kept" and args.mode == "ep":
        parser.error("EP continuation requires the exact response; kept-space filling is a calibration pilot only")
    config = json.loads(args.config.read_text())
    implementation_hash = signature({name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ["charge_solver.py", "filling_campaign.py", "kept_filling.py"]})
    config_hash = signature({"config": config, "implementation_hash": implementation_hash})
    planned = list(jobs(config, args.mode))
    for job in planned:
        job["id"] = signature({"config_hash": config_hash, "mode": args.mode, "backend": args.backend,
                               "nkeep": args.nkeep if args.backend == "kept" else None, **job})
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "run_manifest.json", {"config": config, "config_hash": config_hash,
        "implementation_hash": implementation_hash,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                    "platform": platform.platform()},
        "mode": args.mode, "job_count": len(planned), "jobs": planned,
        "solver": args.backend, "nkeep": args.nkeep if args.backend == "kept" else None,
        "scope": "equilibrium reference filling and finite-chain continuum-self-energy embedding"})
    if args.plan_only:
        print(f"Planned {len(planned)} jobs; none executed: {args.out / 'run_manifest.json'}")
        return
    seed_by_key = {}
    for job in planned[:args.max_jobs] if args.max_jobs else planned:
        path = args.out / "jobs" / (job["id"]+".json")
        if args.resume and path.exists():
            result = json.loads(path.read_text())
        else:
            started = time.monotonic()
            local = {**config, "current_mu": job["mu"]}
            try:
                chains = make_chains(local, job["mu"], args.out / "mapping_cache", job["archived"])
                if args.backend == "kept":
                    result = run_kept(local, chains, job, args.nkeep)
                elif args.mode == "ep":
                    result = continue_ep(local, chains, job, seed_by_key.get((job["mu"], job["sites"])))
                else:
                    result = run_filling(local, chains, job)
            except charge.SectorTooLarge as e:
                result = {"status": "RESOURCE_BLOCKED", "job": job, "reason": str(e)}
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as e:
                result = {"status": "CALCULATION_FAILED", "job": job, "reason": str(e)}
            result["elapsed_seconds"] = time.monotonic()-started
            write_json(path, result)
        if "center" in result and "PASS" in result["status"]:
            seed_by_key[(job["mu"], job["sites"])] = result["center"]
        print(f"{job['id']}: {result['status']} U={job['U']:g} mu={job['mu']:g} sites={job['sites']} "
              f"nd={result.get('filling', {}).get('nd', 'unavailable')}", flush=True)
    summarize(args.out)


def summarize(out):
    ids = {j["id"] for j in json.loads((out/"run_manifest.json").read_text())["jobs"]}
    results = [json.loads(p.read_text()) for p in sorted((out/"jobs").glob("*.json")) if p.stem in ids]
    rows = []
    for r in results:
        job = r["job"]
        center = r.get("center", {})
        row = {k: job.get(k) for k in ["id", "U", "mu", "sites", "epsilon_d", "W"]}
        row.update({"status": r["status"], "epsilon_d_at_ep": center.get("epsilon_d"), "W_at_ep": center.get("W")})
        row.update({k: r.get("filling", {}).get(k) for k in ["nd", "double_occupancy", "local_moment_squared", "nd_z_min", "nd_z_max", "double_minus_gaussian_reference"]})
        rows.append(row)
    if rows:
        with (out/"summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    write_json(out/"summary.json", {"completed_jobs": len(results), "results": rows})


if __name__ == "__main__":
    main()
