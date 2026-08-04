#!/usr/bin/env python3
from __future__ import annotations

import argparse
import cmath
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from common import (
    consecutive_crossing, load_toml, parse_run_summary, read_csv,
    sha256, write_csv,
)


def f_lambda(physics: dict[str, Any]) -> float:
    override = physics.get("f_lambda_override", "nan")
    try:
        value = float(override)
        if math.isfinite(value):
            return value
    except (TypeError, ValueError):
        pass
    model = str(physics.get("f_lambda_model", "d_soc2")).lower()
    lam = float(physics["lambda_soc"])
    kmax = float(physics["k_max"])
    if model == "d_soc2":
        return max(1.0 - (lam / kmax) ** 2, 0.0)
    if model == "gaussian":
        return math.exp(-(lam / kmax) ** 2)
    raise ValueError(f"Unknown f_lambda_model={model}")


def local_matrix_predictions(beta: float, delta: float, U: float,
                             physics: dict[str, Any]) -> dict[str, Any]:
    delta_eff = float(physics["delta0"]) + float(physics["c_delta"]) * beta
    gamma_pt = float(physics["g_gamma"]) * beta
    s2 = delta_eff ** 2 + complex(delta, gamma_pt) ** 2
    s = cmath.sqrt(s2)
    h = np.array(
        [[delta_eff, complex(delta, gamma_pt)],
         [complex(delta, gamma_pt), -delta_eff]],
        dtype=complex,
    )
    eigvals, eigvecs = np.linalg.eig(h)
    local_condition = float(np.linalg.cond(eigvecs))
    local_gap = float(abs(eigvals[0] - eigvals[1]))

    F = f_lambda(physics)
    s0_sq = delta_eff ** 2 - gamma_pt ** 2
    quartic_inside = complex(s0_sq ** 2 + 4.0 * U * U * beta * beta * F)
    seff_sq = 0.5 * (s0_sq + cmath.sqrt(quartic_inside))
    seff = cmath.sqrt(seff_sq)
    seff_gap = 2.0 * abs(seff)
    bio = beta * beta / max(abs(seff), float(physics["bio_floor"]))

    return {
        "beta0": beta,
        "delta_coh": delta,
        "U": U,
        "delta_eff": delta_eff,
        "gamma_pt": gamma_pt,
        "local_s_real": s.real,
        "local_s_imag": s.imag,
        "local_gap": local_gap,
        "local_condition": local_condition,
        "f_lambda": F,
        "quartic_seff_real": seff.real,
        "quartic_seff_imag": seff.imag,
        "quartic_gap_2absseff": seff_gap,
        "bethe_bio_proxy": bio,
        "scope_note": (
            "quartic_seff is an external frozen Bethe/SW benchmark; "
            "F(lambda) is not inserted into the NH-NRG Hamiltonian"
        ),
    }


def discover_runs(output_root: Path) -> list[Path]:
    return sorted(
        path.parent for path in output_root.glob("nhnrg/**/RUN_SUMMARY.txt")
    )


def run_label(run_dir: Path, output_root: Path) -> str:
    return run_dir.relative_to(output_root / "nhnrg").as_posix()


def summarize_transition_weights(run_dir: Path, label: str) -> dict[str, Any]:
    path = run_dir / "impurity_transition_weights.csv"
    if not path.exists():
        return {"run": label, "transition_status": "missing_patched_output"}
    rows = read_csv(path)
    by_iteration: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_iteration[int(row["iteration"])].append(row)

    iteration_rows: list[dict[str, Any]] = []
    global_min_gap = math.inf
    global_min_iteration = None
    global_pair_weight = math.nan
    for iteration, items in sorted(by_iteration.items()):
        selected = sorted(items, key=lambda r: r["total_weight_abs"], reverse=True)[:2]
        if len(selected) < 2:
            continue
        gap_rescaled = abs(complex(selected[0]["energy_real"], selected[0]["energy_imag"]) -
                           complex(selected[1]["energy_real"], selected[1]["energy_imag"]))
        scale = float(selected[0]["scale"])
        physical_gap = gap_rescaled * scale
        pair_weight = float(selected[0]["total_weight_abs"] + selected[1]["total_weight_abs"])
        iteration_rows.append({
            "run": label,
            "iteration": iteration,
            "scale": scale,
            "transition_gap_rescaled": gap_rescaled,
            "transition_gap_physical": physical_gap,
            "pair_total_weight_abs": pair_weight,
            "first_charge": selected[0]["charge"],
            "second_charge": selected[1]["charge"],
        })
        if physical_gap < global_min_gap:
            global_min_gap = physical_gap
            global_min_iteration = iteration
            global_pair_weight = pair_weight

    return {
        "run": label,
        "transition_status": "ok" if iteration_rows else "insufficient_transitions",
        "minimum_transition_gap_physical": (
            global_min_gap if math.isfinite(global_min_gap) else math.nan
        ),
        "minimum_transition_gap_iteration": global_min_iteration,
        "pair_weight_at_minimum": global_pair_weight,
        "_iteration_rows": iteration_rows,
    }


def summarize_flow(run_dir: Path, label: str, cfg: dict[str, Any]) -> dict[str, Any]:
    rows = read_csv(run_dir / "complex_level_flow.csv")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["iteration"])].append(row)

    flow_rows: list[dict[str, Any]] = []
    iterations: list[int] = []
    imag_values: list[float] = []
    for iteration, items in sorted(grouped.items()):
        max_imag = max(abs(float(row["energy_imag"])) for row in items)
        max_abs = max(abs(float(row["energy_abs"])) for row in items)
        scale = float(items[0]["scale"])
        flow_rows.append({
            "run": label,
            "iteration": iteration,
            "scale": scale,
            "max_saved_imaginary_level": max_imag,
            "max_saved_level_abs": max_abs,
            "max_residual": max(float(row["max_residual"]) for row in items),
            "biorth_error": max(float(row["biorth_error"]) for row in items),
            "min_pair_overlap": min(float(row["min_pair_overlap"]) for row in items),
        })
        iterations.append(iteration)
        imag_values.append(max_imag)

    summary: dict[str, Any] = {
        "run": label,
        "final_iteration": max(iterations),
        "final_scale": flow_rows[-1]["scale"],
        "final_max_imaginary_level": flow_rows[-1]["max_saved_imaginary_level"],
        "maximum_residual": max(row["max_residual"] for row in flow_rows),
        "maximum_biorth_error": max(row["biorth_error"] for row in flow_rows),
        "minimum_pair_overlap": min(row["min_pair_overlap"] for row in flow_rows),
        "_flow_rows": flow_rows,
    }
    count = int(cfg["scan"]["consecutive_iterations"])
    for threshold in cfg["scan"]["crossover_thresholds"]:
        threshold = float(threshold)
        crossing = consecutive_crossing(iterations, imag_values, threshold, count)
        summary[f"N_NH_{threshold:.0e}"] = crossing
        if crossing is None:
            summary[f"T_NH_{threshold:.0e}"] = math.nan
        else:
            scale = next(row["scale"] for row in flow_rows
                         if row["iteration"] == crossing)
            summary[f"T_NH_{threshold:.0e}"] = scale
    return summary


def chain_first_excitation(run_dir: Path, iteration: int, scale: float) -> float:
    rows = read_csv(run_dir / "wilson_chain.csv")
    nsites = min(iteration + 1, len(rows))
    onsite = np.array([float(rows[i]["onsite"]) for i in range(nsites)])
    hopping = np.array([
        float(rows[i]["hopping_to_next"]) for i in range(max(0, nsites - 1))
    ])
    H = np.diag(onsite)
    for i, value in enumerate(hopping):
        H[i, i + 1] = H[i + 1, i] = value
    eigenvalues = np.linalg.eigvalsh(H) / scale
    positive = eigenvalues[eigenvalues > 1.0e-7]
    return float(positive[0]) if len(positive) else math.nan


def fixed_point_summary(run_dir: Path, label: str) -> dict[str, Any]:
    rows = read_csv(run_dir / "complex_level_flow.csv")
    final_iteration = max(int(row["iteration"]) for row in rows)
    final = [row for row in rows if int(row["iteration"]) == final_iteration]
    scale = float(final[0]["scale"])
    positive = sorted(
        float(row["energy_real"]) for row in final
        if float(row["energy_real"]) > 1.0e-6
    )
    nrg_first = positive[0] if positive else math.nan
    chain_first = chain_first_excitation(run_dir, final_iteration, scale)
    return {
        "run": label,
        "final_iteration": final_iteration,
        "nrg_first_finite_excitation": nrg_first,
        "chain_first_finite_excitation": chain_first,
        "fixed_point_mismatch": (
            abs(nrg_first - chain_first)
            if math.isfinite(nrg_first) and math.isfinite(chain_first)
            else math.nan
        ),
    }


def parse_named_parameters(summary: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(summary.get("beta0", math.nan)),
        float(summary.get("Delta_coh", math.nan)),
        float(summary.get("U", math.nan)),
    )


def make_figures(analysis_dir: Path, predictions: list[dict[str, Any]],
                 flow_rows: list[dict[str, Any]],
                 transition_summaries: list[dict[str, Any]],
                 rg_dir: Path, saddle_dir: Path) -> None:
    # Figure 1: Bethe/local geometry and NH-NRG detuning flow.
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.5), constrained_layout=True)
    pdelta = sorted(
        [row for row in predictions if row["U"] == 2.0],
        key=lambda row: row["delta_coh"], reverse=True,
    )
    if pdelta:
        x = np.array([row["delta_coh"] for row in pdelta])
        axes[0, 0].loglog(x, [row["local_gap"] for row in pdelta], "o-",
                          label="local Puiseux gap")
        axes[0, 0].loglog(x, [row["quartic_gap_2absseff"] for row in pdelta], "s--",
                          label="frozen quartic gap")
        axes[0, 0].set(xlabel=r"$\Delta_{\rm coh}$", ylabel="gap",
                       title="Local and frozen-Bethe scales")
        axes[0, 0].legend(frameon=False, fontsize=8)

        axes[0, 1].loglog(x, [row["local_condition"] for row in pdelta], "o-",
                          label=r"$\kappa_{\rm loc}$")
        axes[0, 1].loglog(x, [row["bethe_bio_proxy"] for row in pdelta], "s--",
                          label=r"$\widetilde\Gamma_{\rm bio}$")
        axes[0, 1].set(xlabel=r"$\Delta_{\rm coh}$", ylabel="nonnormality proxy",
                       title="Biorthogonal growth")
        axes[0, 1].legend(frameon=False, fontsize=8)

    delta_flows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in flow_rows:
        if row["run"].startswith("delta_scan/"):
            delta_flows[row["run"]].append(row)
    for label, rows in sorted(delta_flows.items()):
        rows = sorted(rows, key=lambda r: r["iteration"])
        axes[1, 0].semilogy(
            [r["iteration"] for r in rows],
            [max(r["max_saved_imaginary_level"], 1e-18) for r in rows],
            label=label.split("/")[-1].replace("delta_", ""),
        )
    axes[1, 0].set(xlabel="Wilson iteration",
                   ylabel=r"$\max|\mathrm{Im}\,E_N^*|$",
                   title="NH-NRG detuning flow")
    if delta_flows:
        axes[1, 0].legend(frameon=False, fontsize=7)

    trows = [
        row for row in transition_summaries
        if row["run"].startswith("delta_scan/")
        and row["transition_status"] == "ok"
    ]
    trows = sorted(trows, key=lambda row: row["delta_coh"])
    if trows:
        axes[1, 1].loglog(
            [row["delta_coh"] for row in trows],
            [row["minimum_transition_gap_physical"] for row in trows],
            "o-", label="NH-NRG impurity-supported minimum",
        )
        pred_map = {row["delta_coh"]: row for row in pdelta}
        common = [row for row in trows if row["delta_coh"] in pred_map]
        axes[1, 1].loglog(
            [row["delta_coh"] for row in common],
            [pred_map[row["delta_coh"]]["local_gap"] for row in common],
            "s--", label="local pole gap",
        )
        axes[1, 1].set(xlabel=r"$\Delta_{\rm coh}$", ylabel="physical gap",
                       title="Bethe/NH-NRG gap benchmark")
        axes[1, 1].legend(frameon=False, fontsize=7)
    fig.savefig(analysis_dir / "Bethe_NHNRG_Benchmark.pdf")
    fig.savefig(analysis_dir / "Bethe_NHNRG_Benchmark.png", dpi=220)
    plt.close(fig)

    # Figure 2: Saddle, causal RG, and NH-NRG scope cross-check.
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4), constrained_layout=True)
    saddle_csv = saddle_dir / "converged_saddles.csv"
    if saddle_csv.exists():
        rows = read_csv(saddle_csv)
        axes[0].plot([r["beta0"] for r in rows], [r["r"] for r in rows], "o-")
        axes[0].set(xlabel=r"$\beta_0$", ylabel=r"$r$",
                    title=r"Low-$T$ saddle branch")
    else:
        axes[0].text(0.5, 0.5, "saddle output missing", ha="center", va="center")
        axes[0].set_axis_off()

    rg_csv = rg_dir / "active_control_scale_comparison.csv"
    if rg_csv.exists():
        rows = read_csv(rg_csv)
        for threshold in (0.30, 0.50, 1.00):
            selected = [r for r in rows if abs(r["threshold"] - threshold) < 1e-12]
            axes[1].plot(
                [r["beta0"] for r in selected],
                [r["active_to_control_scale_ratio"] for r in selected],
                "o-", label=fr"$g_\star={threshold:g}$",
            )
        axes[1].axhline(1.0, lw=0.8, ls=":")
        axes[1].set(xlabel=r"$\beta_0$", ylabel="active/control scale",
                    title="Causal RG: crossover, not universal enhancement")
        axes[1].legend(frameon=False, fontsize=7)
    else:
        axes[1].text(0.5, 0.5, "causal RG output missing", ha="center", va="center")
        axes[1].set_axis_off()

    delta_summaries = [
        row for row in transition_summaries if row["run"].startswith("delta_scan/")
    ]
    if delta_summaries:
        delta_summaries = sorted(delta_summaries, key=lambda r: r["delta_coh"])
        axes[2].loglog(
            [r["delta_coh"] for r in delta_summaries],
            [max(r.get("T_NH_1e-08", math.nan), 1e-20)
             for r in delta_summaries],
            "o-",
        )
        axes[2].set(xlabel=r"$\Delta_{\rm coh}$",
                    ylabel=r"$T_{\rm NH}$ at $10^{-8}$",
                    title="NH-NRG crossover scale")
    else:
        axes[2].text(0.5, 0.5, "NH-NRG output missing", ha="center", va="center")
        axes[2].set_axis_off()
    fig.savefig(analysis_dir / "Saddle_RG_NHNRG_Crosscheck.pdf")
    fig.savefig(analysis_dir / "Saddle_RG_NHNRG_Crosscheck.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_toml(args.config)
    output_root = args.output_root.resolve()
    analysis_dir = output_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    physics = cfg["physics"]

    predictions: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    flow_rows_all: list[dict[str, Any]] = []
    transition_iteration_rows: list[dict[str, Any]] = []
    fixed_rows: list[dict[str, Any]] = []

    for run_dir in discover_runs(output_root):
        label = run_label(run_dir, output_root)
        summary = parse_run_summary(run_dir / "RUN_SUMMARY.txt")
        beta, delta, U = parse_named_parameters(summary)
        pred = local_matrix_predictions(beta, delta, U, physics)
        pred["run"] = label
        predictions.append(pred)

        flow = summarize_flow(run_dir, label, cfg)
        flow_rows_all.extend(flow.pop("_flow_rows"))
        transition = summarize_transition_weights(run_dir, label)
        transition_iteration_rows.extend(transition.pop("_iteration_rows", []))
        fixed = fixed_point_summary(run_dir, label)
        fixed_rows.append(fixed)

        combined = {
            "run": label,
            "beta0": beta,
            "delta_coh": delta,
            "U": U,
            **flow,
            **transition,
            **fixed,
            "local_gap": pred["local_gap"],
            "local_condition": pred["local_condition"],
            "quartic_gap_2absseff": pred["quartic_gap_2absseff"],
            "bethe_bio_proxy": pred["bethe_bio_proxy"],
        }
        run_summaries.append(combined)

    write_csv(analysis_dir / "bethe_local_predictions.csv", predictions)
    write_csv(analysis_dir / "nhnrg_run_summary.csv", run_summaries)
    write_csv(analysis_dir / "nhnrg_flow_by_iteration.csv", flow_rows_all)
    write_csv(analysis_dir / "nhnrg_impurity_transition_by_iteration.csv",
              transition_iteration_rows)
    write_csv(analysis_dir / "fixed_point_comparison.csv", fixed_rows)

    make_figures(
        analysis_dir, predictions, flow_rows_all, run_summaries,
        output_root / "causal_rg", output_root / "saddle",
    )

    completed = len(run_summaries)
    stable = [
        row for row in run_summaries
        if row["maximum_biorth_error"] < 1.0e-4
        and row["minimum_pair_overlap"] > 1.0e-6
    ]
    sumrule_warning = any(
        (run_dir / "lehmann_sumrule.csv").exists()
        for run_dir in discover_runs(output_root)
    )
    decision = {
        "status": "COMPLETE" if completed else "NO_NHNRG_RUNS_FOUND",
        "completed_nhnrg_runs": completed,
        "numerically_stable_runs_at_declared_gate": len(stable),
        "core_conclusions_to_test": {
            "bethe": (
                "Compare frozen local/rapidity splitting and biorthogonal growth "
                "with iteration-resolved impurity-supported NH-NRG transitions."
            ),
            "saddle": (
                "Use the low-temperature finite-r branch only as an independent "
                "mean-field existence check; do not convert it into a Kondo enhancement."
            ),
            "causal_rg": (
                "Compare the NH-NRG crossover scale with the causal matrix RG, "
                "which may amplify early weak flow and suppress later flow."
            ),
            "infrared": (
                "Use late level towers and vanishing relative imaginary levels "
                "to classify the infrared endpoint."
            ),
        },
        "hard_scope_limits": [
            "The quartic s_eff is an external frozen Bethe/SW benchmark and is not inserted into the NH-NRG Hamiltonian.",
            "The kept-space Lehmann curve is not a complete-basis/FDM-NRG spectrum.",
            "An exact Jordan point is outside ordinary eigenvector-based NH-NRG truncation.",
            "No Petermann or biorthogonal residue multiplier is inserted into the physical RG flow or Kondo exponent.",
        ],
    }
    (analysis_dir / "benchmark_decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )

    report = [
        "# Complete PT-Dirac benchmark report",
        "",
        f"NH-NRG runs found: **{completed}**.",
        f"Runs passing the default numerical stability gate: **{len(stable)}**.",
        "",
        "## Interpretation hierarchy",
        "",
        "1. **Frozen Bethe/scattering:** local pole splitting, rapidity-like pole coordinates, and biorthogonal nonnormality.",
        "2. **Self-consistent saddle:** independent low-temperature finite-r branch and its residual checks.",
        "3. **Causal RG:** energy-resolved weak-flow crossover without a Petermann multiplier.",
        "4. **NH-NRG:** nonperturbative many-body level flow and infrared fixed-point tower.",
        "",
        "## Publication-safe conclusion template",
        "",
        "> The frozen biorthogonal Bethe construction organizes the intermediate-energy pole splitting and nonnormality. "
        "The self-consistent saddle and causal matrix RG provide independent checks of the finite-energy regime. "
        "NH-NRG determines whether those structures persist into the infrared. A common crossover trend supports the "
        "analytic organization, whereas an asymptotically real free-chain tower limits the claim to a finite-energy "
        "non-Hermitian crossover rather than a stable many-body EP or enhanced thermodynamic Kondo scale.",
        "",
        "## Important exclusions",
        "",
        "- Do not use the kept-space Lehmann curve as a physical spectral function unless its sum rule is restored.",
        "- Do not identify the Bethe biorthogonal residue with a Kondo temperature.",
        "- Do not describe a local exact EP as an NH-NRG fixed point without a Jordan-aware truncation.",
        "",
        "See the CSV files and `benchmark_decision.json` in this directory.",
    ]
    (analysis_dir / "BENCHMARK_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    targets = sorted(
        path for path in analysis_dir.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    )
    (analysis_dir / "SHA256SUMS").write_text(
        "\n".join(f"{sha256(path)}  {path.name}" for path in targets) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
