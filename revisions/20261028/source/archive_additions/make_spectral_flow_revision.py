"""Regenerate SFig3 using archived v2.1.0 matrix-flow data; corrected scope labels.
The trace-delay curves are recomputed directly from the complete Markov resolvent.
"""
from pathlib import Path
from typing import Any
from types import SimpleNamespace
from datetime import datetime, timezone
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=SimpleNamespace(beta_core=.5)

def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    converted: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {}
        for key, value in row.items():
            try:
                item[key] = float(value)
            except (TypeError, ValueError):
                item[key] = value
        converted.append(item)
    return converted

def main_figure(
    out: Path,
    rg_dir: Path,
    delay_data: list[dict[str, Any]],
) -> None:
    density = read_csv(rg_dir / "causal_channel_density.csv")
    flow = read_csv(rg_dir / "representative_rg_flow.csv")
    comparison = read_csv(rg_dir / "active_control_scale_comparison.csv")
    fig, axes = plt.subplots(2, 2, figsize=(8.35, 6.35), constrained_layout=True)

    for control, style, label in (
        (0.0, "-", r"EP-active"),
        (1.0, "--", r"matched $\Gamma_{\rm PT}=0$"),
    ):
        selected = [row for row in density if row["control_gamma_pt_zero"] == control]
        axes[0, 0].plot(
            [row["omega"] for row in selected],
            [row["rho_max_eigenvalue"] for row in selected],
            style,
            lw=1.6,
            label=label,
        )
    axes[0, 0].set(
        xlabel=r"$\omega$",
        ylabel=r"$\lambda_{\max}[\rho(\omega)]$",
        title=r"(a) Impurity spectral density, $\beta_0=0.50$",
        xlim=(-0.32, 0.32),
    )
    axes[0, 0].legend(frameon=False, fontsize=7)

    styles = {
        "active_beta_0.35": ("tab:blue", "-", r"$\beta_0=0.35$"),
        "active_beta_0.45": ("tab:orange", "-", r"$\beta_0=0.45$"),
        "active_beta_0.50": ("tab:red", "-", r"EP, $\beta_0=0.50$"),
        "matched_gammaPT_zero": ("black", "--", r"matched $\Gamma_{\rm PT}=0$"),
    }
    for case, (color, style, label) in styles.items():
        selected = [row for row in delay_data if row["case"] == case]
        axes[0, 1].plot(
            [row["omega"] for row in selected],
            [row["trace_delay"] for row in selected],
            color=color,
            ls=style,
            lw=1.45,
            label=label,
        )
    axes[0, 1].set(
        xlabel=r"$\omega$",
        ylabel=r"${\rm Tr}\,Q(\omega)$",
        title="(b) Scattering-delay coalescence profile",
        xlim=(-0.32, 0.32),
    )
    axes[0, 1].legend(frameon=False, fontsize=6.6)

    for control, style, label in (
        (0.0, "-", "EP-active"),
        (1.0, "--", r"matched $\Gamma_{\rm PT}=0$"),
    ):
        selected = [row for row in flow if row["control_gamma_pt_zero"] == control]
        axes[1, 0].semilogx(
            [row["cutoff"] for row in selected],
            [row["gmax"] for row in selected],
            style,
            lw=1.55,
            label=label,
        )
    axes[1, 0].invert_xaxis()
    axes[1, 0].axhline(1.0, color="black", lw=0.7, ls=":")
    axes[1, 0].set(
        xlabel=r"running cutoff $\Lambda$",
        ylabel=r"$g_{\max}(\Lambda)$",
        title="(c) Matrix flow diagnostic",
    )
    axes[1, 0].set_xticks((0.10, 0.20, 0.50))
    axes[1, 0].set_xticklabels(("0.10", "0.20", "0.50"))
    axes[1, 0].legend(frameon=False, fontsize=7)

    threshold_styles = (
        (0.30, "o-", "tab:blue"),
        (0.50, "s-", "tab:orange"),
        (0.75, "^-", "tab:green"),
        (1.00, "d-", "tab:red"),
    )
    for threshold, style, color in threshold_styles:
        selected = [
            row for row in comparison
            if abs(row["threshold"] - threshold) < 1.0e-12
        ]
        axes[1, 1].plot(
            [row["beta0"] for row in selected],
            [row["active_to_control_scale_ratio"] for row in selected],
            style,
            color=color,
            ms=3.1,
            lw=1.1,
        )
        axes[1, 1].annotate(
            fr"$g_\star={threshold:.2f}$",
            xy=(
                selected[-1]["beta0"],
                selected[-1]["active_to_control_scale_ratio"],
            ),
            xytext=(4, 0),
            textcoords="offset points",
            color=color,
            fontsize=6.2,
            va="center",
        )
    axes[1, 1].axhline(1.0, color="black", lw=0.7, ls=":")
    axes[1, 1].axvline(P.beta_core, color="tab:red", lw=0.8, ls="--", alpha=0.8)
    axes[1, 1].set(
        xlabel=r"$\beta_0$",
        ylabel=r"$T_\star^{\rm active}/T_\star^{\rm control}$",
        title="(d) Diagnostic threshold-scale ratio",
        xlim=(0.34, 0.59),
    )

    for axis in axes.flat:
        axis.grid(alpha=0.18, lw=0.5)
    fixed_time = datetime(2026, 7, 26, 12, 0, 0, tzinfo=timezone.utc)
    fig.savefig(
        out / "SFig3_causal_rg.pdf",
        metadata={
            "Title": "Impurity spectral-density diagnostic (not a Kondo RG)",
            "Author": "Driven-Dirac impurity reproducibility pipeline",
            "CreationDate": fixed_time,
            "ModDate": fixed_time,
        },
    )
    fig.savefig(
        out / "SFig3_causal_rg.png",
        dpi=240,
        metadata={"Software": "Driven-Dirac impurity causal scattering and RG gate"},
    )
    plt.close(fig)
if __name__ == '__main__':
    root=Path(__file__).resolve().parent.parent
    rows=[]
    sx=np.array([[0,1],[1,0]],complex);sz=np.diag([1,-1]);I=np.eye(2)
    for name,b,g in [('active_beta_0.35',.35,.07),('active_beta_0.45',.45,.09),('active_beta_0.50',.5,.1),('matched_gammaPT_zero',.5,0)]:
        for w in np.linspace(-.6,.6,2401):
            R=np.linalg.inv((w+.12j)*I-(.075+.05*b)*sx-1j*g*sz)
            rows.append({'case':name,'omega':float(w),'trace_delay':float(-2*np.trace(R).imag)})
    main_figure(root,root/'archive_additions/rg_data',rows)
