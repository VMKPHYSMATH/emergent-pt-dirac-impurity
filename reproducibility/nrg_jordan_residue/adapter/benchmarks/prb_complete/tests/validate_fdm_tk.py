#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import py_compile
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "benchmarks" / "prb_complete" / "scripts"


def write_thermo(path: Path, shift: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temps = [10 ** (-6 + i * 6 / 120) for i in range(121)]
    with path.open("w", newline="", encoding="utf-8") as f:
        fields = ["temperature", "entropy_full", "entropy_bath", "entropy_impurity",
                  "heat_capacity_full", "heat_capacity_bath", "heat_capacity_impurity"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for T in temps:
            # Smooth screened-to-local-moment entropy crossing near 1e-3.
            S = math.log(2.0) * (T / (T + 1e-3 * math.exp(shift)))
            w.writerow({"temperature": T, "entropy_full": S + 2.0,
                        "entropy_bath": 2.0, "entropy_impurity": S,
                        "heat_capacity_full": 0.0, "heat_capacity_bath": 0.0,
                        "heat_capacity_impurity": 0.0})


def write_summary(path: Path, valid: bool, tk: float, imag: float = 1e-12) -> None:
    text = f'''method = "Anders-Schiller complete-basis FDM thermodynamics"
criterion = "S_imp(T_K)=entropy_target"
entropy_target = {0.5*math.log(2.0)}
spectrum_real = {str(valid).lower()}
max_centered_imaginary_energy = {imag}
density_matrix_normalization_error = 0.0
complete_basis_count_pass = true
complete_basis_count = "4096"
complete_basis_target = "4096"
low_temperature_entropy = 0.0
maximum_impurity_entropy = {math.log(2.0)}
crossing_log_slope = 0.2
tk_valid = {str(valid).lower()}
T_K_entropy = {tk if valid else 'nan'}
'''
    path.write_text(text, encoding="utf-8")


def main() -> None:
    for script in [SCRIPTS / "run_fdm_tk_scan.py", SCRIPTS / "extract_fdm_tk.py"]:
        py_compile.compile(str(script), doraise=True)

    source = (ROOT / "src" / "PTDiracNHNRG.jl").read_text(encoding="utf-8")
    required = ["struct DiscardedShell", "struct FDMThermoResult",
                "complete_basis_thermodynamics", "complete_basis_count_pass",
                "environment_sites * log(Float64(LOCAL_DIM))",
                "S_imp(T_K)=entropy_target"]
    for marker in required:
        assert marker in source, marker

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "scan"
        out = root / "analysis"
        runs = []
        for beta, z, valid, tk in [(0.4, 0.25, True, 1.0e-3),
                                   (0.4, 0.75, True, 1.1e-3),
                                   (0.51, 0.5, False, math.nan)]:
            run = root / "runs" / f"beta_{beta:.6f}" / f"z_{z:.3f}"
            run.mkdir(parents=True, exist_ok=True)
            write_summary(run / "fdm_tk_summary.toml", valid, tk,
                          imag=1e-12 if valid else 1e-2)
            write_thermo(run / "fdm_thermodynamics.csv", shift=z-0.5)
            runs.append({"beta0": beta, "z_shift": z, "output": str(run), "config": "x"})
        (root / "fdm_tk_manifest.json").write_text(
            json.dumps({"profile": "test", "beta_EP": 0.5, "runs": runs}),
            encoding="utf-8")
        subprocess.run(["python3", str(SCRIPTS / "extract_fdm_tk.py"), str(root),
                        "--out", str(out)], check=True, cwd=ROOT)
        rows = list(csv.DictReader((out / "fdm_tk_vs_beta0.csv").open()))
        r04 = next(r for r in rows if abs(float(r["beta0"])-0.4) < 1e-12)
        r051 = next(r for r in rows if abs(float(r["beta0"])-0.51) < 1e-12)
        assert int(r04["valid_z_count"]) == 2
        assert math.isfinite(float(r04["T_K_entropy_zavg"]))
        assert int(r051["valid_z_count"]) == 0
        assert (out / "FDM_TK_entropy_vs_beta0.pdf").exists()
        assert (out / "FDM_TK_AUDIT.md").exists()

    # Config generation must exclude exact beta_EP and emit thermodynamics.
    dry = Path(tempfile.mkdtemp()) / "dry"
    subprocess.run(["python3", str(SCRIPTS / "run_fdm_tk_scan.py"),
                    "--out", str(dry), "--profile", "smoke", "--dry-run"],
                   check=True, cwd=ROOT)
    configs = list((dry / "generated_configs").glob("*.toml"))
    assert configs
    joined = "\n".join(p.read_text() for p in configs)
    assert "[thermodynamics]" in joined and "enabled = true" in joined
    assert "beta0 = 0.5\n" not in joined
    assert "delta_coh = 0" in joined

    print("SELF-TEST PASS: complete-basis/FDM entropy T_K extraction")


if __name__ == "__main__":
    main()
