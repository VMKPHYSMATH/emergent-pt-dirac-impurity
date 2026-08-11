#!/usr/bin/env python3
"""Installation and archived-fixture smoke test for full-k N scaling."""

from __future__ import annotations

import json
import os
import py_compile
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parents[2]
PRB_ROOT = TEST_DIR.parent
FULLK = PRB_ROOT / "fullk_matrix_chain"
RUNNER = PRB_ROOT / "scripts" / "run_fullk_n_scaling.py"
ANALYZER = PRB_ROOT / "scripts" / "analyze_fullk_n_scaling.py"


def main() -> None:
    required = [
        RUNNER,
        ANALYZER,
        FULLK / "full_k_two_cut_ep_certificate.py",
        FULLK / "full_k_two_cut_ep_summary.json",
        FULLK / "reference_center_U00025.json",
        FULLK / "core" / "exact_zavg_block_chain_response.py",
        FULLK / "core" / "fullk_nrg_ep_predictor.py",
        FULLK / "core" / "fullk_nrg_contour_pilot.py",
        FULLK / "core" / "preflight_lowbands_z4" / "z_chain_manifest.json",
        FULLK / "core" / "preflight_lowbands_z8_l15" / "z_chain_manifest.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing installed files:\n" + "\n".join(missing))

    for path in (RUNNER, ANALYZER):
        py_compile.compile(str(path), doraise=True)

    with tempfile.TemporaryDirectory(prefix="fullk_n_scaling_test_") as temporary:
        temp = Path(temporary)
        dry = temp / "dry"
        subprocess.run(
            [
                sys.executable,
                str(RUNNER),
                "--out", str(dry),
                "--n-values", "8", "10",
                "--mode", "centers",
                "--dry-run",
            ],
            cwd=PROJECT_ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        dry_manifest = json.loads((dry / "fullk_n_scaling_manifest.json").read_text())
        if dry_manifest.get("status") != "DRY_RUN" or len(dry_manifest.get("records", [])) != 2:
            raise SystemExit("dry-run manifest validation failed")

        fixture = temp / "fixture"
        center_dir = fixture / "N_12" / "center"
        contour_dir = fixture / "N_12" / "common_contour"
        center_dir.mkdir(parents=True)
        contour_dir.mkdir(parents=True)
        shutil.copy2(
            FULLK / "reference_center_U00025.json",
            center_dir / "fullk_nrg_ep_predictor_summary.json",
        )
        shutil.copy2(
            FULLK / "core" / "exact_zavg4_contour_u00025_s12_n64" /
            "fullk_nrg_contour_pilot_summary.json",
            contour_dir / "fullk_nrg_contour_pilot_summary.json",
        )
        subprocess.run(
            [sys.executable, str(ANALYZER), str(fixture)],
            cwd=PROJECT_ROOT,
            env={**os.environ, "MPLCONFIGDIR": str(temp / "matplotlib")},
            check=True,
            stdout=subprocess.DEVNULL,
        )
        summary = json.loads(
            (fixture / "analysis" / "fullk_n_scaling_summary.json").read_text()
        )
        if not summary.get("gates", {}).get("all_centers_pass"):
            raise SystemExit("archived center fixture did not pass")
        if not summary.get("gates", {}).get("all_available_topology_gates_pass"):
            raise SystemExit("archived contour fixture did not pass")

    print("PASS: full-k matrix-chain N-scaling extension is installed and self-tested")


if __name__ == "__main__":
    main()
