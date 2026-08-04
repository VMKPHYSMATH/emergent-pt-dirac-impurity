#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[3]
runner = root / "benchmarks/prb_complete/scripts/run_jordan_comprehensive_scan.jl"
analyzer = root / "benchmarks/prb_complete/scripts/analyze_jordan_comprehensive.py"
missing = [str(p) for p in (runner, analyzer) if not p.is_file()]
if missing:
    raise SystemExit("ERROR: missing comprehensive scripts: " + ", ".join(missing))
subprocess.run([sys.executable, "-m", "py_compile", str(analyzer)], check=True)
subprocess.run([sys.executable, str(analyzer), "--self-test"], check=True)
text = runner.read_text(encoding="utf-8")
for required in ("CORE_U", "CORE_DELTA", "nkeep_400", "Lambda_2p5", "SCAN_METADATA.txt"):
    if required not in text:
        raise SystemExit(f"ERROR: runner missing expected token {required}")
print("PASS: comprehensive Jordan scaling suite installed and self-tested")
