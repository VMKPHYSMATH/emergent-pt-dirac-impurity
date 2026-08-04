#!/usr/bin/env python3
from __future__ import annotations
import subprocess, sys, tempfile, tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
required = [
    ROOT/'benchmarks/prb_complete/scripts/dmnrg_response_core.jl',
    ROOT/'benchmarks/prb_complete/scripts/run_dmnrg_response_one.jl',
    ROOT/'benchmarks/prb_complete/scripts/run_dmnrg_response_scan.py',
    ROOT/'benchmarks/prb_complete/scripts/extract_dmnrg_response.py',
    ROOT/'benchmarks/prb_complete/config/dmnrg_response_scan.toml',
    ROOT/'test/dmnrg_response_smoke.jl',
]
for path in required:
    assert path.is_file(), path

core = required[0].read_text(encoding='utf-8')
for marker in [
    'ep_functional_form_imposed = false',
    'peak_model_imposed = false',
    'B*rho - rho*B',
    'if !final_shell && (a in nkeep_set) && (b in nkeep_set)',
    'adjoint(shell.left) * product_operator * shell.right',
]:
    assert marker in core, marker

with tempfile.TemporaryDirectory() as td:
    out = Path(td)/'dry'
    subprocess.run([
        sys.executable,
        str(ROOT/'benchmarks/prb_complete/scripts/run_dmnrg_response_scan.py'),
        '--profile','smoke','--dry-run','--out',str(out),
    ], cwd=ROOT, check=True)
    manifest = tomllib.loads((ROOT/'benchmarks/prb_complete/config/dmnrg_response_scan.toml').read_text())
    assert manifest['dmnrg_response']['components'] == ['x','z']
    generated = list((out/'generated_configs').glob('*.toml'))
    assert len(generated) == 2
    for cfg in generated:
        parsed = tomllib.loads(cfg.read_text())
        assert parsed['thermodynamics']['enabled'] is False
        assert parsed['lehmann']['enabled'] is False
        assert parsed['dmnrg_response']['exploratory_complex'] is True

subprocess.run([sys.executable, '-m', 'py_compile',
                str(ROOT/'benchmarks/prb_complete/scripts/run_dmnrg_response_scan.py'),
                str(ROOT/'benchmarks/prb_complete/scripts/extract_dmnrg_response.py')],
               check=True)
print('PASS: DM-NRG response static and dry-run validation')
