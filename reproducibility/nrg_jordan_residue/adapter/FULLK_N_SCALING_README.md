# PTDirac adapter full-k matrix-chain N-scaling extension

This directory contains the installed, self-contained N-scaling workflow for
the PT-Dirac adapter.

It does not modify the existing Julia source, DM-NRG response scan, FDM
thermodynamics, or archived outputs.  The installed additions are:

- `benchmarks/prb_complete/fullk_matrix_chain/`
- `benchmarks/prb_complete/scripts/run_fullk_n_scaling.py`
- `benchmarks/prb_complete/scripts/analyze_fullk_n_scaling.py`
- `benchmarks/prb_complete/tests/validate_fullk_n_scaling_install.py`

Validate the installed extension from the adapter root with:

```bash
python \
  benchmarks/prb_complete/tests/validate_fullk_n_scaling_install.py
```

See `benchmarks/prb_complete/fullk_matrix_chain/README.md` for staged M2 run
commands and interpretation.
