# Full-k matrix-chain N scaling

This additive adapter extension runs the complete-basis finite-chain solver for
the full momentum-dependent positive `2 x 2` bath measure.  It is logically
separate from the Julia projected-overlap DM-NRG response scan.

## Scope

For every requested chain length `N`, the driver independently re-solves

```text
F_N(z; epsilon_d, W) = 0,
dF_N/dz = 0,
```

and then evaluates all lengths on one common ellipse in `(epsilon_d,W)`.  The
analyzer reports winding, pole exchange, half-winding, tracking and sum-rule
gates, center drift, and the sufficient pairwise Rouché inequality

```text
sup_C |D_Nhigh - D_Nlow| < min_C |D_Nlow|.
```

A passing pairwise inequality preserves the enclosed zero count between those
two finite chain lengths.  It does not alone prove an infinite-chain,
finite-temperature FDM-NRG, or strong-coupling Kondo limit.

## Installation check

From the adapter root:

```bash
python3 benchmarks/prb_complete/tests/validate_fullk_n_scaling_install.py
```

Use the adapter Python environment containing NumPy, SciPy, pandas, and
Matplotlib.  On the existing Mac setup this is normally `.venv/bin/python`.

## Economical M2 sequence

First run only the reoptimized centers:

```bash
caffeinate -i .venv/bin/python \
  benchmarks/prb_complete/scripts/run_fullk_n_scaling.py \
  --out output/fullk_n_scaling_z4 \
  --n-values 10 12 \
  --z-count 4 \
  --mode centers \
  --resume
```

Inspect the center logs.  If all centers pass, run 24-point common contours:

```bash
caffeinate -i .venv/bin/python \
  benchmarks/prb_complete/scripts/run_fullk_n_scaling.py \
  --out output/fullk_n_scaling_z4 \
  --n-values 10 12 \
  --reference-n 12 \
  --z-count 4 \
  --samples 24 \
  --mode common-contours \
  --resume
```

Analyze:

```bash
.venv/bin/python \
  benchmarks/prb_complete/scripts/analyze_fullk_n_scaling.py \
  output/fullk_n_scaling_z4
```

Principal outputs are:

- `analysis/FULLK_N_SCALING_AUDIT.md`
- `analysis/fullk_n_scaling_summary.json`
- `analysis/fullk_n_scaling_centers.csv`
- `analysis/fullk_n_scaling_center_pairs.csv`
- `analysis/fullk_n_scaling_rouche_pairs.csv`
- `analysis/fullk_n_scaling.pdf`

After the 24-point run passes, repeat in a new output directory with
`--samples 64`.  Add `N=14` only after monitoring the center-run memory.
The bundled tests show that very short or mixed-z charge branches need not
belong to the converged sequence: every N must pass its own center gate, and
the analyzer therefore reports the ground-charge sector for every z shift.

## Eight-z check

The bundled eight-z matrix chains are selected with `--z-count 8`:

```bash
caffeinate -i .venv/bin/python \
  benchmarks/prb_complete/scripts/run_fullk_n_scaling.py \
  --out output/fullk_n_scaling_z8 \
  --n-values 10 12 \
  --reference-n 12 \
  --z-count 8 \
  --samples 24 \
  --mode all \
  --resume
```

The implementation is sequential by design to avoid exhausting unified memory
on Apple Silicon.

With `--resume`, a stored result is reused only when its chain length,
interaction, z-grid size, reference center, contour resolution, and contour
half-widths match the current request.  Changing any of those settings causes
the affected stage to run again instead of mixing incompatible outputs.
