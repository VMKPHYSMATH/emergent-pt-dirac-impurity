# Operational low-energy scales versus beta0

This extension adds a restartable beta0 scan and definition-resolved scale
extraction to the corrected PTDirac NH-NRG adapter.

## Extracted quantities

- `T_flow`: Wilson scale where the same-parity complex level flow first stays
  within the configured fixed-point distance for consecutive comparisons.
- `T_pair_split_n4`, `T_pair_split_n5`, `T_pair_split_n6`: half splitting of
  the full-matrix impurity-supported pole pair at the indicated fixed NRG
  iteration. The pair is tracked continuously in beta0 from the nominal EP.
- `T_trace_HWHM`: HWHM of the kept-space trace spectral peak, reported only
  when the Lehmann sum-rule and positivity gates pass.
- `T_J_HWHM`: HWHM of the absolute Jordan-projected kept-space spectrum,
  reported only when the sum-rule gate passes.
- `T_matrix_Jordan_onset`: Wilson scale at which the branch-tracked matrix
  residue first passes the alignment, trace-cancellation, and nilpotent-
  mismatch gates for consecutive same-parity iterations.

These are operational crossover/resonance diagnostics. They are not silently
identified with a thermodynamic Kondo temperature. A thermodynamic `T_K`
requires a finite-temperature impurity susceptibility, entropy, or another
explicitly justified universal criterion.

## Core-output addition

`kept_space_lehmann.csv` now contains three extra columns:

- `minus_ImGx_over_pi`
- `minus_ImGz_over_pi`
- `minus_ImJordanG_over_pi`

where the last quantity is

`-Im Tr[N_EP^dagger G(omega)] / (pi ||N_EP||_F)`.

The original columns are unchanged.

## Run

From the adapter root:

```bash
python3 benchmarks/prb_complete/scripts/run_beta_scale_scan.py \
  --base-config config/example.toml \
  --scan-config benchmarks/prb_complete/config/beta_scale_scan.toml \
  --out output/beta_scale_scan \
  --profile pilot --resume
```

For the full scan, replace `pilot` with `production`.

Analyze existing completed runs without rerunning Julia:

```bash
python3 benchmarks/prb_complete/scripts/extract_beta_scales.py \
  output/beta_scale_scan \
  --scan-config benchmarks/prb_complete/config/beta_scale_scan.toml \
  --out output/beta_scale_scan/analysis
```

Principal outputs:

- `beta_scale_summary.csv`
- `branch_tracked_beta_pairs.csv`
- `flow_distance_vs_iteration.csv`
- `spectral_scale_diagnostics.csv`
- `Operational_scales_vs_beta0.pdf`
- `Scale_extraction_quality_vs_beta0.pdf`
- `BETA_SCALE_AUDIT.md`

## Interpretation

Plot every successfully gated scale. Do not force missing or failed scales
onto a curve, and do not relabel a kept-space width as `T_K`. The comparison
is most informative precisely when the operational definitions disagree.
