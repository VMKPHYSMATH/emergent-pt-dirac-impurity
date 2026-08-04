# Driven-Dirac impurity self-consistent saddle gate

This gate solves the corrected infinite-`U` Coleman saddle with the complete thermal Markov influence kernel. It is isolated from the manuscript, response letter, figure renderer, and public repository.

The outcome has two scopes:

- At `T=1e-6`, tests 7--10 pass for `beta0=0.35...0.55`: all three gauge-fixed seeds reach the same finite saddle, the original constraint and stationarity equations close, the rate matrix remains positive semidefinite, charge continuity holds, and doubled plus superfine quadratures agree.
- At the manuscript reference value `T_b=0.1`, the refined solver finds no finite-`r` saddle in the declared local-moment branch. Therefore the low-temperature result cannot support a manuscript claim made at `T_b=0.1`.

Principal files:

- `converged_saddles.csv`: canonical low-temperature solutions and residuals.
- `seed_convergence.csv` and `seed_independence.csv`: real, negative-real, and complex seed checks.
- `superfine_grid_validation.csv`: level-2 versus level-3 adaptive quadrature validation.
- `reference_temperature_solver.csv`: refined finite-saddle test at `T_b=0.1`.
- `reference_temperature_stability.csv`: diagnostic stationarity scan at `T_b=0.1`.
- `dressed_root_tracking.csv`: exact `k=0` double root and finite-`k` boundary-limited minima.
- `Self_Consistent_Saddle_Gate.pdf`: four-panel summary.
- `self_consistent_gate_summary.json`: parameters, software versions, checks, scope gate, and headline values.
- `SHA256SUMS`: checksums for every exported file and the executable solver.

The finite-`k` minima reach the upper boundary of the declared physical scan and are not classified as exceptional points. The converged low-temperature amplitude decreases across the scan, so this gate does not provide evidence for an enhanced saddle amplitude near the core EP.

Run a fresh validation from the repository root with

```bash
python reproducibility/low_temperature_saddle/self_consistent_saddle_gate.py \
  --out regenerated_low_temperature
```

Add `--reuse-seed-rows` to initialize from the archived canonical roots. The
residual, refinement, continuity, and reference-temperature checks are still
recomputed.
