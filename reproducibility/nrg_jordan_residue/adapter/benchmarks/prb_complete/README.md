# PT-Dirac complete Bethe / saddle / causal-RG / NH-NRG benchmark

This is a **drop-in benchmark layer** for the current cluster-fixed
`PTDirac_NHNRG_adapter` folder.

It does four logically separate calculations and then cross-compares them:

1. **Frozen biorthogonal Bethe/scattering diagnostics**
   - exact local discriminant and Puiseux gap;
   - local eigenvector condition number;
   - external finite-`U` quartic `s_eff` benchmark;
   - biorthogonal residue proxy.

2. **Self-consistent low-temperature saddle**
   - the supplied `self_consistent_saddle_gate.py`;
   - constraint, stationarity, FDR, continuity, seed and grid checks.

3. **Causal channel-resolved RG**
   - the supplied `channel_resolved_rg_gate.py`;
   - positive semidefinite spectral matrix;
   - matrix flow without a Petermann or `Z_bio` multiplier;
   - active/control threshold-scale comparison.

4. **Finite-`U` NH-NRG**
   - detuning scan at the normalized core `beta0=0.5`;
   - `U` scan;
   - metallic, Hermitian, side-point, and convergence controls;
   - many-body complex level flow and fixed-point tower;
   - iteration-resolved impurity addition/removal transition weights.

## What the Julia patch adds

The Hamiltonian, Wilson chain, cluster-aware biorthogonalization, and truncation
logic are unchanged. The patch adds one output:

```text
impurity_transition_weights.csv
```

At every NRG iteration it records the states with the largest basis-rescaling
invariant products

```text
<L0|d|Ri><Li|d†|R0>
<L0|d†|Ri><Li|d|R0>
```

This lets the analysis identify impurity-supported crossovers without pretending
that arbitrary low-energy Wilson-chain levels are Bethe rapidities.

## Install into the current adapter folder

Extract this package anywhere, then run from the extracted package:

```bash
./install_into_current_folder.sh \
  "$HOME/Downloads/PTDirac_NHNRG_adapter"
```

The installer:

- backs up the current `src/PTDiracNHNRG.jl`;
- installs the benchmark-aware clusterfix source;
- creates `benchmarks/prb_complete/`;
- does not remove existing outputs.

Validate:

```bash
cd "$HOME/Downloads/PTDirac_NHNRG_adapter"
python3 benchmarks/prb_complete/tests/validate_install.py
"$HOME/.juliaup/bin/julia" --project=. test/smoke_test.jl
```

## Run

Pilot:

```bash
./benchmarks/prb_complete/run_benchmark_macos.sh pilot
```

Production:

```bash
./benchmarks/prb_complete/run_benchmark_macos.sh production
```

Resume after interruption:

```bash
python3 benchmarks/prb_complete/scripts/run_complete_benchmark.py \
  --profile production --resume
```

Analyze existing outputs only:

```bash
python3 benchmarks/prb_complete/scripts/run_complete_benchmark.py \
  --profile production --analyze-only
```

Skip the expensive saddle calculation while developing:

```bash
python3 benchmarks/prb_complete/scripts/run_complete_benchmark.py \
  --profile pilot --skip-saddle --resume
```

## Main outputs

```text
output/prb_complete_benchmark/
  nhnrg/
  saddle/
  causal_rg/
  analysis/
    bethe_local_predictions.csv
    nhnrg_run_summary.csv
    nhnrg_flow_by_iteration.csv
    nhnrg_impurity_transition_by_iteration.csv
    fixed_point_comparison.csv
    Bethe_NHNRG_Benchmark.pdf
    Saddle_RG_NHNRG_Crosscheck.pdf
    BENCHMARK_REPORT.md
    benchmark_decision.json
    SHA256SUMS
```

## Interpretation discipline

The package deliberately does **not** equate unlike objects:

- `s_eff` is an external frozen Bethe/SW diagnostic; the SOC form factor is not
  silently inserted into the NH-NRG Hamiltonian.
- The saddle is an independent mean-field existence and consistency test.
- Causal RG is an energy-resolved weak-flow test.
- NH-NRG determines the many-body crossover and infrared tower.
- The final kept-space Lehmann curve is not used as a physical spectrum.
- No Petermann or biorthogonal-residue multiplier is inserted into the physical
  RG flow or Kondo exponent.
- The exact Jordan point still requires a Jordan-aware truncation; ordinary
  eigenvector NH-NRG is run only at nonzero detuning.

## Recommended PRB use

The strongest publication-facing comparison is:

1. local Puiseux gap and condition number versus `delta_coh`;
2. NH-NRG imaginary-level crossover versus Wilson iteration;
3. impurity-supported NH-NRG minimum gap versus frozen pole/rapidity gap;
4. late NH-NRG tower versus the bare pseudogap Wilson chain;
5. causal RG active/control scale ratio and the low-temperature saddle as
   independent cross-checks.

The result should be stated as a benchmark of a finite-energy crossover unless
the production convergence controls demonstrate a distinct stable infrared
fixed point.
