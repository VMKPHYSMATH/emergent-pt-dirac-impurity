# Complete-basis/FDM entropy scale

This extension adds equilibrium finite-temperature thermodynamics to the
corrected biorthogonal PT-Dirac NH-NRG adapter.  It stores every discarded NRG
state on a common physical energy axis and constructs the Anders--Schiller
complete-basis density matrix

\[
\rho(T)=Z^{-1}\sum_{n,s\in D_n,e}e^{-E_{ns}/T}
|R_{nse}\rangle\langle L_{nse}|.
\]

The environment multiplicity is exactly `4^(N-n)`.  The complete-basis state
count is checked against the full finite Wilson-chain Hilbert-space dimension.
The bath contribution is calculated exactly from the same finite Wilson chain
and subtracted to obtain `S_imp(T)` and `C_imp(T)`.

## Scope

A thermodynamic label requires positive Boltzmann weights.  The adapter emits
`T_K^S` only when the complete-basis spectrum is real within tolerance, the
frame is `relative`, the density matrix normalizes, the complete-basis count
closes, a screened low-temperature entropy is present, and a local-moment
entropy regime is resolved.  Broken-PT, passive-loss, and exactly defective
runs receive no equilibrium `T_K`.

The explicit definition is

\[
S_{\rm imp}(T_K^S)=\frac{1}{2}\ln 2.
\]

This is a thermodynamic entropy crossover scale.  It is not a spectral HWHM and
is not silently equated with every other convention for the Kondo temperature.

## Pilot

```bash
python3 benchmarks/prb_complete/scripts/run_fdm_tk_scan.py \
  --base-config config/example.toml \
  --scan-config benchmarks/prb_complete/config/fdm_tk_scan.toml \
  --out output/fdm_tk_scan \
  --profile pilot --resume
```

The exact `beta0=0.5` Jordan point is omitted.  The pilot uses `z=0.5`; the
production profile uses three z shifts and includes broken-side adverse controls.

## Outputs

Each run writes:

- `fdm_thermodynamics.csv`
- `fdm_shell_weights.csv`
- `fdm_tk_summary.toml`

The scan analysis writes:

- `fdm_tk_vs_beta0.csv`
- `FDM_TK_entropy_vs_beta0.pdf`
- `FDM_impurity_entropy_curves.pdf`
- `FDM_thermodynamic_quality_vs_beta0.pdf`
- `FDM_TK_AUDIT.md`
