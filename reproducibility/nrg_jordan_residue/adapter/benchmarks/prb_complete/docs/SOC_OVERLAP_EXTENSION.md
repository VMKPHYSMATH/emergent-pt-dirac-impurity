# SOC-overlap extension for the quartic/Jordan benchmark

## Why the SOC factor belongs in a matrix hybridization

For the two chiral branches

\[
\epsilon_\pm(k)=\epsilon_0(k)\pm\lambda k,
\]

the local densities of states are identical after the full symmetric momentum
integration because the branches map into one another under \(k\to-k\).  A
constant Wilson-site flavour splitting therefore does not reproduce the
projected overlap factor

\[
F(\lambda)=\max\!\left[1-(\lambda/k_{\max})^2,0\right].
\]

The minimal low-energy embedding used here keeps the scalar Wilson chain as the
bath radial degree of freedom and places SOC in the channel-coherent impurity
hybridization:

\[
S_\lambda=
\begin{pmatrix}
1 & x\\
x & 1
\end{pmatrix},\qquad
x=\mathrm{clip}(\lambda/k_{\max},-1,1),
\]

\[
V_\lambda=V S_\lambda^{1/2}.
\]

Then

\[
\frac{V_\lambda V_\lambda^\dagger}{V^2}=S_\lambda,
\qquad
\det S_\lambda=1-x^2=F(\lambda).
\]

The trace of the normalized hybridization remains two for every \(\lambda\), so
SOC redistributes coherent channel overlap without artificially increasing the
total bath coupling.  At \(|\lambda|=k_{\max}\), one hybridization eigenchannel
decouples and the determinant vanishes.

This is a **minimal projected SOC embedding**, not a full momentum-resolved
block-Lanczos Wilson chain.  It is appropriate for testing whether the proposed
\(F(\lambda)\)-dependent quartic relation emerges dynamically in the present
low-energy NRG model.  A full microscopic calculation would require the exact
matrix hybridization function \(\Gamma_{ab}(\omega,\lambda)\) and a block
star-to-chain transformation.

## Run

```bash
cd "$HOME/Downloads/PTDirac_NHNRG_adapter"

"$HOME/.juliaup/bin/julia" --project=. \
  benchmarks/prb_complete/scripts/run_soc_quartic_scan.jl pilot
```

Analyze iteration zero with the run-specific form factor:

```bash
python3 benchmarks/prb_complete/scripts/quartic_pole_audit.py \
  output/soc_quartic_benchmark \
  --out output/soc_quartic_benchmark/analysis \
  --F auto \
  --b-mode hybridization \
  --iteration 0
```

Important outputs:

- `soc_hybridization_matrix.csv`: the coupling matrix and normalized
  hybridization matrix for each run;
- `quartic_pole_pairs.csv`: pole gaps and quartic residuals with the actual
  run-specific \(F(\lambda)\);
- `SOC_Gap_and_Jordan_vs_F.pdf`;
- `SOC_Quartic_Residual_vs_F.pdf`.

## Acceptance checks

1. The scalar control and overlap mode at \(\lambda=0\) must agree numerically.
2. `soc_normalized_hybridization_det` must agree with `F_lambda` to numerical
   precision.
3. The \(U=0\) controls identify changes caused purely by the one-body SOC
   overlap.
4. A quartic floor requires the finite-\(U\) pole gap to approach a nonzero
   \(F^{1/4}\)-dependent limit under detuning extrapolation.
5. A surviving EP instead gives a vanishing pole gap and finite pairwise Jordan
   coefficient.
