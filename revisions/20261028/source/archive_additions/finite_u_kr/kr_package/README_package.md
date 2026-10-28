# Stationary full-bath slave-boson check

17 October 2026. New exploratory calculations for Vinayak M. Kulkarni.
This package is separate from the APS Open Science manuscript revision.
No submission, email, repository release or manuscript update was performed.

## Supported result

A zero-temperature, finite-U matrix KR/Gutzwiller saddle of the full stationary
radial bath has a numerically verified retuned resonance EP at mean filling
n_d=1 and U=0.0025. Its constraints, local stability, integration convergence,
double-pole conditions and self-consistent pole exchange pass the checks in
`results/validation.json`.

Selected parameters (absolute energy convention):

- K=pi/4, SOC=0.05, W=0.1, epsilon_O=120, Omega=60.
- epsilon_d=0.008761698951796248, V_O=0.9458641465915298.
- mu=0.006656973796272641, T=0.
- z_EP=0.006535134304575924 - 0.002983035811336982 i.
- n_d=1, D=0.228945653167805, Z_+=Z_-=0.997395502349013.
- P_single=n_d-2D=0.54210869366439.
- Local density eigenvalues approximately (0.4053765225, 0.5946234775).
- Pole separation half-winding -1/2; exchange error 8.49e-16.

**This is not a developed local-moment EP.** Empty and double configurations
each retain probability 0.22895. Mean occupation one alone is insufficient.
These results are approximate interacting saddle results, not an exact NRG
continuation or a theorem about the exact many-body impurity.

## Stronger-U comparison

The distinct occupied gate path fixes mu=0.012, V_O=1.21342200961, W=0.1 and
epsilon_d=0.0151703497834-U/2. This is not particle-hole symmetry of the bath.
At U=0.2 it yields n_d=0.9940551563 and D=0.00125994048. Its density eigenvalues
are approximately (0.00131599, 0.99273916): the state is strongly polarized.
The formal saddle has a bound pole near -0.08527794 and a separate outgoing
resonance near 0.10881797-0.00240659i. Neither is a double pole.

This does not prove that all occupied-regime EPs are absent. The controls on
this gate path were not chosen to enforce the EP equations. Local continuation
attempts are not an exhaustive search. Large-energy strong-U poles from a
coherent saddle should not be treated as controlled Hubbard satellite spectra.

## Why a shifted Hubbard-dimer EP is compatible with these results

An EP continuing to a new control value and a split pair at the old control
value are compatible. In this impurity problem the saddle and bath frequency
dependence both enter the pole condition. The correct statement is that an EP
was recovered by retuning in the tested weak-U regime, while an occupied,
strongly correlated EP remains unestablished.

It also matters which particle-number sector is compared. For the usual
number-conserving spinful two-site Hubbard model, the interaction vanishes in
the one-electron sector and is U times the identity in the three-electron
sector. Their EPs cannot diagnose a nontrivial effect of U. The two-electron
singlet sector mixes singly and doubly occupied states and has nontrivial
interaction dependence. This observation does not presume which sector was
used in any separate project.

No universal theorem says interactions can only move EPs. See Schäfer, Budich
and Luitz, Physical Review Research 4, 033181 (2022),
https://doi.org/10.1103/PhysRevResearch.4.033181, for an interacting fermion model
with EP continuation, interaction-induced EPs, and annihilation. Local smooth
continuation requires the combined saddle/EP Jacobian to remain nonsingular;
global survival at arbitrary U does not follow from that local argument.

## Method and scope

- Exact frequency-dependent stationary bath kernel, with the same static
  spin-even subtraction as the manuscript's U=0 certificate.
- Grand-canonical equilibrium of the rotating-frame Hamiltonian is an explicit
  state assumption. A static rotating-frame Hamiltonian does not alone prove
  laboratory thermalization.
- Finite-U, normal, real-sector single-orbital KR/Gutzwiller saddle with four
  local configuration probabilities, a variational natural spin axis and a
  real symmetric 2x2 constraint matrix. Spin polarization and off-diagonal
  renormalization are allowed. The omitted complex spin direction is checked
  for local stability at the reported points.
- Imaginary-axis integration evaluates equilibrium density and bath forces
  without broadening. This is not an analytic reconstruction from sampled
  Matsubara data. The complex resonance function is available analytically.
- Saddle parameters are fixed while differentiating in spectral z. They are
  re-solved when external controls move, including on the encircling contour.
- The coherent physical Green function has spectral weight R R^dagger. It is
  not a complete, normalized all-frequency physical spectral function. Local
  occupations are read from the constrained probabilities, not from the
  coherent spectral weight alone.
- A normalized transverse stationarity equation avoids spurious roots at the
  coordinate singularity where the two natural occupations coincide.
- Positive constrained Hessians demonstrate local stability. No global
  free-energy minimization theorem or exact error estimate is claimed.

## Reproduce

Python 3.12; tested with NumPy 2.3.5 and SciPy 1.17.0. Matplotlib generates the
figure; a standard LaTeX installation builds the note. Run from this directory:

```bash
export OPENBLAS_NUM_THREADS=1
python src/filling_continuation.py --order 32 --umax .003 --out results/ep_n1.json
python src/select_ep.py
python src/occupied_scan.py
python src/validate.py
python src/plot_results.py
pdflatex Stationary_SB_Calculation_Note.tex
```

Additional paths:

```bash
python src/matrix_saddle.py --ep --order 32 --out results/ep_mu000.json
python src/filling_continuation.py --order 32 --control W --umax .0075 --out results/ep_n1_W.json
```

`src/full_k_two_cut_ep_certificate.py` is copied from the existing manuscript
reproducibility source. The saddle, continuation and validation scripts are new.
Failed solver attempts are not evidence of EP absence and must not be counted
as valid points. Use the stored success flag and residual thresholds.

## Files

- `Stationary_SB_Calculation_Note.pdf` and `.tex`: four-page calculation note.
- `figures/Stationary_SB_Checks.pdf` and `.png`: vector/raster scientific figure.
- `results/validation.json`: independent checks, refined selected saddle and
  complete 128-step self-consistent pole-exchange record.
- `results/ep_n1_U0025.json`: selected saddle before final refinement.
- `results/ep_n1.json`: fixed-W EP continuation with n_d=1.
- `results/ep_n1_W.json`: fixed-V_O EP continuation with n_d=1.
- `results/ep_mu000.json`: low-filling, fixed-mu EP continuation through U=0.2.
- `results/occupied_gate_path.json`: distinct occupied gate path through U=0.2.
- `results/benchmark_table.csv`: selected observables for straightforward reuse.
- `SHA256SUMS.txt`: archive content checksums.

Method references: Kotliar and Ruckenstein, PRL 57, 1362 (1986),
https://doi.org/10.1103/PhysRevLett.57.1362; Lechermann et al., PRB 76, 155102
(2007), https://arxiv.org/abs/0704.1434.
