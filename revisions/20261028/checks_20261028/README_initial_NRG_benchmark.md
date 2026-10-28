# Local-moment finite-chain feasibility check

27 October 2026. This is a new diagnostic package, separate from the submission manuscripts.

## Conclusion

The existing full-matrix Hermitian chain solver can access a nearly singly occupied, charge-quenched impurity at U=0.2. Its kept-space occupation calculation agrees closely with exact charge-sector diagonalization for the same eight-block bath. This establishes computational feasibility for short chains. It does not establish a screened Kondo state, a local-moment resonance exceptional point, continuum convergence, or an interaction threshold for destroying EPs.

## Model and settings

The full auxiliary-band stationary rotating-frame model is used with W=0.1, lambda=0.05, V_O=1.2134220096136235, epsilon_O=120, Omega=60 and cutoff pi/4. Set mu=0.012, U=0.2, and absolute epsilon_d=0.01517035-U/2=-0.08482965. This gate choice is not a particle-hole-symmetry condition for the asymmetric bath.

The bath is rebuilt around this chemical potential, with 512 radial quadrature nodes, Lambda=1.5, one Wilson shift z_W=0.5, 64 logarithmic intervals, and a 24-block mapping. Only the first six or eight blocks enter the many-body calculation. N counts bath blocks; each block has two fermionic modes. The impurity adds two more modes. The ground state is the zero-temperature grand-canonical Hermitian reference.

Exact ED has no many-body truncation within the tested finite chain. The pilot retains a specified number of states after each enlargement. Its occupations are propagated as independent operators; they are not reconstructed as products of projected fermion operators.

## Results

P1 = n_d - 2D is the probability of single occupancy. The spin magnitude is |<S_d>| = (rho_max-rho_min)/2, in units of hbar. The response error is the Frobenius norm of the matrix anticommutator sum-rule defect, ||C-I||_F, before broadening.

| Bath blocks | Retained states | n_d | D | P1 | Spin magnitude | Response sum-rule error |
| --- | --- | --- | --- | --- | --- | --- |
| 6 | Exact ED | 0.990520332 | 0.000000000 | 0.990520332 | 0.495260166 | 3.140e-16 |
| 6 | 64 | 0.990663043 | 0.000000000 | 0.990663043 | 0.495331521 | 8.663e-03 |
| 6 | 128 | 0.990513596 | 0.000000000 | 0.990513596 | 0.495256798 | 2.213e-03 |
| 6 | 256 | 0.990520332 | 0.000000000 | 0.990520332 | 0.495260166 | 4.891e-04 |
| 8 | Exact ED | 0.994455141 | 0.001811282 | 0.990832577 | 0.494750624 | 3.146e-16 |
| 8 | 256 | 0.994483639 | 0.001835920 | 0.990811798 | 0.494737825 | 6.442e-03 |
| 8 | 512 | 0.994453726 | 0.001809650 | 0.990834426 | 0.494751385 | 2.087e-03 |

At eight blocks the exact impurity density eigenvalues are 0.002476947 and 0.991978194. The moment is nearly saturated. Small double occupancy is therefore not, by itself, evidence of a screened Kondo regime. Susceptibility, impurity entropy or spin correlations and their scale dependence are needed to distinguish fluctuating moments from polarization and screening.

A material limitation is the number of electrons represented by these short chains: the exact grand-canonical ground charge is one at N=6 and two at N=8, counting impurity plus bath. At N=6, D=0 follows from having only one electron in the whole chain; its ground-state occupation cannot establish nontrivial interaction physics. The N=8 result includes genuine two-electron interaction effects, but is still a very sparse representation of the occupied continuum.

For N=8, increasing retained states from 256 to 512 reduces the occupation errors relative to ED. The spectral sum-rule error remains about 0.00209 at 512 states, even though n_d and D are accurate to roughly 1.5e-6 and 1.6e-6. Accurate occupations do not certify the response or complex-pole reconstruction. Complete-basis dynamics, or exact finite-sector responses where feasible, are needed for the next pole benchmark.

The full 24-block bath mapping differs from the continuum by about 0.8-1.2 percent at the recorded upper-half-plane check points. Shorter chain errors, additional mesh shifts, and Lambda dependence have not been certified by this test. No EP retuning or winding calculation was performed.

## Recommended continuation

1. Extend chain length and compare occupancies, polarization and the represented occupied bath. Repeat for multiple z_W shifts and smaller Lambda. Use exact charge-sector results wherever the sector dimension permits; the present safety cap is 50,000 states per sector.
2. Check spin response/correlations and a nonpolarized control before interpreting the charge-quenched state as evidence for Kondo screening. A finite Hermitian chain has real many-body energies, not defective complex resonance eigenvalues.
3. Validate the physical real-axis matrix Green function and interaction self-energy with U=0 recovery, spectral sum rules, high-frequency moments, causality, and truncation/discretization checks. The kept-space pilot is not complete-basis or FDM dynamics.
4. Use the validated finite-chain self-energy only as an explicitly labeled continuum embedding approximation, and retune two controls such as epsilon_d and W. Test the double-zero conditions, rank-one kernel, nonzero second derivative and pole exchange. Split poles on a fixed gate path do not exclude a retuned EP.
5. Keep the existing manuscript conclusions unchanged unless these stronger checks pass. Finite-chain survival would not alone establish a continuum local-moment EP.

The September v4 M2 adapter patch was also inspected. It adds exchange-model, causal/Lax and Jordan-scaling diagnostics; it does not itself supply the full-matrix driven-bath complete-basis response missing from the kept-space pilot. A non-Hermitian many-body solver is unnecessary for the Hermitian equilibrium reference used here.

## Reproduce

Python 3.10 or newer with NumPy, SciPy and Matplotlib is required by the bundled source modules. Run from this directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 benchmark_local_moment.py
```

The script uses the copied input configuration, regenerates or reuses the cached chains, and writes feasibility_results.json. The source modules are unchanged copies of the focused APS candidate; their hashes are recorded in source_sha256.json. Re-running may slightly change eigensolver rounding and timings. Cached chains can be removed to force remapping.

Methodological references: Bulla, Costi and Pruschke, Rev. Mod. Phys. 80, 395 (2008), https://arxiv.org/abs/cond-mat/0701105; Kugler, Phys. Rev. B 105, 245132 (2022), https://arxiv.org/abs/2202.04063. The latter provides an improved self-energy estimator; it has not been implemented in this diagnostic.
