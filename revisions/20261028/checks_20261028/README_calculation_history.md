# Update: bath-convergence checks, 28 October 2026

New U=0 checks of chain length, Wilson shifts, Lambda and radial quadrature are in `bath_convergence/README.md`. They clarify which parts of the numerical convergence campaign remain unfinished. The preceding finite-U comparisons are preserved below.

# Matched finite-chain ED, NRG pilot and finite-U slave-boson comparison

Calculations: 27 October 2026; package completed 28 October 2026. This extends the earlier finite-chain feasibility package. It is a diagnostic calculation, separate from the submission manuscripts.

## Result

The finite-U matrix Kotliar-Ruckenstein (KR) saddle reproduces the tested eight-block exact-diagonalization (ED) occupations closely along the occupied gate path. At U=0.2, its filling differs from ED by -2.71e-5, and double occupancy is 3.85 percent lower. The existing 512-state NRG pilot agrees more closely with ED for these static quantities.

This validates the numerical implementation and measures the saddle's static error for this particular finite bath and polarized parameter path. It does not validate the full continuum interacting self-energy, Kondo screening, a local-moment EP, or an EP destruction threshold. No new complex-pole search was performed.

## Matched setup

All three methods use U=0.2 (or the listed scan value), W=0.1, mu=0.012, and epsilon_d=0.01517035-U/2. Bath parameters are lambda=0.05, V_O=1.2134220096136235, epsilon_O=120, Omega=60, and k_c=pi/4. The finite bath is rebuilt around mu, retaining the remote auxiliary bands. Lambda_W=1.5, z_W=0.5 and eight two-mode bath blocks are used. Temperature is zero.

The comparison has two distinct purposes:

- ED versus KR on the identical finite chain measures the error of the mean-field approximation for the reported static observables.
- Finite-chain KR versus continuum KR measures bath-discretization/truncation sensitivity within the same approximation. At U=0 both solvers are exact for their respective baths, providing a direct noninteracting bath-convergence diagnostic.

The saddle retains separate empty, two single-occupancy and double-occupancy probabilities, with a freely rotating spin-density axis and a real symmetric matrix constraint. No spin-symmetric or scalar-renormalization restriction is imposed. This is the finite-U saddle already used in the focused manuscript, not the infinite-U Coleman approximation.

## Numerical comparison

| U | n_d, ED | n_d, chain KR | D, ED | D, chain KR | Relative D error | D, continuum KR |
| --- | --- | --- | --- | --- | --- | --- |
| 0.000 | 0.850657 | 0.850657 | 0.08848215 | 0.08848215 | -0.00% | 0.22737609 |
| 0.002 | 0.869825 | 0.869766 | 0.08357513 | 0.08427990 | +0.84% | 0.21861582 |
| 0.020 | 0.953989 | 0.953405 | 0.04484612 | 0.04568811 | +1.88% | 0.05983649 |
| 0.080 | 0.987370 | 0.987238 | 0.00890186 | 0.00855203 | -3.93% | 0.00670945 |
| 0.200 | 0.994455 | 0.994428 | 0.00181128 | 0.00174151 | -3.85% | 0.00125994 |

The U=0.2 endpoint, including the earlier NRG pilot:

| Method | n_d | D | P1=n_d-2D |
| --- | --- | --- | --- |
| Exact eight-block ED | 0.994455141 | 0.001811282 | 0.990832577 |
| NRG pilot, 512 retained states | 0.994453726 | 0.001809650 | 0.990834426 |
| KR, same eight-block bath | 0.994428050 | 0.001741508 | 0.990945034 |
| KR, continuum bath | 0.994055156 | 0.001259940 | 0.991535275 |

At U=0.2 all methods find strong polarization, with spin magnitude approximately 0.495 (maximum 1/2). Accurate filling or small D does not establish a screened Kondo state. The endpoint KR renormalization eigenvalues are about 0.9961 and 0.9844 on the finite chain, consistent with a highly polarized state rather than evidence for a very small Kondo coherence scale.

## What the bath comparison reveals

Already at U=0, finite-chain ED and KR agree to about 4e-13 in n_d, but their n_d=0.850657 differs substantially from the continuum value 0.962817. D is 0.088482 on the chain versus 0.227376 in the continuum, and the impurity spin magnitude is 0.3040 versus 0.0662. These discrepancies are present without interaction. They demonstrate that this short chain is not a converged representation of the occupied bath at this chemical potential.

At U=0.2 the chain-to-continuum KR change in D is about 27.7 percent, larger than the 3.85 percent KR-versus-ED difference on the same chain. This comparison is specific to the observable and path; it is not a bound on continuum mean-field error. The exact eight-block ground state has only two electrons in impurity plus chain.

## Implementation and numerical checks

The finite-chain extension evaluates the KR quasiparticle functional by direct diagonalization of the same one-body bath matrix. The occupied quasiparticle projector gives the impurity density and the derivative with respect to the hybridization renormalization. Chemical potentials and absolute versus relative energies are matched explicitly.

Checks performed:

- U=0 recovery against exact charge-sector ED on the same chain.
- Agreement of direct-diagonalization density and hybridization-force calculations with an independent imaginary-axis quadrature: maximum discrepancies below 7e-13 and 1.2e-14 respectively at the comparison points.
- Continuation of the chain saddle along the occupied gate path, with residual checks. Two additional seeds at U=0.2 recover the same density to below 7e-12. This is a branch-consistency check, not a proof that no other stationary solution exists.
- Continuum saddle re-solved at each tabulated U using the same bare controls.
- Exact finite-chain occupancies evaluated at each tabulated U. The existing NRG pilot was only compared at U=0.2; there is no new NRG scan across all five U values.

The prior NRG result still has a matrix spectral sum-rule defect about 2.09e-3 at 512 retained states. Static agreement does not establish complete-basis dynamics. The chain KR quasiparticle response should likewise not be identified with an exact full-frequency physical electron spectrum: its coherent saddle does not supply the missing incoherent spectral weight.

## Next calculation

Extend chain length and Wilson-shift averaging while tracking filling, polarization, double occupancy and spin response. Demonstrate improved U=0 continuum recovery before interpreting interacting differences. Then compare the matrix interaction self-energy and retuned resonance-pole conditions. Equality of fillings alone cannot establish equality of EP locations or topology.

## Files and reproduction

- `kr_comparison/comparison_results.json`: saddle paths and matched ED/KR results.
- `kr_comparison/comparison_table.csv`: concise numerical table.
- `kr_comparison/ED_KR_comparison.pdf` and `.png`: three-panel comparison. Lines connect the five computed comparison points; they are guides to the eye.
- `kr_comparison/endpoint_seed_checks.json`: independent endpoint-seed checks.
- `kr_comparison/compare_kr.py`: new finite-chain KR adapter and comparison driver.
- `kr_comparison/plot_comparison.py`: reproduce the comparison figure from JSON.
- `benchmark_local_moment.py`, `feasibility_results.json`, and `README_initial_NRG_benchmark.md`: original NRG pilot benchmark.
- `solver/`: unchanged source modules and the archived continuum seed path.

Dependencies: Python 3.10+, NumPy, SciPy, Matplotlib. From this package directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 kr_comparison/compare_kr.py
```

The output overwrites the comparison JSON; remove the local chain cache to force bath remapping. Numerical rounding and timing may differ. The bundled source hashes identify the versions used; no manuscript or public repository was changed by this calculation.

Background: R. Fresard and colleagues, The Slave-Boson Approach to Correlated Fermions, https://www.cond-mat.de/events/correl15/manuscripts/fresard.pdf. The numerical statements above are computed for the specific model and finite bath in this package, not inferred from that reference.


## Two-sided EP and causality tests — 28 October 2026

See `two_sided_ep/REPORT.md` for the new calculation. The known U = 0 EP and selected finite-chain embedded EPs pass two-sided reconstruction and pole-exchange checks. Occupied interacting EP centers do not converge between six, eight and ten bath blocks. Split poles pass the sampled physical-sheet positivity tests; splitting is not evidence of lost causality. No continuum local-moment EP is claimed. The earlier static ED/NRG/KR and noninteracting bath-convergence checks below remain available.


## Optimized passive matrix bath — 28 October 2026

See `optimized_bath/REPORT.md` for the new fit and fixed-control interacting comparison. Selected static quantities improve, but the full spin density and matrix self-energy remain sensitive to bath size and fitting weights. No interacting continuum EP is certified. This extension includes four fit objectives, held-out validation, occupied spectral-moment checks, a 32-orbital one-body comparison, and exact finite-bath U=0.2/U=0.0025 responses.


## Exact interacting PT control — 28 October 2026

See `pt_interacting_control/PT_Interacting_Control_Note.pdf` and `pt_interacting_control/REPORT.md`. The spin-symmetric finite Anderson control now has a nonzero double-pole amplitude proof for the half-filled dimer at every finite repulsive U, plus 14 exact finite-system checks with one, three and five bath sites. The amplitude is measured directly; the calculation does not fit a Jordan self-energy. Two-sided splitting and pole exchange are verified at U=4 and 8 with three bath sites. This strengthens the existing restricted-control result; it does not certify an occupied interacting EP in the original passive driven continuum bath. The original manuscripts remain unchanged.
