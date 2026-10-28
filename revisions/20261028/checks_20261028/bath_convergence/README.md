# Bath convergence check — 28 October 2026

## What had actually been tested previously

The recent finite-U ED/NRG/KR comparison used one Wilson shift z_W=0.5, Lambda=1.5, W=0.1, and six/eight bath blocks. It demonstrated retained-state convergence of some static observables for a fixed finite bath. It did not implement a full interacting scan in chain length, mesh averaging, Lambda and response reconstruction.

The earlier note to Rok Zitko recorded a two-shift occupied-regime test (z_W=0.25,0.75), which changed pole separations by about 1.7–3.5 percent but did not resolve the larger six/eight-block differences. A systematic converged real-axis interacting response and outgoing-sheet reconstruction remain unfinished. The standard equation-of-motion correlator estimator is present; Kugler's improved estimator has not been implemented.

## New checks in this package

These are U=0 exact one-body calculations, so there is no many-body truncation. The physical bandwidth, W, rotating drive, chemical potential and impurity energy are held fixed. The stationary Hamiltonian includes the full auxiliary bands. The analytic continuum reference is n_d=0.962817234350311, D=0.227376091762845, and |<S_d>|=0.0661677030555713. Imaginary-axis quadrature orders 48 and 72 agree in the density matrix within 2.8e-16.

The initial scan tests 1, 2, 4 and 8 Wilson shifts; Lambda=2,1.5,1.25; 512/2048 radial quadrature nodes; and available chain lengths among 8,12,16,24,32,48,64 blocks. The compressed star sometimes contains fewer than 128 modes, so not every requested chain length exists; actual lengths and counts are recorded. No unavailable chain is silently substituted.

At eight blocks and Lambda=1.5, increasing the shift average alone does not repair the short bath:

| Number of shifts | n_d | Error versus continuum |
| --- | --- | --- |
| 1 | 0.850656532 | -0.112160703 |
| 2 | 0.850436340 | -0.112380894 |
| 4 | 0.880194174 | -0.082623060 |
| 8 | 0.869826022 | -0.092991212 |

The finer scan removes chain-tail truncation by diagonalizing the entire compressed one-body star. It uses eight shifts, 2048/8192 radial nodes, and Lambda=1.5,1.25,1.1,1.05. The more accurate radial run gives:

| Lambda | n_d, full star | Error versus continuum | Frobenius error of density matrix |
| --- | --- | --- | --- |
| 1.50 | 0.953238976 | -0.009578258 | 0.00692344 |
| 1.25 | 0.960307588 | -0.002509647 | 0.00181987 |
| 1.10 | 0.962873939 | +0.000056705 | 0.000289806 |
| 1.05 | 0.963297272 | +0.000480038 | 0.0004634 |

Thus substantially closer agreement is attainable. However, the particularly small n_d error at Lambda=1.1 should not be advertised as a certified tolerance: Lambda=1.05 shifts the result again, and radial quadrature still affects the answer. With 2048 nodes, the Lambda=1.05 value is 0.965045583; with 8192 it is 0.963297272. Mesh spread by itself is an inadequate error estimate: different meshes can share a systematic discretization error.

The fine-star scan uses scipy.special.roots_legendre to generate the same Gauss-Legendre quadrature more efficiently, with 400 logarithmic intervals and a smaller compression tolerance (1e-15). These settings are recorded in the driver. The initial scan uses 96 intervals and tolerance 1e-13. Comparison across these two scans is a combined refinement, not an isolated chain-length experiment. Double occupancy is averaged per mesh, not obtained by taking the determinant of the averaged density matrix.

## Physical interpretation

The known U=0 resonance EP uses the analytic continuum hybridization directly and is independent of these finite-chain occupancy errors. In the interacting calculation, the projected kernel is

K_U(zeta) = (zeta-epsilon_d) I - Delta_b(zeta) - Sigma_U(zeta).

Its matrix dimension is only two, but its interaction self-energy depends on the full many-body bath. Accurate local matrix dimensions do not eliminate bath or truncation requirements. The stationary rotating frame makes the reference Hamiltonian time independent; it does not make Sigma_U frequency independent or remove the continuum limit.

Wilson z-averaging reduces discretization errors; it cannot reconstruct excitations omitted by many-body truncation or compensate automatically for a very short chain. Changing physical bandwidth or hybridization W changes the model. Numerical spectral broadening can smooth discrete peaks but does not itself establish continuum convergence. None of these U=0 tests establishes an occupied interacting EP.

## What follows

First establish the noninteracting occupied-bath density and matrix response to a stated tolerance under radial resolution, Lambda, mesh and chain refinement. Then perform the corresponding interacting chain-length and retained-state checks, using complete-basis dynamics or exact sector responses where feasible. Only after that compare retuned EP coordinates and topology. The evidence supports continued calculation; it does not show that all recommended methods have been tried and failed, or that a local-moment EP is impossible.

Reference: R. Zitko, Phys. Rev. B 84, 085142 (2011), https://doi.org/10.1103/PhysRevB.84.085142. This paper distinguishes residual discretization, finite-size and truncation errors even after z-averaging.

## Reproduce

From the package root, with Python, NumPy and SciPy:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 bath_convergence/bath_convergence.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 bath_convergence/fine_star_convergence.py
```

JSON results include every shift and aggregate; timings are environment dependent. No manuscript or public repository was modified.
