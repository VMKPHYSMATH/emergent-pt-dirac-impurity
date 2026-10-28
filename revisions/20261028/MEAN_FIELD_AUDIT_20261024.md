# Slave-boson convergence audit - 24 October 2026

**Outcome:** the numerical checks pass within the stated mean-field equations.
This validates numerical solutions and selected local stability; it is not
an error bound relative to an exact interacting impurity calculation.
Candidate version: 2.2.0-rc.3. Nothing was published or submitted by this audit.

## Finite-U matrix KR/Gutzwiller saddle

The model is the normal, zero-temperature, real-spin matrix saddle of the
specified Hermitian stationary rotating-frame bath. The saddle uses the
physical-sheet continuum hybridization. Resonance calculations hold the
saddle fixed while differentiating with respect to complex frequency and
continue the bath to the stated outgoing two-cut sheet. On a control loop,
the saddle is re-solved at every control point.

| Check | Fresh result |
| --- | --- |
| Accepted stored path points re-solved | 98; orders 32 and 64, held-state check at order 96 |
| Largest change of any element of R (32 to 64) | 5.141e-13 |
| Largest absolute change among listed observables/controls | 1.382e-11 in model units |
| Largest held-state normalized residual at order 96 | 5.143e-12 |
| Tail cutoff 1e5 to 1e6 | density change below 2.3e-16; force below 4.4e-18 |
| Analytic radial integral versus adaptive quadrature | 1.537e-16 |
| Density/force versus exact 194-dimensional Hermitian star | 1.139e-12 / 1.481e-14 |
| Independent log-determinant functional derivatives | discrepancy <= 5.066e-10 over tested steps and displaced states |
| Six perturbed initial seeds at the selected filling-one EP | same physical saddle; largest R difference < 3e-15 |
| Local constrained and transverse curvature checks | positive at the selected EPs and occupied reference points |

The path audit covers the low-filling branch to U=0.2, both filling-one control
families and their stored near-fold extensions, and all 57 occupied-path
points to U=0.2. Failed continuation attempts remain marked as failures and
are not counted as accepted points or interpreted as proofs of EP absence.
Some data points overlap between the original and extended paths; 98 is the
number of accepted records checked, not 98 distinct physical parameter sets.
Force residuals use the script's fixed 0.02 normalization; density constraints
are dimensionless. Quoted changes are finite-grid comparisons, not rigorous
error bounds between every quadrature node.

At the selected U=0.0025, n_d=1 EP:
- z_EP = 0.006535134304576 - 0.002983035811337 i;
- epsilon_d = 0.008761698951796, V_O = 0.945864146591530,
  mu = 0.006656973796273, W = 0.1;
- double occupancy D = 0.228945653168; single-occupancy probability = 0.542108693664;
- Z_+ = Z_- = 0.997395502349; R = 0.998696902142 I.

This is a charge-fluctuating filling-one EP, not a demonstrated local-moment
EP. The separately tested U=0.2 occupied point has n_d=0.9940551563 and
D=0.00125994048, but is strongly polarized and its tested poles are split.
Local positive curvature does not prove a unique global variational minimum.

## Pole certificates and finite-chain comparison

The original 128-sample loops were rerun. Four loops were additionally
sampled at 256 intervals and integration order 64:

| Point/control family | Winding | Pole half-winding | Endpoint permutation |
| --- | --- | --- | --- |
| low_fill_U002 | 1.000000 | 0.500000 | (12) |
| low_fill_U02 | 1.000000 | 0.500000 | (12) |
| W_U00025 | -1.000000 | -0.500000 | (12) |
| n1_U00025 | -1.000000 | -0.500000 | (12) |

Every loop retains two open channels, nonzero second derivative, rank-one
center kernel, small pole residuals and unambiguous tracked exchange. Across
the four refined loops, the largest assignment/half-separation ratio is 0.177,
the smallest |F_zz| is above 1.72, the largest pole residual is below 5.4e-21,
and the largest normalized saddle residual is below 6.0e-12.

SciPy reported lack-of-progress flags at 5 loop solves for U=0.02 and 9 for
U=0.2, despite residuals satisfying the explicit acceptance tolerances.
These flags are retained in JSON. No accepted point is justified solely by a
solver's Boolean success flag. The winding is a numerically sampled local
degree diagnostic; it is not an interval-arithmetic proof over all controls.

The nearby U=0.0032 EP was also rechecked beyond the exactly-filling-one fold.
Its existence shows that the constrained fold is not a global end of EPs.
The fold was refined to U=0.0030826663167 in the fixed-W family.

Four N=12-mesh KR comparison points (U=0.001 through 0.0025) were recomputed.
Their saddle and EP residuals pass. The ED centers in that comparison are
stored reference inputs and were not re-diagonalized in this audit. The
historical N=10 to N=12 common-contour summary passes its consistency check;
that is not a fresh ED contour calculation and is not continuum convergence.
A continuum-versus-chain KR difference is not an exact interaction-error bound.

## Scalar Coleman saddle: corrected constant-core benchmark

This is the separate U=infinity, Q=1 spin-scalar benchmark with epsilon_d=-1,
Gamma_0=0.12, cutoff D=4 and T=1e-6. Archive convention:
Delta_0=0.075+0.05 beta and Gamma_PT=0.2 beta. A second calculation retains
the manuscript's fixed Delta_0=0.10 convention; the conventions are labeled.
At beta=0.5 they coincide.

The original grid resolved narrow poles but under-resolved the logarithmic
occupied-band tail. The corrected solver adds logarithmic points, refines
the uniform base grid, tightens the constraint root tolerance and checks
residuals. The legacy solver is retained only for the correction diagnostic.

At the EP the finite-cutoff reference gives **r=0.00820293449** and
**delta_c-1=8.745632e-10**. The old r=0.00980741173 overestimates the
r-squared-dependent width/scale by 42.95% at these parameters.
On the corrected grid the old point has a stationarity residual of about
0.0264; its apparent agreement on the old grid was not adequate validation.

| beta | Independent r | Finest-grid r | Relative r discrepancy |
| --- | --- | --- | --- |
| 0.35 | 0.00875360325 | 0.00875361098 | 8.833e-07 |
| 0.50 | 0.00820293449 | 0.00820294148 | 8.520e-07 |
| 0.55 | 0.00795604253 | 0.00795604929 | 8.485e-07 |

The three grid levels use approximately 22k-34k, 45k-67k and 90k-134k points,
depending on beta. Each grid is separately solved; the residual is also
evaluated with the independent finite-cutoff formula. Finest-grid relative
r errors are below 8.9e-7, independent residuals below 1.4e-7, and on-grid
residuals below 1e-11. The production safeguarded Brent solver converges at
all three tested beta values. Passivity, spectral positivity and the thermal
FDR/charge identity pass within roundoff for the declared Markov closure.
The latter is a consistency check of that closure, not independent proof of
physical thermalization.

At T=0 the primitive is a finite-cutoff logarithm. At finite T the original
digamma expression is a wide-band approximation. The independent audit adds
-log(1+z/D) to restore the lower-edge correction (thermal edge errors are
exponentially small for the tested D/T) and checks it by adaptive real-axis
quadrature. Maximum resulting saddle residual is below 5e-15. The low-T
sequence tends smoothly to r(T=0)=0.008298290454; r(T=1e-6)=0.008202934488.
Temperature dependence is physical within mean field, not a quadrature error.

## Scalar Coleman saddle: full stationary bath

For the full radial stationary bath, the effective-W=0.1 reference gives
n_f=0.03468136476, r=0.98250630290, delta_c=0.000674088041,
epsilon_d(bare)=0.01449626174, W(bare)=0.10178051754. The control-map Jacobian
determinant is 1.129159962, so the local retuning map is nonsingular here.

With bare W=0.1 held fixed, the retuned scalar-saddle EP is
z_EP=0.0117701165832 - 0.00270876327187 i. Its joint residual is 1.31e-14.
Integration orders 48 and 96 differ by less than 8.4e-15. Matsubara sums at
T=0.001, 0.01 and 0.1 use 32k, 64k and 128k terms plus analytic tails; the
largest change from 64k to 128k is 2.3e-12. These stationary-bath r values
must not be confused with the much smaller r of the constant-core benchmark.

## Reproduce and cite

See the root README and reproduce.py. The fresh checks live in
`revisions/20261026/source/archive_additions/slave_boson_audit_20261024/`.
Results include kr_paths.json, coleman.json, stationary_coleman.json,
variational_functional.json and loop_resolution.json, with per-loop records.
The additional KR validation, certificates, chain comparison and fold outputs
remain alongside their producing scripts. The tested environment is Python
3.12.14, NumPy 2.3.5, SciPy 1.17.0 and Matplotlib 3.10.8, with one BLAS thread.

The numerical audit was executed on 24 October. This package now contains
the 26 October focused editorial revision of the main/Supplement, incorporating that
evidence and the requested JNCASR doctoral-affiliation/independent-work note.
The title, scientific figures and numerical results are unchanged. The actual
new fixed archive DOI must be entered before journal upload.
No new repository release or fixed DOI is asserted. Earlier full finite-chain
and residue calculations remain at DOI 10.5281/zenodo.21888925; that fixed
DOI must not be reused for this new snapshot.

The supported wording is: “The reported slave-boson EPs and saddle observables
are numerically converged within the specified mean-field approximations.”
The calculation does not establish a controlled exact local-moment EP,
thermodynamic NRG survival, or pumping-enhanced Kondo screening.
