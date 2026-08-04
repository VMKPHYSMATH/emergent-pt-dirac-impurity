# Local validation report — 2026-08-04

## Manuscript build

- `paper/main.tex`: 18 pages.
- `paper/supplemental_material.tex`: 21 pages.
- Final logs contain no undefined citations/references or LaTeX/package errors.
- All 18+21 generated pages raster-match the checked publication PDFs.
- Main bibliography: 47 entries; Supplemental bibliography: 14 entries.

## Exact and numerical gates

- finite-\(U\) contact YBE residual: `1.144e-15`;
- \(GL(2,\mathbb C)\) covariance residual: `4.475e-16`;
- CPT diagonalization residual: `1.655e-16`;
- dressed-rapidity factorization residual: `4.857e-17`;
- singlet/triplet eigenvalue residuals: zero;
- exact contact/Smith symbolic identities: PASS;
- full-matrix scattering-phase/Smith identity: PASS;
- independent NRG creation/annihilation propagation: PASS;
- NRG adapter/install self-checks: PASS;
- comprehensive Jordan-analysis self-test: PASS.

The NRG archive reports a reliable tracked-pair fraction `0.9939393939`,
median Jordan alignment `0.999999630827`, normalized trace fraction
`8.356705e-4`, and nilpotent mismatch `7.3179e-7`. The exponent and
complex-quartic convergence gates fail; the manuscript reports those failures
and uses only the matrix-residue conclusion.

## Repository checks

- 35 Python files parsed successfully.
- `CITATION.cff` and the GitHub Actions workflow parse as YAML.
- No file exceeds GitHub's 100 MB per-file limit.
- The active snapshot contains no obsolete rapidity/phase-map figure or
  universal quartic-floor implementation.
- The repository has not been pushed.
