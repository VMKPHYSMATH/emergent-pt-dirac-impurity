# Validation and scope - 28 October 2026

The manuscript separates the stationary-drive result from model-specific
interaction evidence. Numerical acceptance does not prove convergence to the
exact interacting continuum problem.

| Evidence | Established level | Remaining boundary |
| --- | --- | --- |
| U=0 full radial bath | Double zero, rank one, Fzz nonzero, physical passivity, pole exchange | Specified radial channel model and resonance sheet |
| Four-shift N=10/12 historical embedding | Matched sampled topology at a nearly empty endpoint | Not an infinite-bath self-energy or continuation proof |
| Continuum matrix KR | Re-solved low-filling and filling-one EPs; 98 accepted records refined; four 256-step loops | Mean-field approximation; n=1 point is charge fluctuating |
| Scalar Coleman | Corrected logarithmic grid and independent integral checks | Separate U=infinity scalar saddle |
| Occupied finite embeddings | Separate EPs at 6/8/10 two-mode bath blocks; sided/loop checks at 6/8 | Centers and fillings vary substantially with bath length |
| Optimized passive finite baths | Stable near-unit filling and small double occupancy | Self-energy changes by 0.330 reference widths at 10 to 12 orbitals; fit choice remains important |
| Scalar-bath PT control | Exact dimer residue and 14 untruncated finite cases; loops and negative control | Different model and biorthogonal state, no original-model continuum inference |

Detailed retained mean-field results are in
`revisions/20261028/MEAN_FIELD_AUDIT_20261024.md`. New raw numerical records
and reports are in `revisions/20261028/checks_20261028/`. The 98 count is
accepted stored records, including overlaps between paths, not 98 distinct
parameter sets. Historical dates inside reports identify when each test ran.

`python reproduce.py --verify-stored` checks file integrity and selected
machine-readable acceptance conditions. The finite-bath convergence failure
is checked as an honestly reported limitation, not converted into a passing
continuum result. `--quick` re-solves its explicitly listed subset; no full
finite-U continuum or thermodynamic calculation is claimed.

The manuscript builds at 19 pages plus a 30-page Supplement, retaining seven
main and five supplementary figures. All Supplement citation keys are in the
main bibliography; no citation or reference is unresolved. TeX's known bundled
REVTeX/nameref compatibility notices are distinct from layout/reference errors.

## Fresh assembled-package rerun

The `--quick` run completed successfully on 28 October in a separate copy of
the assembled package. It re-solved the 98-record KR grid check, independent
functional derivatives, the U=0 full-bath EP, contact algebra and exact PT
control verification. Stored checks then passed. See
`revisions/20261028/REPRODUCTION_20261028.txt`. The broader ED/bath data are
retained calculations, not a new continuum-convergence claim.
