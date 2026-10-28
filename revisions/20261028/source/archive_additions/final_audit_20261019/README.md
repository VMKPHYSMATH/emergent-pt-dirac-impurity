# Independent checks supporting the final 19 October revision

These files preserve the 18–19 October audit and its numerical outputs.
The manuscript corrections have now been applied; statements in the dated
18 October report saying that the source is unchanged describe that earlier
stage. The 17 October SBMF audit ZIP is also historical supporting material.
Current conclusions are in `../../review/SELF_REVIEW.md`.

## Reproduction

Python with NumPy and SciPy is required. Run from the source root:

```sh
OPENBLAS_NUM_THREADS=1 python3 archive_additions/final_audit_20261019/recheck_new_results.py
OPENBLAS_NUM_THREADS=1 python3 archive_additions/final_audit_20261019/check_beyond_fold.py
```

The first script re-solves the low-filling U=0.2 KR certificate and refines
the exactly filling-one fold, then follows the branch without fixing n=1.
It writes `new_results_recheck.json`. The second uses that file and gives the
order-64, 128-point certificate at U=0.0032 in `beyond_fold_certificate.json`.
Both use the bundled `../finite_u_kr` package. The latter check was rerun
from this portable package on 19 October. This is numerical certification
within KR, not an exact interacting result or an interval proof.

## Separate research extension

`check_field_temperature.py` and `check_perpendicular_field.py` with their
JSON outputs audit static-field and width/temperature proposals. The general
tilted-field check includes both field components on the bath. These results
are not included in the submission's scientific claims.

The supplied 19 October archive's original field certificate drivers were
also rerun without changing their kernels; the fresh logs are under
`../../validation/field_temperature_20261019/`. They reproduce the original
results, including cases whose strict solver gates fail despite exchange.
A resolved independent contour can pass where a particular original contour
does not; this distinction is retained. Search failures are not nonexistence
proofs, static-field equivalence is only approximate, and real-part threshold
crossing does not prove a complex branch-point collision.

The original field code uses the full Zeeman splitting 2|b|. The preliminary
factor-of-two convention concern in the earlier audit is resolved on reading
those scripts. Thermal numbers remain linewidth-resolution estimates, with
an explicitly chosen conversion from dimensionless units.
