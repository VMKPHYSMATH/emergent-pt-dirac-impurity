# Reproducibility additions — updated 19 October 2026

These files accompany the revised manuscript. They are not yet part of public Release v2.1.0 or its existing Zenodo DOI. Run commands from the source-package root. Python 3.10+ with NumPy, SciPy and Matplotlib is required; the recorded reruns used Python 3.12, NumPy 2.3.5 and SciPy 1.17.0.

## Checks and figures rerun during the 16 October revision

```sh
OPENBLAS_NUM_THREADS=1 python3 archive_additions/full_k_two_cut_ep_certificate.py --soc .05 --out validation/u0_soc005
OPENBLAS_NUM_THREADS=1 python3 archive_additions/full_k_two_cut_ep_certificate.py --soc .10 --out validation/u0_soc010
python3 archive_additions/pt_phase_diagram.py archive_additions/splitting_diagnostic_data.npz
python3 archive_additions/make_pt_phase_figure.py archive_additions/splitting_diagnostic_data.npz Fig2_splitting_diagnostic.pdf
python3 archive_additions/check_retarded_map.py
OPENBLAS_NUM_THREADS=1 python3 archive_additions/make_smith_revision.py
python3 archive_additions/make_spectral_flow_revision.py
OPENBLAS_NUM_THREADS=1 python3 reproducibility/filling/filling_campaign.py --config reproducibility/filling/configs/submitted_baseline.json --mode submitted --out validation/filling_N10_N12
python3 archive_additions/verify_finiteU_contact_algebra.py
```

The map script retains its historical filename. It computes a real-frequency splitting diagnostic, not a PT phase diagram. Its grid evaluation is slower than the other figure scripts. `check_retarded_map.py` tests five representative positive-frequency points against direct radial quadrature and eta refinement; it is not an interval proof of the whole map.

`make_spectral_flow_revision.py` preserves the v2.1.0 matrix-flow CSV data in `rg_data/` and corrects the scientific labels. The trace-delay curves are recomputed from the complete Markov resolvent. The flow is a diagnostic, not a derived Kondo RG.

The filling command uses the four archived low-band meshes in `reproducibility/filling/reference/submitted_chains/`, overriding the generic mapping choices when `--mode submitted` is selected. N=10 and 12, U=0 and 0.0025 are evaluated. The interacting rows use the same reported N=12 endpoint controls. Do not use the default pilot configuration to reproduce the manuscript table.

## Archived evidence

`full_k_ep_soc_family.py` and its CSV are copied from v2.1.0. The SOC family, full interacting continuation, common-contour data and residue audit remain archived evidence; the complete expensive finite-U pole continuation was not rerun for this editorial revision.

```sh
python3 archive_additions/validate_fullk_n_scaling_gate.py
```

That command validates consistency of the archived summary. Its legacy filenames and status strings contain “Rouche”; the revised manuscript explains the sampled-boundary and Brouwer-degree interpretation. It is not a fresh many-body run or a rigorous bound between contour samples.

For provenance, see `validation/validation_summary.json`, `review/SELF_REVIEW.md`, and the file hashes in `SHA256SUMS`. The full v2.1.0 data remain at https://doi.org/10.5281/zenodo.21888925.

## Final audit and continuum KR additions

`finite_u_kr/README.md` documents the continuum saddle and its comparison with
the bundled finite chains. `final_audit_20261019/README.md` distinguishes the
fresh independent checks from historical exploratory claims. In particular,
run the final nearby-EP check from the source root with:

```sh
OPENBLAS_NUM_THREADS=1 python3 archive_additions/final_audit_20261019/check_beyond_fold.py
OPENBLAS_NUM_THREADS=1 python3 archive_additions/finite_u_kr/scripts/chain_rho.py
```

These commands write their indicated JSON/log outputs when redirected.
The near-one EP is charge fluctuating. A constrained filling-one fold is not
an assertion that all nearby EPs disappear. The separate field/temperature
extension is included only in the audit, not claimed in the manuscript.
The current inventory is `validation/validation_summary.json`; the original
16 October inventory is preserved in `review/history/`.
## Main/SM figure placement update

```sh
python3 archive_additions/finite_u_kr/scripts/make_filling_figures.py
```

This regenerates main Fig. 7 (former S5(a–c)) and SM Fig. S5 (former S5(d))
from unchanged stored results. It performs no new saddle or pole solve.
