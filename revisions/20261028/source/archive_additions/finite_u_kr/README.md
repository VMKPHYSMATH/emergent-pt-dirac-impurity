# Numerical audit update - 24 October 2026

See `../slave_boson_audit_20261024/` and the root VALIDATION_REPORT.md for
freshly rerun convergence. `scripts/saddle_gate_copy.py` now includes the
logarithmic grid correction; `saddle_gate_legacy.py` preserves the old grid
solely for diagnosing its error. `probe_grid.py` explicitly imports legacy.
All 98 accepted stored KR path points were re-solved; failed attempts remain
identified and are not interpreted as proofs that an EP is absent.

# Finite-U Kotliar–Ruckenstein saddle of the continuum bath

19 October 2026 · LM19711J · supports main Secs. III F–G, Figs. 6–7, SM App. N and Fig. S5 and the corrected SM App. D 1.
Author-review material: not part of the public v2.1.0 release.

## Layout

- `kr_package/`: the finite-U matrix KR/Gutzwiller saddle package (17 October), with its own README, note source and JSON results. `kr_package/src/matrix_saddle.py` is the saddle; `full_k_two_cut_ep_certificate.py` is the manuscript's U=0 bath kernel.
- `scripts/`: drivers and checks used for the manuscript (listed below).
- `results/`: JSON outputs of the drivers.
- `fig/` (created on demand): rendered main Fig. 6 and SM Fig. S5.

## Scripts

| Script | Purpose | Output |
|---|---|---|
| `kr_certificate.py` | Large-U/linewidth, low-filling winding/pole-exchange certificates at U = 0.02, 0.2 (μ = 0, W fixed), saddle re-solved on each loop point; local stability; threshold and passivity | `results/cert_mu0_VO_U*.json`, `cert_summary.json` |
| `kr_on_chain.py`, `compare_bath.py` | KR saddle on the archived four z-shifted 12-site chains vs the historical finite-chain embedding centers; continuum (ε_d, W) branch; certificate on the main Fig. 5 ellipse | `results/compare_bath.json`, `cert_cont_W_U0.0025.json` |
| `extend_n1.py`, `fold_n1.py`, `fold_check_order.py` | Mean-filling-one EP branch: continuation in U and fold with U as an unknown; quadrature-order check | `results/ep_n1_extended_*.json`, `fold_n1_*.json` |
| `sb_kondo_closed_form.py`, `probe_grid.py`, `saddle_gate_copy.py` | U → ∞ saddle: closed form at T = 0, digamma form at T > 0, corrected grid; copy of the archived solver | printed tables (SM App. D 1) |
| `slopes_vs_hf.py`, `chain_rho.py`, `sbmf_fullk*.py`, `fullk_cert.py` | Independent first-order (Hartree–Fock) rates with continuum and chain density matrices | printed tables (SM App. N 4) |
| `make_fig5.py`, `make_filling_figures.py` | Render main Figs. 6–7 and SM Fig. S5 from unchanged `results/` and `kr_package/results/` data | `fig/`; filling PDFs also copied to the source root |
| `audit_numbers.py`, `hess_vec.py` | Recompute every number quoted in the manuscript; soft Hessian directions | printed |

## Running

Python 3.11 or later with NumPy, SciPy and Matplotlib. Run from `scripts/`:

```bash
export OPENBLAS_NUM_THREADS=1
python3 kr_certificate.py
python3 compare_bath.py          # bundled chains and reference ED centers
python3 extend_n1.py V_O 0.1 ../results/ep_n1_extended_VO.json
python3 extend_n1.py W 0.1 ../results/ep_n1_extended_W.json
python3 fold_n1.py VO ../results/fold_n1_VO.json
python3 fold_n1.py W ../results/fold_n1_W.json 0.0002
python3 make_fig5.py
python3 make_filling_figures.py
python3 audit_numbers.py
```

`compare_bath.py`, `kr_on_chain.py` and `chain_rho.py` default to the four
bundled meshes in `../../reproducibility/filling/reference/submitted_chains`
(relative to this package) and `reference_ed_centers.json`. The latter copies
the ED center values from the archived comparison output, without rerunning ED.
Optionally set EPT_REPO to an existing public repository root to use its
`reproducibility/full_k_finite_u/` inputs. Leave it unset for the self-contained
submission package.

## Scope

These are T = 0 mean-field saddle results for the specified rotating-frame ground state. They do not bound the
error of the KR approximation relative to the exact impurity, do not describe a Kondo resonance, and do not
establish a local-moment EP. The fold at mean filling one is established for two control families only.

## Final interpretation and nearby EP

The ED estimator averages G and Q before forming Q G^{-1}; this KR comparison
averages individual inverse-kernel corrections. Their first-order slopes
therefore differ slightly (−4.497758 vs −4.501190). Do not treat their
difference as an exact method-error bound or as necessarily second order.
The additive chain-to-continuum transferred estimate is heuristic and has no
independent pole certificate. Use the actual continuum KR result instead.

The exactly filling-one branches have folds, but a nearby charge-fluctuating
EP at U=0.0032, n_d=0.985869, D=0.213296 is numerically verified in
`../final_audit_20261019/beyond_fold_certificate.json`. Its driver is included
there. This does not establish a local-moment EP or validate KR as an exact
strong-correlation method. The corrected Dyson-sign expression is in SM N7.

The former four-panel S5 has been split: `Fig6_kr_filling_one.pdf` is main
Fig. 7 and `SFig5_kr_occupied_path.pdf` is SM Fig. S5. The historical driver
name `make_figS5.py` now forwards to `make_filling_figures.py`. The old four-panel layout is not part of this public candidate.
The plotting change leaves all saddle and pole data unchanged.
