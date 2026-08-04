# Emergent PT Symmetry and Exceptional Points in a Driven Dirac Impurity

Reproducibility code, reference outputs, figures, and the final checked TeX
sources for:

> Vinayak M. Kulkarni, “Emergent PT Symmetry and Exceptional Points in a
> Driven Dirac Impurity” (2026), [arXiv:2505.17811](https://arxiv.org/abs/2505.17811).

This repository is a preprint and journal-revision archive. A journal name in
the manuscript history identifies a submission target and does not imply
acceptance or publication.

## Scientific scope

The microscopic driven Hamiltonian is Hermitian. Non-Hermiticity appears only
after projection of the bath and off-shell auxiliary modes into a passive
retarded impurity kernel. The repository keeps four logically distinct results
separate:

1. The complete curved, frequency-dependent kernel obeys exact one-particle
   Fisher–Lee, contact-phase, and Wigner–Smith identities.
2. A controlled equal-velocity, branchwise-linearized, scalar-hybridization
   submanifold has the standard finite-\(U\) Anderson contact \(R\)-matrix.
   Its rational dressed-rapidity form satisfies YBE, and a non-unitary
   CPT/metric similarity preserves the same bulk algebra by
   \(GL(2,\mathbb C)\) covariance. It does not create a new bulk \(R\)-matrix.
3. At the exceptional point the diagonalizer is singular, while the
   finite-volume datum is an invertible unipotent boundary twist. The
   arbitrary-particle RLL/RTT and many-body Jordan results are developed in
   the separate companion work
   [arXiv:2604.21547](https://arxiv.org/abs/2604.21547).
4. The operator-resolved biorthogonal NRG audit validates a nearly traceless
   transition-residue matrix aligned with the nilpotent EP direction. It does
   **not** validate a universal pole exponent, quartic coefficient,
   interaction-induced EP floor, thermodynamic Kondo enhancement, or phase
   boundary.

For the complete passive embedding,

\[
\mathcal A(\omega)=\frac{i}{2\pi}\left[G^R(\omega)-G^A(\omega)\right]
=\frac{1}{2\pi}G^R(\omega)\Gamma_{\rm bath}(\omega)G^A(\omega)\succeq0.
\]

The dense causal scan gives a positive strict pre-EP minimum channel-density
eigenvalue of `2.4101604576062974e-2` over the stated manuscript window. The
exact local/core EP is `beta_0 = 0.50`; `0.4975` is only the finite-momentum
operational Schur-tracked minimum on the plotted path.

## Repository map

- `paper/` — checked main and Supplemental TeX sources, prebuilt
  bibliographies, compiled PDFs, figures, and bundled REVTeX support files.
- `figures/main/` — main Figs. 1–2.
- `figures/supplement/` — Supplemental Figs. S1–S8.
- `reproducibility/causal_smith_rg/` — passive-density, RG, resolvent,
  Fisher–Lee, and Wigner–Smith gates and reference output.
- `reproducibility/low_temperature_saddle/` — low-temperature saddle,
  grid/seed, and charge-continuity checks.
- `reproducibility/finite_u_contact/` — explicit finite-\(U\) rational
  \(4\times4\) contact matrix, CPT covariance, and three-particle YBE check.
- `reproducibility/nrg_jordan_residue/` — Julia adapter, independent
  creation/annihilation transition-operator propagation, complete exported
  scan, gate summary, and Fig. S8 generator.
- `reproducibility/bethe_smith_phase/` — one-particle scattering-phase
  quantization check; it is not a multiparticle Bethe-ansatz construction.
- `reproducibility/analytic_checks/` — exact contact and Smith identities.
- `docs/PUSH_INSTRUCTIONS.md` — review, commit, tag, and release checklist.
- `VALIDATION_REPORT.md` — final local build and residual audit.

Obsolete frozen “Bethe rapidity,” interaction phase-map, universal quartic
floor, and signed relative-kernel DOS claims are excluded from this snapshot.
They must not be restored as conclusions of the revised manuscript.

## Python environment

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Principal checks

```bash
python reproducibility/finite_u_contact/verify_finiteU_contact_algebra.py
python reproducibility/analytic_checks/contact_and_smith_identity_check.py
python reproducibility/bethe_smith_phase/bethe_smith_phase_gate.py
python reproducibility/causal_smith_rg/channel_resolved_rg_gate.py --out regenerated_rg
python reproducibility/causal_smith_rg/resolvent_cancellation_gate.py --out regenerated_resolvent
python reproducibility/causal_smith_rg/smith_delay_rg_gate.py \
  --out regenerated_smith \
  --rg-dir reproducibility/causal_smith_rg/reference_rg_output
```

The low-temperature saddle is more expensive:

```bash
python reproducibility/low_temperature_saddle/self_consistent_saddle_gate.py \
  --out regenerated_low_temperature
```

The lightweight NRG installation and analysis checks are:

```bash
cd reproducibility/nrg_jordan_residue/adapter
python benchmarks/prb_complete/tests/validate_biorth_fix.py
python benchmarks/prb_complete/tests/validate_install.py
python benchmarks/prb_complete/tests/validate_jordan_comprehensive.py
```

The complete NRG rerun requires Julia and is documented in
`reproducibility/nrg_jordan_residue/README.md`. The archived gate summary gives
a tracked-pair fraction `0.9939393939`, median Jordan alignment
`0.999999630827`, median normalized trace fraction `8.356705e-4`, and median
nilpotent mismatch `7.3179e-7`. The exponent and complex-quartic gates fail and
are reported as failures.

## Compile the manuscript

Prebuilt `.bbl` files are included:

```bash
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error supplemental_material.tex
pdflatex -interaction=nonstopmode -halt-on-error supplemental_material.tex
```

The checked publication copies are `paper/manuscript.pdf` and
`paper/supplement_checked.pdf`; compilation writes disposable `main.pdf` and
`supplemental_material.pdf` files.

## Citation and archival DOI

Copy-ready metadata are provided in `CITATION.cff` and `CITATION.bib`.
The stable Zenodo concept DOI for all repository versions is
[10.5281/zenodo.21434682](https://doi.org/10.5281/zenodo.21434682). The
version-specific DOI for v2 should be added only after the verified archive is
deposited.

## License

- Python and Julia source and supporting software configuration: MIT
  (`LICENSE`).
- Numerical data and validation outputs: CC BY 4.0 (`LICENSE-DATA`).
- Manuscript text, TeX, compiled PDFs, and publication figures are not covered
  by those software/data licenses; copyright remains with the author subject
  to the applicable preprint, journal, and repository terms.
