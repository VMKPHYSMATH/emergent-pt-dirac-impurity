# Finite-U biorthogonal NRG residue audit

This archive reproduces the narrowly scoped numerical check reported in Appendix K and Fig. S8 of the Supplemental Material for manuscript Driven-Dirac impurity.

## What the audit establishes

For the scalar-hybridization Wilson-chain control specified in the manuscript, the independently propagated transition operators produce two tracked complex poles with a nearly traceless residue-difference matrix aligned with the nilpotent exceptional-point direction. The recorded gate summary gives

- reliable tracked-pair fraction: `0.9939393939`;
- median Jordan-direction alignment: `0.999999630827`;
- median normalized trace fraction: `0.0008356705`;
- median nilpotent mismatch: `7.3179e-7`.

The same convergence audit does **not** validate a universal pole-splitting exponent or a universal complex quartic coefficient. The supplied data must therefore not be interpreted as a thermodynamic Kondo-scale enhancement, an interaction-induced exceptional-point floor, or a many-body phase boundary.

## Included material

- `adapter/`: Julia implementation, independent annihilation/creation transition-operator propagation, regression tests, scan scripts, and exported scan data;
- `adapter/output/jordan_comprehensive/analysis/gate_summary.json`: machine-readable gate decision;
- `adapter/output/jordan_comprehensive/analysis/`: processed tables and diagnostic plots;
- `figure/make_nrg_jordan_structural.py`: generator for the publication figure;
- `figure/Fig_NRG_Jordan_structural_clean.pdf`: vector figure used as Fig. S8;
- `figure/Fig_NRG_Jordan_structural_clean.png`: raster preview.

## Reproduction

From the `adapter` directory, instantiate the Julia environment and run

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. benchmarks/prb_complete/scripts/run_jordan_comprehensive_scan.jl comprehensive core scalar output/jordan_comprehensive
python3 benchmarks/prb_complete/scripts/analyze_jordan_comprehensive.py output/jordan_comprehensive --out output/jordan_comprehensive/analysis --iterations 4 5 6
```

The Python dependencies used by the analysis are listed in `adapter/benchmarks/prb_complete/requirements.txt`. To regenerate Fig. S8 from the exported analysis, run from this archive's top directory:

```bash
python3 figure/make_nrg_jordan_structural.py \
  --analysis-dir adapter/output/jordan_comprehensive/analysis \
  --output-dir figure/rebuilt
```

The archive already contains the complete exported scan and analysis, so the figure can be regenerated without rerunning the Julia scan.

## Scope and provenance

The implementation is an adapter/benchmark layer whose upstream provenance and modifications are recorded in `adapter/UPSTREAM_NOTICE.md`, `adapter/PATCH_SCOPE.md`, and `adapter/BIORTH_OPERATOR_FIX.md`. It is a controlled numerical residue audit, not a claim of exact integrability for the full curved, frequency-dependent driven kernel.
