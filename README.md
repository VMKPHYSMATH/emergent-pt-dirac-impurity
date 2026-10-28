# Emergent PT symmetry and exceptional points in a driven Dirac impurity

Vinayak M. Kulkarni - revision candidate **2026-10-28**.

The latest reproducibility code, numerical data and validation outputs are in
[`revisions/20261028/`](revisions/20261028/). This candidate retains the original
stationary-drive microscopic model. The historical v2.1.0 files and tags remain
separate; this revision has not been assigned a new Zenodo DOI or journal decision.

## Scientific result and scope

A stationary rotation of remote fermionic orbitals tunes a nonlinear
exceptional point of two decaying impurity resonances in a specified Hermitian
finite-band radial SOC model. The noninteracting certificate retains the full
frequency dependence and both open helicity cuts. It checks a double zero,
rank-one inverse kernel, passivity on the physical sheet and pole exchange.
SOC continuation supplies an EP family with retuned controls.

Interactions are studied in the same bath at two stated approximation levels:
an embedding from exact finite-chain responses, and a continuum matrix
Kotliar-Ruckenstein saddle. The latter has an EP at mean filling one, but its
substantial double occupancy identifies a charge-fluctuating state. Increasing
U/linewidth on a nearly empty branch is not evidence of a Kondo regime.
The occupied-bath tests do **not** establish a converged local-moment continuum
EP. The exact scalar-bath PT dimer in the Supplement is a separate comparison.

## Read and reproduce

- [Validation report](VALIDATION_REPORT.md), including the distinction between
  freshly run checks and retained evidence.
- [New finite-bath and symmetry checks](revisions/20261028/checks_20261028/README.md).
- [Detailed mean-field audit](revisions/20261028/MEAN_FIELD_AUDIT_20261024.md).
- [Release notes](RELEASE_NOTES.md) and [archive instructions](UPLOAD_GUIDE.md).

```bash
python -m pip install -r requirements.txt
python reproduce.py --verify-stored
python reproduce.py --quick
# Optional, more extensive mean-field rerun:
python reproduce.py --full
```

`--verify-stored` checks hashes and stored numerical acceptance tests without
re-solving equations. `--quick` reruns the 98-record KR audit, independent
functional derivatives, U=0 certificate, contact algebra and exact-control
verification. `--full` adds the existing scalar-saddle and KR loop/stability
checks; it does not repeat every expensive finite-U bath fit or ED campaign.
Those commands are documented in their individual folders. Reruns overwrite
outputs in the working copy; preserve the downloaded candidate for hash checks.
Use one BLAS thread. Python/library versions are in `environment.json`.

Manuscript sources, manuscript PDFs and editorial letters are maintained separately
and are not part of this code revision. Future uploads are restricted to
reproducibility code, data, figures and validation documentation.

## Historical archive and rights

The August 11 snapshot is [v2.1.0](https://github.com/VMKPHYSMATH/emergent-pt-dirac-impurity/releases/tag/v2.1.0),
fixed [DOI 10.5281/zenodo.21888925](https://doi.org/10.5281/zenodo.21888925).
It contains historical ED/NRG outputs that are referenced, not relabeled as new
calculations. The all-versions record is
[10.5281/zenodo.21434682](https://doi.org/10.5281/zenodo.21434682).
Do not assign the old fixed DOI to the current bytes.

Software: MIT. Specified numerical data: CC BY 4.0 (`LICENSE-DATA`).
Manuscript text, TeX, PDFs and publication figures retain author copyright;
bundled third-party TeX files retain their notices. Citation metadata identifies
this as a dated candidate until a fixed release and deposit exist.
