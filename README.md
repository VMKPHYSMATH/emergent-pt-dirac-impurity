# Emergent PT Symmetry and Exceptional Points in a Driven Dirac Impurity

Reproducibility code, tabulated data, figure outputs, and manuscript sources for

**Vinayak M. Kulkarni, “Emergent PT Symmetry and Exceptional Points in a Driven Dirac Impurity.”**

The final manuscript uses a passive-PT projected-retarded-kernel interpretation. The
microscopic Hamiltonian is Hermitian; non-Hermiticity appears only after the driven
bath and off-shell auxiliary sector are projected into the retarded impurity kernel.

## Repository contents

- `prb_all_in_one_final.py` - Figs. 1-4 and supplementary numerical diagnostics.
- `bethe_fig5_fig6_final.py` - Figs. 5 and 6.
- `data_fig3_weight_transfer/` - exported Fig. 3/SFig. 3 data.
- `figures/main/` and `figures/supplement/` - final PDF figures.
- `paper/` - final manuscript and Supplemental Material sources and PDFs.
- `CITATION.cff` and `CITATION.bib` - software and manuscript citation metadata.
- `LICENSE` and `LICENSE-DATA` - scoped licenses for code and numerical data.

## Environment

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproduce Figs. 1-4

Generate all outputs:

```bash
python prb_all_in_one_final.py --out figs_unified
```

Generate one main figure:

```bash
python prb_all_in_one_final.py --out figs_unified --which fig3
```

The all-in-one script also writes the numerical `.dat` files used for the spectral and
scale diagnostics.

## Reproduce Figs. 5-6

```bash
python bethe_fig5_fig6_final.py
```

This script writes `fig5_rapidities.pdf/.png` and `fig6_phasediag.pdf/.png` in the
script directory.

## Final notation and interpretation

- `t_av`: Wigner center time; `tau`: relative time; `T_b`: bath temperature.
- `k_max`: momentum cutoff; `D_uv`: ultraviolet energy cutoff.
- `beta_0`: signed frozen drive-control amplitude.
- `tilde beta = |beta_0||b_c|`: gauge-invariant SBMF projected hybridization.
- `Gamma_PT`: passive relative decay imbalance after common damping is removed.
- `Delta_coh`: Kramers-Kronig coherent detuning.
- `s_eff`: interaction-regularized impurity splitting.
- SBMF positive spectral-weight transfer is an independent spectral diagnostic; it is
  not identified with a thermodynamic Kondo temperature.
- FWHM/HWHM values are used only where a positive locally resolved resonance has two
  valid half-maximum crossings.

## Citation

Please cite the associated manuscript and the repository. Ready-to-copy entries are in
`CITATION.bib`; GitHub-compatible metadata are in `CITATION.cff`.

## Repository URL

https://github.com/VMKPHYSMATH/emergent-pt-dirac-impurity

The manuscript and Supplemental Material cite this repository using the key
`kulkarni2026code`.

## License

- The Python source code and supporting software configuration files are
  licensed under the MIT License; see `LICENSE`.
- The numerical data and numerical validation files in
  `data_fig3_weight_transfer/` and `validation_diagnostics/` are licensed
  under the Creative Commons Attribution 4.0 International License
  (CC BY 4.0); see `LICENSE-DATA`.
- The manuscript text, TeX sources, compiled PDFs, and publication figures
  are not covered by the software or data licenses. Copyright in those
  materials remains with the author and is subject to the applicable journal,
  preprint-server, and repository terms.


Validation diagnostics
---------------------------
The `validation_diagnostics/` directory contains the exported condition-number and FDR residual files used to verify the reported values, including the distinction between the projected EP value 0.4975 and the discrete-grid kappa maximum at 0.5025.
