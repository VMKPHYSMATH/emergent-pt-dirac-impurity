# Local validation report — 2026-08-11

## Scientific gates

- The stationary lambda=0.05 full-k causal pole EP remains certified at
  U=0 and is continued smoothly through U=0.0010, 0.0015, 0.0020, and
  0.0025.
- At U=0.0025 the exact four-shift, 12-site complete-basis center is
  z_EP=0.01177669854-0.00225815426 i,
  epsilon_d=0.01440607426, and W=0.09036305152.
- The interaction is small relative to the radial cutoff,
  U/D_uv=3.18e-3, but moderate relative to the retuned local width,
  U/Gamma_bar_self=1.225.
- The interacting local discriminant has winding
  -1.0000000000001215, endpoint permutation [1,0], and pole-separation
  half-winding -0.5000000000000521.
- The minimum boundary modulus is 1.5060e-9, the maximum phase increment is
  0.27725, and the minimum absolute F_zz is 1.71153.
- The maximum pole residual is 2.5452e-21 and the maximum assignment to
  half-separation ratio is 0.42415, below the 0.5 guard.
- The exact finite-chain sum-rule error is 3.7682e-15.
- The independent EOM/Dyson absolute and relative residuals are 4.67e-16
  and 1.32e-11.
- Nz=4 to 8 changes Re z, Im z, epsilon_d, and W by 0.022%, 0.71%,
  0.14%, and 0.33%.
- At Nkeep=960 the maximum kept-space Sigma_U and sum-rule deviations are
  4.18% and 1.40%.  They are reported only as convergence evidence.
- Independently reoptimized N=10 and N=12 centers differ by at most
  8.737e-5 absolutely and 3.746e-3 relatively.
- On the shared 256-point contour, both finite chains have winding -1,
  endpoint permutation [1,0], and pole-separation half-winding -1/2.
- The common-contour sufficient inequality is
  sup_C |D_12-D_10|=1.638e-9 < min_C |D_10|=2.417e-9, giving a Rouche ratio
  0.6775<1.  The same gate passes at 128 points with ratio 0.6758.

## Passed executables

- verify_finiteU_contact_algebra.py: PASS.
  Yang--Baxter-equation (YBE) residual 1.144e-15; GL(2,C) covariance
  4.475e-16; metric diagonalization 1.655e-16; dressed-rapidity
  factorization 4.857e-17.
- full_k_two_cut_ep_certificate.py: PASS.
  U=0 nonlinear EP and topology certified; two-cut sheet residual
  4.024e-9; both physical rate eigenvalues positive.
- full_k_ep_soc_family.py: PASS.
  99 continued points over 0.02 <= lambda <= 0.1578; independent
  certificates at lambda=0.05, 0.10, and 0.15.
- finiteU_fullk_reproducibility/validate_archived_finiteU_gates.py: PASS.
- fullk_n_scaling_gate_summary.json: FINITE_N_COMMON_CONTOUR_ROUCHE_PASS.
- validate_fullk_n_scaling_gate.py: PASS.
- make_finiteU_fullk_exact_figure.py: PASS; vector PDF and 400-dpi PNG
  regenerated from the archived summaries.

## Certificate and document checks

- The contour certificate is stated as a two-real-variable Brouwer-degree
  calculation: a nonzero boundary winding of the branch-tracked local
  discriminant forces an enclosed double zero.  The nonzero `F_zz`, rank-one
  inverse-kernel, pole exchange, and half-winding gates distinguish an EP2
  from a higher-order or scalar degeneracy.
- The biorthogonal cancellation is proved by the complete resolvent identity;
  the finite-U NRG residue and sum-rule calculations are identified only as
  independent physical-basis consistency tests.
- The expanded quartic, isolated-channel, and contact/Yang--Baxter derivations
  are retained in the Supplemental Material rather than repeated in the main
  article.
- Main article: 18 pages.  The one-page increase is retained deliberately so
  that both full-width diagnostic figures remain legible at print scale.
- Supplemental Material: 26 pages.
- The PDFs were rendered page by page and visually inspected.
- Main Fig. 1 is replotted as a four-panel full-width certificate.  The local
  discriminant is displayed as the dimensionless quantity
  `Dloc_hat=Dloc/D_star`, while the absolute scale
  `D_star=4.603e-9` is retained in the panel and caption.  The positive
  normalization does not alter the contour winding or Brouwer degree.
- Main Fig. 2 combines the frozen-core EP and matched control in one
  print-legible comparison.  Its time-domain data are separated into stacked
  traces instead of overlapping phase portraits; the Bloch-trajectory
  diagnostic remains in Supplemental Fig. S1.
- All main and Supplemental figures are legible and have no clipping or
  overfull boxes.
- Final logs contain no undefined citations, undefined references, fatal
  LaTeX errors, or overfull horizontal boxes.  REVTeX emits two benign
  2.67-point vertical-box warnings while balancing the final bibliography;
  page rendering confirms that no text or reference is clipped.
- The bundled REVTeX class emits harmless natbib multiply-defined warnings
  when resolved AUX bibliography entries are reread; each checked BBL has
  unique keys and both PDFs contain complete reference lists.
- Every distributed file listed in SHA256SUMS.txt verifies successfully.

## Scope gate

The positive statement is an exact zero-temperature finite-chain topology
certificate with a pairwise Rouch\'e stability gate over N=10 and N=12.  It
is not promoted to an infinite-chain, thermodynamic,
finite-temperature full-density-matrix NRG (FDM-NRG), density-matrix NRG
(DM-NRG), strong-coupling Kondo, or phase-boundary
result.  The short-chain four-band auxiliary-pair test is unresolved because
different logarithmic shifts select charges 1,2,2,1 and is interpreted as
neither a pass nor a failure.  The nonlinear causal pole EP is not identified
with the nilpotent Markov-core EP and does not imply full-momentum
YBE, $RLL$, or $RTT$ integrability or many-body Jordan amplification.
