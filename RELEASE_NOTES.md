# Version 2.1.0 — 2026-08-11

- Replaced the earlier manuscript presentation with the final focused
  18-page main article and 26-page Supplemental Material.
- Added a causal outgoing-sheet full-\(k\) EP2 certificate for the stationary
  rotating-drive kernel, including local-discriminant winding, pole exchange,
  half-winding, nonzero second derivative, and rank-one geometry guards.
- Added the 99-point SOC continuation family with independently certified
  points at \(\lambda=0.05\), \(0.10\), and \(0.15\).
- Added the exact zero-temperature finite-chain finite-\(U\) continuation on
  four logarithmic-shifted matrix block-Wilson chains. At \(U=0.0025\), the
  12-site calculation has winding \(-1\), pole exchange, half-winding
  \(-1/2\), and an exact finite-chain sum-rule error of `3.77e-15`.
- Added independently reoptimized \(N=10\) and \(N=12\) centers and a common
  256-point contour. The sufficient Rouche ratio is `0.6775 < 1`, preserving
  the enclosed zero count between those tested finite chains.
- Added the exact EOM/Dyson check, \(N_z=4\) to 8 comparison, and an
  \(N_{keep}=960\) kept-space calibration. The kept-space errors are reported
  only as convergence evidence.
- Replotted the two main figures at full width. The local discriminant and
  pole displacement are normalized by explicitly reported positive scales;
  the normalization cannot change their winding.
- Kept the scope explicit: no continuum, thermodynamic, finite-temperature
  FDM-NRG, strong-coupling Kondo, universal interaction-floor, or many-body
  phase-boundary conclusion is inferred.


# Version 2.0.1 — 2026-08-05

- Added a root-level third-party software notice recording the upstream
  NonHermitianNRG repository, pinned revision, Burke–Mitchell publication,
  and upstream MIT license.
- Linked the third-party notice from the code-and-data availability record.
- Added the Burke–Mitchell DOI and corrected two bibliography-provenance
  mismatches: the unrelated Li et al. work was removed from the
  non-Hermitian Kondo citation cluster, and the skin-effect statement now
  cites Yao and Wang, Phys. Rev. Lett. 121, 086803 (2018).
- Rebuilt the checked main and Supplemental bibliographies and PDFs.
- This patch changes attribution and bibliography provenance only; it does
  not change the model, numerical data, figures, or scientific conclusions.



# Version 2.0.0 — 2026-08-04

- Replaced the obsolete frozen Bethe/phase-map interpretation with exact
  one-particle Fisher–Lee, contact-phase, and Wigner–Smith statements.
- Restricted Yang–Baxter integrability to the explicitly stated
  equal-velocity, branchwise-linearized, scalar-hybridization finite-\(U\)
  Anderson submanifold.
- Added the exact dressed-rapidity \(4\times4\) rational contact matrix,
  three-particle YBE validation, and \(GL(2,\mathbb C)\) covariance check.
- Clarified that a CPT/non-unitary rotation preserves the existing rational
  bulk algebra and that the exceptional-point datum is an invertible
  unipotent boundary twist; no new bulk \(R\)-matrix is claimed.
- Added the causal passive-density, resolvent-cancellation, Smith-delay, and
  channel-resolved weak-flow audits.
- Added the operator-resolved finite-\(U\) biorthogonal NRG residue audit and
  Supplemental Fig. S8. The audit validates the nearly traceless nilpotent
  residue direction but rejects a universal pole exponent and complex
  quartic plateau.
- Preserved all 47 main-manuscript references and the checked Supplemental
  bibliography.
- Removed active copies of the obsolete rapidity plot, interaction phase map,
  universal quartic-floor construction, and signed relative-kernel DOS
  interpretation.
- Cross-linked the separate mathematical companion work
  [arXiv:2604.21547](https://arxiv.org/abs/2604.21547), which contains the
  arbitrary-particle RTT/Jordan and projective descendant-root results.

This is a pre-publication repository candidate. Journal target names do not
imply acceptance. Add the version-specific Zenodo DOI only after deposition.

## Historical releases

- `v1.0.1` — licensing clarification and Zenodo archival metadata.
- `v1.0.0` — initial public reproducibility snapshot.
