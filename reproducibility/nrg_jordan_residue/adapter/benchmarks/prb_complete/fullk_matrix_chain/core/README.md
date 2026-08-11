# Finite-\(U\) full-\(k\) block-Wilson certificate

This directory contains the solver modules and archived runs supporting the
finite-chain result in the main article and Supplemental Appendix C.

## Certified scope

The load-bearing calculation is an exact zero-temperature correction-vector
response in untruncated fixed-charge sectors on four logarithmic-shifted
12-site \(2\times2\) matrix block-Wilson chains.  At \(U=0.0025\), the local
nonlinear discriminant has winding \(-1\), the two nearby poles exchange, and
their separation has half-winding \(-1/2\).

This directory does not establish an infinite-chain or finite-temperature
FDM/DM-NRG result.  The kept-space calculations are included only as direct
truncation-error calibrations against the exact response on the same finite
chains.

## Contents

- full_k_wilson_preflight.py: matrix spectral-density discretization and
  block-Lanczos chain construction.
- exact_block_chain_response.py: exact finite-chain fixed-charge response.
- exact_zavg_block_chain_response.py: response averaged before forming the
  interaction self-energy.
- fullk_nrg_ep_predictor.py: self-consistent double-zero continuation.
- fullk_nrg_contour_pilot.py: local-discriminant winding and pole-exchange
  gates.
- validate_exact_eom_identity.py: independent EOM/Dyson identity test.
- kept_zavg_block_chain_response.py and validate_kept_against_exact.py:
  kept-space calibration.
- preflight_lowbands_z4 and preflight_lowbands_z8_l15: archived matrix chains
  and reconstruction summaries.
- exact_zavg4_*, exact_zavg8_*, and kept_exact_*: machine-readable outputs
  used in the manuscript.

Run the lightweight archived-output gate from this directory:

    python validate_archived_finiteU_gates.py

The main-figure plotting script is at the submission-package top level and
reads the compact copied JSON summaries there.
