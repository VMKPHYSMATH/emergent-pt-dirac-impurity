# Reproducibility gates

The active v2 checks are organized by scientific claim.

| Directory or script | What it establishes | What it does not establish |
|---|---|---|
| `causal_smith_rg/channel_resolved_rg_gate.py` | Passive spectral positivity, sum rule, basis covariance, one-loop matrix RG, step refinement, and exact inverse-flow agreement | A universal strong-coupling solution or enhanced thermodynamic Kondo temperature |
| `causal_smith_rg/smith_delay_rg_gate.py` | Unitary Fisher–Lee scattering and Hermitian Wigner–Smith delay in the passive window | An independent Petermann or Kondo-scale multiplier |
| `causal_smith_rg/resolvent_cancellation_gate.py` | Cancellation of singular projectors in the complete pole and finite-\(U\) charge resolvents | A divergent observable at the EP |
| `finite_u_contact/verify_finiteU_contact_algebra.py` | Explicit rational \(4\times4\) contact matrix, CPT covariance, and three-particle YBE | Integrability of the full curved, energy-dependent microscopic kernel |
| `bethe_smith_phase/` | Exact one-particle phase quantization for the full two-channel scattering matrix | A multiparticle Bethe ansatz |
| `low_temperature_saddle/` | A converged low-\(T\) saddle with residual, seed, grid, and continuity checks | A finite-\(r\) solution at the manuscript reference temperature \(T_b=0.1\) |
| `nrg_jordan_residue/` | Independent creation/annihilation transition-operator propagation and a nearly traceless finite-\(U\) residue aligned with the nilpotent EP direction | A universal pole exponent, complex quartic plateau, interaction floor, thermodynamic Kondo enhancement, or phase boundary |
| `frozen_kernel_figures.py` | Retained frozen Figs. 1, 2, and S1 | Physical-DOS, RG, or frozen-Bethe evidence |

Each reference-output directory contains a machine-readable summary and a
`SHA256SUMS` file. Run `sha256sum -c SHA256SUMS` from that directory to
verify the archive.
