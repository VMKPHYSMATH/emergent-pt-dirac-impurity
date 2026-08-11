# Full-k causal and finite-U EP certificates

This directory contains the code, archived summaries, tabulated contours, and
figure generators used for main Fig. 1 and Supplemental Figs. S9--S10.

The stationary rotating-drive calculation first certifies an EP2 of the
frequency-dependent outgoing-sheet full-k kernel.  The local discriminant is
evaluated on a continuously tracked critical-pole branch satisfying
`dF/dz = 0`.  Its nonzero boundary winding is the Brouwer degree of the map
from the two real controls to the complex discriminant and therefore forces an
enclosed zero.  Nonzero `F_zz`, rank-one inverse-kernel geometry, pole
exchange, and half-winding distinguish the zero as an EP2.

The finite-U calculation represents the positive 2x2 bath spectral measure by
four logarithmic-shifted matrix block-Wilson chains.  The exact
zero-temperature fixed-charge response gives `G` and `F`, with
`Sigma_U = U F G^{-1}`.  The archived U=0.0025 contour has winding -1, pole
exchange, and half-winding -1/2.  Independently reoptimized N=10 and N=12
centers retain those invariants on a common 256-point contour, and

```text
sup_C |D_12-D_10| / min_C |D_10| = 0.6775 < 1.
```

This sufficient pairwise Rouche inequality preserves the enclosed zero count
between the two tested finite chains.  It does not establish an infinite-chain
or thermodynamic limit.

Run the lightweight archived gates from the repository root:

```bash
python reproducibility/full_k_finite_u/finiteU_fullk_reproducibility/validate_archived_finiteU_gates.py
python reproducibility/full_k_finite_u/validate_fullk_n_scaling_gate.py
```

The complete scaling driver and analyzer are also installed under
`reproducibility/nrg_jordan_residue/adapter/benchmarks/prb_complete/`.

Scope: these are exact finite-chain, zero-temperature results.  They are not
finite-temperature FDM-NRG or DM-NRG calculations, do not prove the continuum
limit, and do not establish a strong-coupling Kondo fixed point or a phase
boundary.
