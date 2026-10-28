# Full-k two-cut nonlinear pole EP certificate

## Certified witness

- `lambda = 0.10`, `W = 0.10`,
  `K = pi/4`.
- `epsilon_O = 120`, `Omega = 60`.
- Auxiliary vertex `V = 1.833184476735`.
- Pole `z_EP = 0.020357014817 -0.002928554068 i`.
- Real impurity gate `epsilon = 0.023234134085`.

## Algebraic gates

- `|F| = 4.372e-20`.
- `|F_z| = 3.950e-16`.
- `F_zz = 1.810243053095 -0.337949513813 i`.
- Kernel singular values: `1.955e-03` and
  `2.233e-17`.
- Local-discriminant winding: `1.000000000000`.
- Endpoint pole permutation: `[1, 0]`.
- Pole half-winding: `0.500000000000`.
- Maximum tracking-step/half-separation ratio:
  `0.099975`.

## Causal and microscopic gates

- Analytic integral versus adaptive quadrature:
  `6.969e-19`.
- Two-cut sheet-continuity residual:
  `7.795e-09`.
- Physical rate eigenvalues:
  `0.00153401` and
  `0.00346599`.
- Passivity margin `Gamma_min/Gamma_bar =
  0.613604`.
- Threshold gap in pole widths:
  `0.576784`.
- Exact SOC matrix-shift identity residual:
  `3.923e-16`.
- Off-shell ratio `V/(epsilon_O-Omega) = 0.03055`.
- `Omega/K = 76.394`.
- Empty-auxiliary thermal exponent `(epsilon_O-Omega)/T_b =
  600`.

The exact frequency dependence of both the spin-even and spin-odd auxiliary
terms is retained.  The constant value of the spin-even principal-value term
at the Fermi energy is absorbed into the bath chemical potential; its
remaining value at the pole is reported in the JSON output.

## Interpretation

This is a genuine nonlinear continuum pole EP: `F=F_z=0`, the nonlinear
kernel has geometric multiplicity one, `F_zz` is nonzero, the local degree is
nonzero, and the two poles exchange on a closed control loop.  It establishes
a controlled microscopic possibility for the full curved momentum-summed
kernel.

It is **not** the nilpotent constant-matrix EP used by the Jordan-twist
construction.  No full-k YBE, RLL, RTT, or many-body Jordan amplification is
inferred from this certificate.
