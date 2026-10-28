# Full-k two-cut nonlinear pole EP certificate

## Certified witness

- `lambda = 0.05`, `W = 0.10`,
  `K = pi/4`.
- `epsilon_O = 120`, `Omega = 60`.
- Auxiliary vertex `V = 1.213422009614`.
- Pole `z_EP = 0.011972834346 -0.002807216274 i`.
- Real impurity gate `epsilon = 0.015170349783`.

## Algebraic gates

- `|F| = 1.444e-21`.
- `|F_z| = 1.088e-15`.
- `F_zz = 1.686448433812 -0.190336728319 i`.
- Kernel singular values: `1.207e-03` and
  `1.222e-18`.
- Local-discriminant winding: `1.000000000000`.
- Endpoint pole permutation: `[1, 0]`.
- Pole half-winding: `0.500000000000`.
- Maximum tracking-step/half-separation ratio:
  `0.103110`.

## Causal and microscopic gates

- Analytic integral versus adaptive quadrature:
  `7.933e-19`.
- Two-cut sheet-continuity residual:
  `4.024e-09`.
- Physical rate eigenvalues:
  `0.00190368` and
  `0.00309632`.
- Passivity margin `Gamma_min/Gamma_bar =
  0.761472`.
- Threshold gap in pole widths:
  `1.351248`.
- Exact SOC matrix-shift identity residual:
  `3.483e-16`.
- Off-shell ratio `V/(epsilon_O-Omega) = 0.02022`.
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
