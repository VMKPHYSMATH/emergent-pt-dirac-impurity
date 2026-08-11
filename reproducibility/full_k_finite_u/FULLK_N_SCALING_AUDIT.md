# Full-k finite-chain common-contour audit

## Decision

The complete-basis matrix-Wilson-chain calculation passes a pairwise
common-contour Rouché gate between chain lengths (N=10) and (N=12).
It establishes finite-(N) persistence over this tested pair, not an
(N\to\infty) or thermodynamic-limit result.

## Fixed settings

- Interaction: (U=0.0025).
- Wilson discretization: \(\Lambda=1.5\).
- Logarithmic shifts: (N_z=4).
- State truncation: none (complete basis on each stated finite chain).
- Common ellipse: half-width (2\times10^{-4}) in the gate coordinate and
  (2\times10^{-3}) in the hybridization coordinate.
- The center is reoptimized independently at each chain length; the common
  contour is centered on the (N=12) solution.

## 256-point gate

| Diagnostic | (N=10) | (N=12) |
| --- | ---: | ---: |
| Local-discriminant winding | (-1) | (-1) |
| Pole-separation half-winding | (-1/2) | (-1/2) |
| Endpoint permutation | ([1,0]) | ([1,0]) |
| Minimum boundary modulus | (2.417\times10^{-9}) | (2.897\times10^{-9}) |
| Assignment/half-separation ratio | (0.1388) | (0.1352) |
| Maximum sum-rule error | (1.256\times10^{-15}) | (6.280\times10^{-15}) |

The pairwise sufficient inequality is

\[
\sup_C |D_{12}-D_{10}|=1.638\times10^{-9}
<\min_C|D_{10}|=2.417\times10^{-9},
\]

which gives the conservative Rouché ratio (0.6775<1).  Therefore the
enclosed zero count cannot change between these two finite chains on the
stated common contour.

## Resolution and center checks

At 128 points the same winding, pole exchange, and half-winding are obtained,
with Rouché ratio (0.6758).  Doubling the contour resolution reduces the
assignment ratios from (0.2782\to0.1388) for (N=10) and from
(0.2707\to0.1352) for (N=12).  The maximum independently reoptimized
center shift is (8.737\times10^{-5}), or (3.746\times10^{-3}) relative.

## Scope

No result is claimed for `N>12`, the omitted Wilson-chain tail,
`Lambda -> 1`, finite-temperature FDM-NRG, a strong-coupling Kondo fixed
point, a phase boundary, or universal interaction scaling.  A continuum-limit
claim would require an independently controlled Wilson-chain-tail analysis.
