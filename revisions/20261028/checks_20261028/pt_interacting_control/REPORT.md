# Exact interacting EP in a spin-symmetric Anderson control

28 October 2026. New calculation; the manuscript has not been edited.

## Result and scope

The proposed restricted route works. A half-filled Anderson dimer with a local
PT field has an impurity Green-function double pole with a strictly nonzero
coefficient for every finite repulsive U and nonzero hopping V. This is an
analytic finite-system result, independently checked by untruncated charge-sector
resolvents. Finite controls with three and five bath sites also pass the checks
listed below. No Jordan or PT form is fitted to the self-energy or Green function.

The model is H_A + d† N d, where H_A is a spin-rotation-invariant Anderson
Hamiltonian and N=Delta sigma_x+i gamma sigma_z. At gamma=Delta>0, N²=0.
The antiunitary symmetry is internal spin exchange followed by complex
conjugation; it is not an assertion about the physical spin time-reversal
operator of a generic SOC host. The bath in this control is spin independent.

This is the imposed non-Hermitian impurity-core control already discussed in
the Supplement, not the original passive full-k driven continuum model.
The state is the biorthogonal analytic continuation of an isolated Hermitian
singlet. Its scalar nd and D agree with those of that Hermitian reference.
No claim about a physical pumped steady-state occupation is made.

## What the proof adds

Complex-spin covariance gives G(z,N)=g0(z) I+g1(z) N at the null field, but
alignment alone does not guarantee a double pole. For an isolated spin-doublet
addition transition with weight Z and local-field slope m, differentiation of
the Hermitian Zeeman response gives the double-pole coefficient B=Z m N.
Consequently Z*m != 0 closes the missing amplitude condition. No arbitrary
nonzero-residue theorem for all baths or all continuum resonances is claimed.

This also supplies the nonlinear inverse-kernel certificate: at the pole,
K=G^-1 tends to -(m/Z)N, with rank one; det(K) has a double zero with
second derivative 2/Z² != 0. A double-pole amplitude and a coincident pair
of scalar eigenvalues are not being substituted for this kernel condition.

For the half-filled dimer eps_d=-U/2, bath energy zero, and hopping V>0, set
A=sqrt(U²/16+V²), C=sqrt(U²/16+4V²). The lower and upper positive transition
energies are w_l=C-A and w_h=C+A. The exact formulas are

    Z_l=(w_l²-9V²)/[2(w_l²-w_h²)]
    m_l=(1+U/(4A))/2
    D=(1-U/sqrt(U²+64V²))/4
    B_l=Z_l*m_l*N

For every finite U>=0, 0<w_l<=V and w_h>w_l, so Z_l>0 and m_l>=1/2.
Thus the selected pole is genuinely defective at gamma=Delta. The symmetry
pins this control condition, while U changes the pole energy and its residue.
At fixed V and large U, w_l~6V²/U and Z_l*m_l~18V²/U²: the amplitude is not
bounded away from zero in the infinite-U limit.

## Exact numerical results

All entries use hopping V=0.5, Delta=gamma=0.1 and eps_d=-U/2; the chemical
potential is zero. Bath sites are spinful sites, not single fermion modes.
The tabulated w is the lowest positive addition pole, and B=(Z*m)N.

| Bath sites | U | nd | D | w | Z*m |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 1.000000 | 0.25000000 | 0.50000000 | 0.25000000 |
| 1 | 0.5 | 1.000000 | 0.21899132 | 0.49239402 | 0.30016059 |
| 1 | 2 | 1.000000 | 0.13819660 | 0.41092721 | 0.28086767 |
| 1 | 4 | 1.000000 | 0.07322330 | 0.29617957 | 0.16191959 |
| 1 | 8 | 1.000000 | 0.02639320 | 0.17451516 | 0.05928735 |
| 1 | 16 | 1.000000 | 0.00746437 | 0.09197675 | 0.01679289 |
| 3 | 0 | 1.000000 | 0.25000000 | 0.30901699 | 0.13090170 |
| 3 | 0.5 | 1.000000 | 0.21323297 | 0.30377399 | 0.16790889 |
| 3 | 2 | 1.000000 | 0.12183237 | 0.24713264 | 0.17984741 |
| 3 | 4 | 1.000000 | 0.05766369 | 0.16983695 | 0.10268709 |
| 3 | 8 | 1.000000 | 0.01863860 | 0.09464473 | 0.03460202 |
| 3 | 16 | 1.000000 | 0.00494421 | 0.04800584 | 0.00913318 |
| 5 | 4 | 1.000000 | 0.05242211 | 0.12289020 | 0.07763771 |
| 5 | 8 | 1.000000 | 0.01602654 | 0.06659547 | 0.02548446 |

This is finite-system evidence of suppressed charge fluctuations and a local
moment amplitude. It does not establish continuum Kondo screening or an
unscreened local-moment fixed point. The change of w and residue with bath size
is not a continuum convergence certificate.

## Verification

- Exact finite charge sectors; no NRG truncation, mean field or tail fitting.
- Every selected addition doublet has algebraic multiplicity 2; nullities of
  H-E and (H-E)² are 1 and 2 respectively.
- Direct contour integration extracts a nonzero double-pole coefficient, not
  merely its normalized direction. Halving the contour radius and doubling
  its mesh reproduces the coefficient; the third-order coefficient vanishes
  within numerical precision.
- Analytic dimer pole, weight, field slope, double occupancy and scalar Green
  function agree with the independent Fock-space calculation.
- Independent direct and Schur resolvents agree. At U=0 the many-body response
  agrees with the one-body inverse. The equal-time anticommutator sum rule holds.
- At U=4 and 8, four total sites, the squared splitting approaches
  4*m²*(Delta²-gamma²) from both sides of the EP.
- Two control loops (radii 1e-4 and 5e-5, meshes 128 and 256) at each U yield
  squared-splitting winding -1 and pole exchange. The second loop control is
  a real impurity sigma_z field, intentionally leaving the PT line.
- A real bath sigma_x field of 0.005 preserves the internal PT symmetry but
  splits the poles at the old EP condition. This shows the additional role of
  spin covariance; PT alone is insufficient to pin the EP.

Machine-readable maximum residuals are in results/summary.json.

## Self-energy interpretation

At the null field the interacting self-energy has the form Sigma0(z) I+
Sigma1(z) N. Sigma1 is not generally zero. From G=g0 I+g1 N one finds
Sigma1=g1/g0²-1. At the selected pole its limiting value is m/Z-1.
For the dimer the scalar self-energy is U/2+(U²/4)z/(z²-9V²).
Therefore this calculation includes a dynamical interaction self-energy and
a nontrivial local spin vertex. Freezing the self-energy to a scalar function
would give B=Z² N instead of the correct Z*m*N.

## Manuscript use

This could strengthen the existing structural-control subsection from
"nilpotent direction retained" to "nonzero double pole established in a
specified finite interacting model." It should be presented as a controlled
comparison with the full driven problem, not as evidence that the original
occupied full-k EP has converged. It does not justify replacing the original
spin-dependent bath by a symmetric fit and calling that a solution.

Interaction-stable symmetry-protected EPs already exist in the literature:
R. Schaefer, J. C. Budich and D. J. Luitz, Physical Review Research 4, 033181
(2022), https://doi.org/10.1103/PhysRevResearch.4.033181. Novelty of this precise
impurity residue formulation needs a dedicated comparison before claiming it.

## Reproduction

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python pt_control.py
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python pt_control.py --sites 6 --U 4 8 --output results/six_sites.json
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python verify_control.py
    python build_report.py
    pdflatex -interaction=nonstopmode -halt-on-error PT_Interacting_Control_Note.tex

Dependencies: Python 3, NumPy, SciPy, Matplotlib; LaTeX for the optional note.
All energies use a common arbitrary unit, with hopping set to 0.5. The five-bath-site checks use
924 states in the half-filled sector and 792 states in each response sector.
