# Biorthogonal Wilson-operator propagation fix

## Scope

This patch corrects the PT-Dirac adapter's Wilson-chain recursion.  It does not
change the impurity Hamiltonian, Wilson coefficients, SOC-overlap control, pole
selection, or quartic/Jordan audit.

## Correct rule

For a biorthonormal left/right basis, `L'R = I`, every operator must be
transformed as

```text
O_RL = L' O R.
```

Therefore annihilation and creation operators must be propagated separately:

```text
c_RL       = L' c  R,
c_dagger_RL = L' c† R.
```

In general,

```text
c_dagger_RL != adjoint(c_RL)
```

because the basis transformation is nonunitary.

## Code changes

- `NRGState` now stores both `last_ann` and `last_cre`.
- `coupling_term` receives both matrices and never reconstructs creation as
  `adjoint(last_ann)`.
- `initialize_nrg` and `add_site` independently transform the new Wilson-site
  annihilation and creation operators.
- `test/biorth_operator_regression.jl` checks the canonical anticommutation
  relations without truncation and compares the recursive hopping term with an
  explicit full-Fock-space basis transformation.

## Upstream provenance

This was an adapter-specific simplification error.  The public
`PhillipBC/NonHermitianNRG` Anderson implementation keeps separate creation and
annihilation matrices (`UM` and `UMd`) and propagates both.  Andrew Mitchell's
original code is Hermitian, where the iterative transformation is unitary and
ordinary adjunction is valid.

## Scientific status after the fix

The corrected recursion is required before interpreting iterations `n >= 1`.
Existing iteration-zero pole and residue files are unchanged.  All later NRG
flows and kept-space spectra must be regenerated.

The SOC-overlap boundary model remains only a control model; this patch does
not turn it into the microscopic chiral bath `epsilon_{k,±}=k^2±lambda k`.
