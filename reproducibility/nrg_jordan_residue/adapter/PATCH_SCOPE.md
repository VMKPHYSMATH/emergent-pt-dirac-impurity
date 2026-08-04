# Scope of the adaptation

The upstream `AIM_QNum_nonHerm.jl` is organized around charge and `S_z`
quantum-number blocks and scalar local Anderson parameters. The projected
Dirac impurity has a local channel-conversion term

\[
(\Delta_{\rm coh}+i\Gamma_{\rm PT})
(d_+^\dagger d_-+d_-^\dagger d_+),
\]

which preserves charge but breaks `S_z`/pseudospin conservation. A literal
small patch to the upstream initial Hamiltonian is therefore insufficient:
the block labels, operator propagation, and spectral operators must all be
changed consistently.

This package implements the required **charge-only** recursion separately,
while retaining the upstream methodological choices:

1. iterative Wilson-chain enlargement and truncation;
2. complex left/right eigensystems;
3. explicit biorthogonality and residual checks;
4. configurable complex-energy sorting;
5. aborting near exact defectiveness instead of normalizing a vanishing
   left-right overlap.

The next scientific extension is a non-Hermitian Anders-Schiller/FDM complete
basis. Until that is added, the level flow is the main NRG result and the
Lehmann output is a kept-space convergence diagnostic.
