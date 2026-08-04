# DM-NRG response construction and interpretation boundary

For the final-shell biorthogonal ground-state projector, the reduced density
matrix is traced backward through the stored NRG isometries.  At shell `n`, the
retarded response uses

\[
\chi^R_{AB}(\omega)
 = \sum_{ab}^{\text{not }KK}
 \frac{A_{ab}[B\rho_n-\rho_n B]_{ba}}
 {\omega-(E_b-E_a)+i\eta}.
\]

The `not KK` restriction is the complete-basis shell partition: at every
nonterminal shell, transitions with both states kept are deferred to later
iterations.  At the terminal shell, all remaining states are declared
discarded.

The physical impurity operators

\[
S_x=\tfrac12(d_1^\dagger d_2+d_2^\dagger d_1),\quad
S_y=-\tfrac{i}{2}(d_1^\dagger d_2-d_2^\dagger d_1),\quad
S_z=\tfrac12(n_1-n_2)
\]

are transformed directly as `L' * S_a * R` and propagated independently.

To control output size, transitions are ranked only after the full shell
commutator weights have been evaluated.  Selection continues until the
configured fraction of total absolute weight is retained or the per-shell cap
is reached.  `dmnrg_response_quality.csv` records the retained fraction for
every shell and response-matrix element.

The analyzer uses generic peak finding on the largest singular value of
`-Im chi / pi`.  It does not impose two peaks.  All raw matrix elements remain
available for alternative analysis.

Complex-spectrum curves are algebraic exploratory responses.  Only rows with
`equilibrium_gate=true` may be discussed as equilibrium zero-temperature
susceptibilities.
