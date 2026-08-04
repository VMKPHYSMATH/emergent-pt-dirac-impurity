# Driven-Dirac impurity causal channel-resolved RG gate

Status: **CAUSAL_MATRIX_RG_PASS__TRANSIENT_AMPLIFICATION_BUT_NO_ROBUST_KONDO_SCALE_ENHANCEMENT**.

The finite-`U` Schrieffer--Wolff transformation gives `J0=0.24000000`. The
channel density is generated directly from

`rho(omega)=i[G^R(omega)-G^A(omega)]/(2*pi)`

and the matrix flow `dJ/dl=J[rho(+Lambda)+rho(-Lambda)]J` is integrated
without an inserted Petermann or `Zbio` factor.  No commutativity is assumed
by the solver.  In this particle-hole-symmetric benchmark the
shell-symmetrized density matrices share a fixed eigenbasis; their maximum
commutator is `1.712e-14`.  This permits the independent exact
inverse-flow check

`J^(-1)(l)=J0^(-1)-integral_0^l rho_shell(l') dl'`,

which agrees with the nonlinear Riccati integration to
`6.761e-13` in the threshold scales tested.

The calculation is causal and passive.  A dense scan over
`0 <= beta0 <= 0.60` and `-4 <= omega <= 4` gives minimum density eigenvalue
`-2.220e-16` (roundoff
at the passive boundary), while the strict pre-EP scan over
`0.35 <= beta0 <= 0.4999` and `-0.5 <= omega <= 0.5` gives the positive
minimum `2.410e-02`.
Analytically, positivity at every real frequency follows from the
resolvent-sandwich identity whenever the complete bath-rate matrix is
positive semidefinite.  The diagonal DOS is real and nonnegative.  The
resolvent/sandwich identity closes to
`2.674e-15`, the
two-state spectral sum rule closes to `0.000e+00`, and the
maximum RG step-refinement drift is `6.480e-10`.  An adverse
relative-only kernel with an indefinite rate matrix is correctly rejected
and develops a negative density eigenvalue.

The result is not a simple enhancement. At `beta0=0.50`, the active/control
operational-scale ratio is `1.056654` at the weak threshold `g*=0.30`,
but `0.916556` at `g*=1`. The flows cross: biorthogonal structure
amplifies the early weak-coupling eigenchannel, then suppresses the later
approach to strong coupling relative to the matched `Gamma_PT=0` control.
The operational crossover occurs near `g*=0.4627`.

Thus this gate supplies direct evidence for **transient channel-resolved RG
amplification**, not for a threshold-independent enhanced Kondo scale.

Files:

- `causal_channel_density.csv`: PSD spectral eigenchannels from `G^R/G^A`.
- `causal_density_sum_rules.csv`: full-axis two-state spectral sum rules.
- `matrix_rg_threshold_scales.csv`: all thresholds and solver refinements.
- `active_control_scale_comparison.csv`: active/control scale ratios.
- `representative_rg_flow.csv`: near-EP and control flow curves.
- `rg_step_refinement.csv`: ODE convergence.
- `inverse_flow_crosscheck.csv`: exact inverse-flow/Riccati comparison.
- `Channel_Resolved_RG_Gate.pdf`: four-panel review figure.
- `channel_resolved_rg_summary.json`: machine-readable decision.
- `SHA256SUMS`: hashes for every output, the executable, and its local model
  helper.

No manuscript or public repository file was modified.
