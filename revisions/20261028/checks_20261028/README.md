# Calculation checks accompanying the 28 October revision

These records extend the original stationary-drive problem; the final folder
is an explicitly separate symmetry control. `README_calculation_history.md`
preserves the chronology. Its earlier statements about no pole search or no
manuscript edits describe those earlier stages, not this assembled revision.

| Folder | What it establishes | What it does not establish |
| --- | --- | --- |
| `solver/`, `kr_comparison/` | Matched finite-chain ED/NRG-pilot/KR static observables | Converged interacting continuum poles |
| `bath_convergence/` | Noninteracting bath-resolution diagnostics | Completion of the finite-U convergence problem |
| `two_sided_ep/` | Pointwise occupied finite-embedding EPs; sided splitting and loops for selected sizes | Stable continuum EP coordinates |
| `optimized_bath/` | Positive-residue bath fits and exact finite-bath responses; stable charge can coexist with changing matrix self-energy | A converged local-moment continuum EP |
| `pt_interacting_control/` | Exact scalar-bath dimer double-pole residue and 14 finite-system checks | The driven SOC bath's strong-coupling EP |

Read each folder's report for parameters, units, basis sizes, script commands,
and failed checks. Wilson `z` shifts are discretization parameters, not complex
frequency. Bath blocks contain two fermion modes; optimized-bath sizes instead
count individual fermion orbitals. Real-axis causality checks are sampled,
and continuation of the finite rational self-energy is an embedding assumption.

The exact PT comparison uses a biorthogonally continued singlet reference;
it is not a passive driven steady-state simulation. Original data files are
retained. The sole portability edit removes an obsolete absolute-path fallback
from `two_sided_ep.py`; it now requires the bundled sibling `solver/` directory.

Per-folder checksums refer to the files included in this candidate; omitted
terminal logs are not listed. The original calculation archive is retained
separately. The repository-level SHA256SUMS.txt covers the complete candidate.
