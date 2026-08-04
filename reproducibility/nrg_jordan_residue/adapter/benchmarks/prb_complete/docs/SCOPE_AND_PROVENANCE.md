# Scope and provenance

- `patches/PTDiracNHNRG_benchmark.jl` is derived from the cluster-aware
  `PTDiracNHNRG_clusterfix.jl` used in the current conversation.
- `vendor/self_consistent_saddle_gate.py` and
  `vendor/channel_resolved_rg_gate.py` are the supplied Driven-Dirac impurity revision gates.
- The benchmark layer is independent of, and has not been reviewed by, the
  authors of the upstream Burke/Mitchell NH-NRG repository.
- The package does not claim an exact driven Bethe/TBA solution, a complete
  FDM-NRG spectrum, or an NH-NRG realization of the globally twisted JMP
  many-body Jordan theorem.
