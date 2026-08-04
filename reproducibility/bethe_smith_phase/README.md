# Full-scattering Bethe–Smith phase gate

This gate validates the retained one-particle Bethe phase quantization using the complete two-channel Fisher–Lee scattering matrix. It does not assume that the chiral scattering matrix is diagonal and does not claim a multiparticle interacting Bethe solution.

The exported checks cover scattering unitarity, unimodularity of both scattering eigenvalues, off-diagonal components in both the original spin and fixed Hadamard/chiral representations, the finite-difference determinant-phase/Wigner–Smith identity, and the finite-window phase count. Matrix components are basis dependent; the eigenphases and Smith trace are the invariant checks. Run `python bethe_smith_phase_gate.py` from the directory containing the script.
