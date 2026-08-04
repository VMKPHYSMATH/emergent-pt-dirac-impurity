# Resolvent cancellation gate

Status: **PASS__DIVERGENT_PROJECTORS_CANCEL_IN_COMPLETE_RESOLVENTS**.

The normalized individual projector grows to
`57.735749` at
`beta0=0.4999`, but the complete
one-particle pole sum agrees with the direct resolvent to
`1.113e-14`.  The two particle-hole virtual charge sectors
agree with the closed finite-`U` resolvent to `4.441e-16`.
The complete charge factor remains finite and tends to one at the core EP.

This gate checks cancellation of the singular eigenprojector pieces.  It does
not insert a Petermann factor into a Green function, density of states, or RG
equation.
