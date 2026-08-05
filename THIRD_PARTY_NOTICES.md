# Third-party software notices

## NonHermitianNRG

The adapter under
`reproducibility/nrg_jordan_residue/adapter/` was developed with reference to
the method and public interface of the open-source NonHermitianNRG project:

* Upstream repository: https://github.com/PhillipBC/NonHermitianNRG
* Upstream revision inspected for provenance:
  `b22832852328690da830598fc7161a1ab6664355`
* Authors: Phillip C. Burke and Andrew K. Mitchell
* Associated publication: “Non-Hermitian Numerical Renormalization Group:
  Solution of the Non-Hermitian Kondo Model,” *Physical Review Letters* **135**,
  206502 (2025), https://doi.org/10.1103/19td-1k9s
* Upstream license: MIT

The local PT-Dirac adapter is a separate charge-only implementation for the
pseudospin-mixing projected Anderson impurity. It is not an upstream release
and has not been reviewed, endorsed, or validated by the upstream authors.
The scope of the local implementation and the independent
creation/annihilation transition-operator correction are documented in:

* `reproducibility/nrg_jordan_residue/adapter/UPSTREAM_NOTICE.md`
* `reproducibility/nrg_jordan_residue/adapter/PATCH_SCOPE.md`
* `reproducibility/nrg_jordan_residue/adapter/BIORTH_OPERATOR_FIX.md`

The local adapter remains subject to the licensing terms stated in its own
`LICENSE` file.

## Upstream MIT license

MIT License

Copyright (c) 2025 Phillip C. Burke

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
