# Verification and later GitHub update

This snapshot intentionally contains no `.git` directory and has not been
pushed. Verify it first, then update the existing repository from a fresh
clone.

## 1. Verify the snapshot

```bash
python -m pip install -r requirements.txt
python reproducibility/finite_u_contact/verify_finiteU_contact_algebra.py
python reproducibility/analytic_checks/contact_and_smith_identity_check.py
make paper
```

Review `git diff --check` and the generated PDFs after copying into a clone.

## 2. Create a review branch in the existing repository

```bash
git clone https://github.com/VMKPHYSMATH/emergent-pt-dirac-impurity.git
cd emergent-pt-dirac-impurity
git switch -c release/v2.0.0-aps-final
```

Copy the verified snapshot contents into the clone while preserving the
clone's `.git` directory. Then run:

```bash
git add -A
git status --short
git diff --cached --check
git diff --cached --stat
git commit -m "Prepare v2.0.0 APS-final reproducibility release"
git push -u origin release/v2.0.0-aps-final
```

Open a pull request into `main`. Keep it as a draft until the repository,
manuscript PDFs, arXiv version, and archival metadata have been compared.

## 3. Release only after verification

After merging, create tag `v2.0.0`, build the release archive from that exact
commit, deposit that archive, and then add the returned version-specific DOI
without replacing the stable concept DOI `10.5281/zenodo.21434682`.

Do not restore the removed rapidity/phase-map or universal quartic-floor
claims during conflict resolution.
