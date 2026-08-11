# Verification and v2.1 release update

Verify the candidate branch before merging or tagging it.

## 1. Verify the snapshot

```bash
python -m pip install -r requirements.txt
python reproducibility/finite_u_contact/verify_finiteU_contact_algebra.py
python reproducibility/analytic_checks/contact_and_smith_identity_check.py
python reproducibility/full_k_finite_u/finiteU_fullk_reproducibility/validate_archived_finiteU_gates.py
python reproducibility/full_k_finite_u/validate_fullk_n_scaling_gate.py
make paper
```

Review `git diff --check` and the generated PDFs after copying into a clone.

## 2. Review the candidate branch

```bash
git add -A
git status --short
git diff --cached --check
git diff --cached --stat
```

Open a pull request into `main`. Keep it as a draft until the repository,
manuscript PDFs, arXiv version, and archival metadata have been compared.

## 3. Release only after verification

After merging, create tag `v2.1.0`, build the release archive from that exact
commit, deposit that archive, and then add the returned version-specific DOI
without replacing the stable concept DOI `10.5281/zenodo.21434682`.

Do not restore the removed rapidity/phase-map or universal quartic-floor
claims during conflict resolution.
