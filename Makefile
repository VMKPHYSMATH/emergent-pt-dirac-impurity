.PHONY: check paper clean

check:
	python reproducibility/finite_u_contact/verify_finiteU_contact_algebra.py
	python reproducibility/analytic_checks/contact_and_smith_identity_check.py
	python reproducibility/bethe_smith_phase/bethe_smith_phase_gate.py
	cd reproducibility/nrg_jordan_residue/adapter && python benchmarks/prb_complete/tests/validate_biorth_fix.py
	cd reproducibility/nrg_jordan_residue/adapter && python benchmarks/prb_complete/tests/validate_install.py
	cd reproducibility/nrg_jordan_residue/adapter && python benchmarks/prb_complete/tests/validate_jordan_comprehensive.py
	cd reproducibility/nrg_jordan_residue/adapter && python benchmarks/prb_complete/tests/validate_fullk_n_scaling_install.py
	python reproducibility/full_k_finite_u/full_k_two_cut_ep_certificate.py --soc 0.05 --out /tmp/full_k_two_cut_ep
	python reproducibility/full_k_finite_u/full_k_ep_soc_family.py --out /tmp/full_k_ep_soc_family
	python reproducibility/full_k_finite_u/finiteU_fullk_reproducibility/validate_archived_finiteU_gates.py
	python reproducibility/full_k_finite_u/validate_fullk_n_scaling_gate.py

paper:
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error supplemental_material.tex
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error supplemental_material.tex

clean:
	cd paper && latexmk -c main.tex supplemental_material.tex
