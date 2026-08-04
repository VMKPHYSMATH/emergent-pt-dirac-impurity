.PHONY: check paper clean

check:
	python reproducibility/finite_u_contact/verify_finiteU_contact_algebra.py
	python reproducibility/analytic_checks/contact_and_smith_identity_check.py
	python reproducibility/bethe_smith_phase/bethe_smith_phase_gate.py
	cd reproducibility/nrg_jordan_residue/adapter && python benchmarks/prb_complete/tests/validate_biorth_fix.py
	cd reproducibility/nrg_jordan_residue/adapter && python benchmarks/prb_complete/tests/validate_install.py
	cd reproducibility/nrg_jordan_residue/adapter && python benchmarks/prb_complete/tests/validate_jordan_comprehensive.py

paper:
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error supplemental_material.tex
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error supplemental_material.tex

clean:
	cd paper && latexmk -c main.tex supplemental_material.tex
