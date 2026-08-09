# Reproduit les figures et les nombres du manuscrit depuis les donnees
# deposees.  Aucune simulation n'est relancee : voir `make sweeps`.

PY ?= python3

.PHONY: figures numbers check sweeps clean

figures:            ## les six figures du manuscrit
	$(PY) src/make_schematic.py
	$(PY) src/make_figures_v3.py

numbers:            ## les nombres cites dans le texte
	$(PY) src/theory_rho1_invariant.py
	$(PY) src/test_selectivite_rho1.py
	$(PY) src/analyse_noise_families.py

check: figures numbers  ## tout

sweeps:             ## RELANCE les simulations (des heures de calcul)
	$(PY) src/run_phasecurve_v2.py
	$(PY) src/run_phasecurve_v2.py --L 30 --alphas 0.75,1,1.5,2 --n-eta 22 --n-seeds 12
	$(PY) src/run_noise_families.py

clean:
	rm -f figures/*.pdf
