.PHONY: bootstrap bootstrap-locked audit-check env-smoke test reproduce-paper reproduce-final

bootstrap:
	.venv/bin/python -m pip install -c constraints.txt -r requirements.txt
	.venv/bin/python -m pip install -e . --no-deps

bootstrap-locked:
	.venv/bin/python -m pip install -c constraints.txt -r requirements.lock.txt
	.venv/bin/python -m pip install -e . --no-deps

audit-check:
	.venv/bin/python scripts/validate_first_round.py

env-smoke:
	.venv/bin/python scripts/environment_smoke.py

test:
	.venv/bin/python -m pytest -q

reproduce-paper:
	.venv/bin/python scripts/generate_paper_figures.py --manifest-dir data/manifests --output-dir reports/figures_paper

reproduce-final:
	bash scripts/finalize_gate7.sh
