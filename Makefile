.PHONY: bootstrap audit-check env-smoke

bootstrap:
	.venv/bin/python -m pip install -c constraints.txt -r requirements.txt

audit-check:
	.venv/bin/python scripts/validate_first_round.py

env-smoke:
	.venv/bin/python scripts/environment_smoke.py
