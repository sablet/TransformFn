.PHONY: check setup format lint typecheck complexity duplication audit test

UV ?= uv

CHECK_DIRS := packages apps

check: duplication format lint typecheck complexity

setup:
	$(UV) sync --all-extras

format:
	$(UV) run ruff format $(CHECK_DIRS)

lint:
	$(UV) run ruff check --fix --unsafe-fixes $(CHECK_DIRS)

typecheck:
	$(UV) run mypy $(CHECK_DIRS)

complexity:
	$(UV) run xenon $(CHECK_DIRS)

duplication:
	jscpd --config .jscpd.json $(CHECK_DIRS)

test:
	$(UV) run pytest

audit:
	$(UV) run python -m xform_auditor \
		apps/algo-trade
