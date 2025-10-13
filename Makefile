.PHONY: check setup format lint typecheck complexity duplication audit test

UV ?= uv

CHECK_DIRS := packages apps

check: duplication format lint typecheck complexity

setup:
	$(UV) sync --all-groups

format:
	$(UV) run ruff format $(CHECK_DIRS)

lint:
	$(UV) run ruff check --fix --unsafe-fixes $(CHECK_DIRS)

typecheck:
	$(UV) run mypy $(CHECK_DIRS)
	$(UV) run pyright $(CHECK_DIRS)

complexity:
	$(UV) run xenon $(CHECK_DIRS)

duplication:
	jscpd --config .jscpd.json $(CHECK_DIRS)

audit:
	@echo "Run '$(UV) run python -m xform_auditor apps/pipeline-app/pipeline_app' once the CLI is implemented."

test:
	$(UV) run pytest
