.PHONY: check format lint typecheck complexity duplication test trace trace_examples

CHECK_DIRS := src app *.py

check: duplication format lint typecheck complexity

format:
	uv run ruff format .

lint:
	uv run ruff check --fix --unsafe-fixes $(CHECK_DIRS)

typecheck:
	uv run pyright $(CHECK_DIRS)

# Uses xenon thresholds from pyproject.toml
complexity:
	uv run xenon $(CHECK_DIRS)

duplication:
	jscpd --config .jscpd.json $(CHECK_DIRS)

test:
	uv run pytest

# trace:
# 	uv run python run_trace.py $(MODULE)

# trace_examples:
# 	uv run python run_trace.py components.examples.transformers
