from __future__ import annotations

import importlib
import json
import sys
import textwrap
from pathlib import Path

from xform_auditor import AuditStatus, audit
from xform_auditor.__main__ import cli, main

CLI_TOTAL_EXPECTED = 2
CLI_VIOLATION_EXPECTED = 1


def _write_module(module_dir: Path, name: str, content: str) -> str:
    path = module_dir / f"{name}.py"
    path.write_text(textwrap.dedent(content))
    if name in sys.modules:
        sys.modules.pop(name)
    importlib.invalidate_caches()
    return name


def test_audit_successful_transform(module_dir: Path) -> None:
    module_name = _write_module(
        module_dir,
        "sample_ok",
        """
        from typing import Annotated, TypedDict

        from xform_core import Check, ExampleValue, transform

        class Payload(TypedDict):
            value: int


        def check_positive(value: int) -> None:
            if value < 0:
                raise AssertionError("value must be non-negative")


        @transform
        def sample(
            data: Annotated[Payload, ExampleValue({"value": 3})]
        ) -> Annotated[int, Check("sample_ok.check_positive")]:
            'Return a positive number for auditing tests.'

            return data["value"] + 1
        """,
    )

    report = audit((module_name,))
    assert report.summary.total == 1
    assert report.summary.ok == 1
    assert report.summary.exit_code == 0

    result = report.results[0]
    assert result.transform.endswith("sample")
    assert result.status is AuditStatus.OK
    assert result.message is None


def test_audit_reports_violation(module_dir: Path) -> None:
    module_name = _write_module(
        module_dir,
        "sample_violation",
        """
        from typing import Annotated, TypedDict

        from xform_core import Check, ExampleValue, transform

        class Payload(TypedDict):
            value: int


        def ensure_even(value: int) -> None:
            if value % 2:
                raise ValueError("value must be even")


        @transform
        def sample(
            data: Annotated[Payload, ExampleValue({"value": 3})]
        ) -> Annotated[int, Check("sample_violation.ensure_even")]:
            'Return an odd number to trigger violation.'

            return data["value"]
        """,
    )

    report = audit((module_name,))
    assert report.summary.total == 1
    assert report.summary.violation == 1
    assert report.summary.exit_code == 1

    result = report.results[0]
    assert result.status is AuditStatus.VIOLATION
    assert "ensure_even" in (result.message or "")


def test_audit_missing_example(module_dir: Path) -> None:
    module_name = _write_module(
        module_dir,
        "sample_missing",
        """
        from dataclasses import dataclass
        from typing import Annotated

        from xform_core import Check, ExampleType, transform


        @dataclass
        class MissingInput:
            value: int


        def dummy_check(value: MissingInput) -> None:
            pass


        @transform
        def sample(
            data: Annotated[MissingInput, ExampleType(MissingInput)]
        ) -> Annotated[int, Check("sample_missing.dummy_check")]:
            'Return the underlying value to simulate missing example.'

            return data.value
        """,
    )

    report = audit((module_name,))
    assert report.summary.missing == 1
    assert report.summary.exit_code == 0

    result = report.results[0]
    assert result.status is AuditStatus.MISSING
    assert "MissingInput" in (result.message or "")


def test_cli_json_output(module_dir: Path) -> None:
    module_path = module_dir / "cli_target.py"
    module_path.write_text(
        textwrap.dedent(
            """
            from typing import Annotated, TypedDict

            from xform_core import Check, ExampleValue, transform

            class Payload(TypedDict):
                value: int


            def ensure_positive(value: int) -> None:
                if value <= 0:
                    raise AssertionError("value must be positive")


            def ensure_negative(value: int) -> None:
                if value >= 0:
                    raise ValueError("value must be negative")


            @transform
            def transform_ok(
                data: Annotated[Payload, ExampleValue({"value": 2})]
            ) -> Annotated[int, Check("cli_target.ensure_positive")]:
                'Produce positive number.'

                return data["value"]


            @transform
            def transform_violation(
                data: Annotated[Payload, ExampleValue({"value": 1})]
            ) -> Annotated[int, Check("cli_target.ensure_negative")]:
                'Produce violation.'

                return data["value"]
            """
        )
    )

    if "cli_target" in sys.modules:
        sys.modules.pop("cli_target")
    importlib.invalidate_caches()

    from click.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(cli, [str(module_path), "--format", "json"])
    assert result.exit_code == CLI_VIOLATION_EXPECTED

    payload = json.loads(result.output)
    assert payload["summary"]["total"] == CLI_TOTAL_EXPECTED
    assert payload["summary"]["violation"] == CLI_VIOLATION_EXPECTED

    # main() は exit code を返す
    code = main([str(module_path), "--format", "text"])
    assert code == CLI_VIOLATION_EXPECTED
