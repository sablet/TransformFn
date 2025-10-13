from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_DIR = REPO_ROOT / "apps" / "pipeline-app"
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from xform_auditor.auditor import AuditReport, AuditStatus, audit  # noqa: E402
from xform_auditor.report import render_text  # noqa: E402
from xform_core import allow_transform_errors, normalize_transform  # noqa: E402


def test_faulty_transform_triggers_violation() -> None:
    with allow_transform_errors():
        from pipeline_app import faulty_transforms  # noqa: E402

    normalize_transform(faulty_transforms.produce_invalid_feature_map)

    report = audit(["pipeline_app.faulty_transforms"])
    results = {result.transform: result for result in report.results}
    target = "pipeline_app.faulty_transforms.produce_invalid_feature_map"

    assert results[target].status in {AuditStatus.VIOLATION, AuditStatus.MISSING}

    error_expectations = {
        "pipeline_app.faulty_transforms.tr001_missing_first_argument": "TR001",
        "pipeline_app.faulty_transforms.tr002_non_annotated_input": "TR002",
        "pipeline_app.faulty_transforms.tr003_missing_example_metadata": "TR003",
        "pipeline_app.faulty_transforms.tr004_incompatible_example_value": "TR004",
        "pipeline_app.faulty_transforms.tr005_non_annotated_return": "TR005",
        "pipeline_app.faulty_transforms.tr007_non_literal_check_target": "TR007",
        "pipeline_app.faulty_transforms.tr008_missing_check_target": "TR008",
        "pipeline_app.faulty_transforms.tr009_missing_docstring": "TR009",
    }

    violation_targets = {
        "pipeline_app.faulty_transforms.tr006_missing_check_metadata",
    }

    for fqn, code in error_expectations.items():
        result = results[fqn]
        assert result.status is AuditStatus.ERROR
        assert result.message is not None and code in result.message

    for fqn in violation_targets:
        result = results[fqn]
        assert result.status in {AuditStatus.VIOLATION, AuditStatus.MISSING}

    _assert_summary(
        report,
        expected_errors=len(error_expectations),
        expected_problematic=len(violation_targets) + 1,
    )

    text_output = render_text(report)
    for code in error_expectations.values():
        assert code in text_output


def _assert_summary(
    report: AuditReport, *, expected_errors: int, expected_problematic: int
) -> None:
    summary = report.summary
    assert summary.error == expected_errors
    assert summary.violation + summary.missing >= expected_problematic
    assert summary.total >= expected_errors + expected_problematic
    assert summary.ok == 0
