from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_DIR = REPO_ROOT / "apps" / "pipeline-app"
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from xform_auditor.auditor import AuditReport, AuditStatus, audit  # noqa: E402
from xform_auditor.report import render_text  # noqa: E402


def test_faulty_transform_triggers_violation() -> None:
    report = audit(["pipeline_app.faulty_transforms"])
    results = {result.transform: result for result in report.results}
    target = "pipeline_app.faulty_transforms.produce_invalid_feature_map"

    assert results[target].status is AuditStatus.VIOLATION

    expected_tr_errors = {
        "pipeline_app.faulty_transforms.tr001_missing_first_argument": "TR001",
        "pipeline_app.faulty_transforms.tr002_non_annotated_input": "TR002",
        "pipeline_app.faulty_transforms.tr003_missing_example_metadata": "TR003",
        "pipeline_app.faulty_transforms.tr004_incompatible_example_value": "TR004",
        "pipeline_app.faulty_transforms.tr005_non_annotated_return": "TR005",
        "pipeline_app.faulty_transforms.tr006_missing_check_metadata": "TR006",
        "pipeline_app.faulty_transforms.tr007_non_literal_check_target": "TR007",
        "pipeline_app.faulty_transforms.tr008_missing_check_target": "TR008",
        "pipeline_app.faulty_transforms.tr009_missing_docstring": "TR009",
    }

    for fqn, code in expected_tr_errors.items():
        result = results[fqn]
        assert result.status is AuditStatus.ERROR
        assert result.message is not None and code in result.message

    _assert_summary(report, violation=1, error=len(expected_tr_errors))

    text_output = render_text(report)
    for code in expected_tr_errors.values():
        assert code in text_output


def _assert_summary(report: AuditReport, *, violation: int, error: int) -> None:
    summary = report.summary
    assert summary.total == violation + error
    assert summary.ok == 0
    assert summary.missing == 0
    assert summary.violation == violation
    assert summary.error == error
