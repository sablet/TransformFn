from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_DIR = REPO_ROOT / "apps" / "pipeline-app"
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from xform_auditor.auditor import AuditStatus, audit  # noqa: E402


def test_faulty_transform_triggers_violation() -> None:
    report = audit(["pipeline_app.faulty_transforms"])
    results = {result.transform: result.status for result in report.results}
    target = "pipeline_app.faulty_transforms.produce_invalid_feature_map"

    assert results[target] is AuditStatus.VIOLATION
