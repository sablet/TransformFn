"""監査結果の整形ロジック。"""

from __future__ import annotations

import json
from typing import Dict

from .auditor import AuditReport, AuditResult


def render_text(report: AuditReport) -> str:
    lines = []
    results = report.results
    if not results:
        lines.append("No @transform functions discovered.")
    else:
        for result in results:
            lines.append(_format_result_line(result))

    summary = report.summary
    lines.append(
        (
            "Summary: total={total} ok={ok} violation={violation} "
            "error={error} missing={missing}"
        ).format(
            total=summary.total,
            ok=summary.ok,
            violation=summary.violation,
            error=summary.error,
            missing=summary.missing,
        )
    )
    return "\n".join(lines)


def render_json(report: AuditReport) -> str:
    payload: Dict[str, object] = {
        "summary": {
            "total": report.summary.total,
            "ok": report.summary.ok,
            "violation": report.summary.violation,
            "error": report.summary.error,
            "missing": report.summary.missing,
        },
        "results": [
            {
                "transform": result.transform,
                "status": result.status.value,
                "message": result.message,
                "detail": result.detail,
            }
            for result in report.results
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _format_result_line(result: AuditResult) -> str:
    base = f"[{result.status.value}] {result.transform}"
    if result.message:
        base = f"{base} - {result.message}"
    return base
