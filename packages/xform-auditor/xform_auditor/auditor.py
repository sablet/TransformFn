"""xform-auditor のコアロジック。"""

from __future__ import annotations

import inspect
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Sequence, Tuple

from xform_core import check_registry, example_registry
from xform_core.transforms_core import allow_transform_errors

from .discover import TransformHandle, discover_transforms
from .examples import ExampleMaterializationError, materialize_entry


class AuditStatus(Enum):
    OK = "OK"
    VIOLATION = "VIOLATION"
    ERROR = "ERROR"
    MISSING = "MISSING"


@dataclass(frozen=True)
class AuditResult:
    transform: str
    status: AuditStatus
    message: str | None = None
    detail: str | None = None


@dataclass(frozen=True)
class AuditSummary:
    total: int
    ok: int
    violation: int
    error: int
    missing: int

    @property
    def exit_code(self) -> int:
        return 1 if (self.violation > 0 or self.error > 0) else 0


@dataclass(frozen=True)
class AuditReport:
    results: Tuple[AuditResult, ...]
    summary: AuditSummary


@dataclass(frozen=True)
class CallArgs:
    args: Tuple[object, ...]
    kwargs: Dict[str, object]


class CheckViolationError(RuntimeError):
    def __init__(self, target: str, message: str) -> None:
        super().__init__(message)
        self.target = target


class CheckExecutionError(RuntimeError):
    def __init__(self, target: str, message: str, detail: str) -> None:
        super().__init__(message)
        self.target = target
        self.detail = detail


def audit(targets: Sequence[str]) -> AuditReport:
    with allow_transform_errors():
        handles = discover_transforms(targets)
    results = tuple(_evaluate_transform(handle) for handle in handles)
    summary = _build_summary(results)
    return AuditReport(results=results, summary=summary)


def _evaluate_transform(handle: TransformHandle) -> AuditResult:
    transform_fqn = handle.fqn

    if handle.transform is None:
        error = handle.error
        err_message = (
            str(error) if error is not None else "failed to normalize transform"
        )
        err_detail = None
        if error is not None and not isinstance(error, ValueError):
            err_detail = repr(error)
        return AuditResult(
            transform=transform_fqn,
            status=AuditStatus.ERROR,
            message=err_message,
            detail=err_detail,
        )

    func = handle.func
    if func is None:  # 非常時: 正常化は成功したが Function 参照が欠落
        return AuditResult(
            transform=transform_fqn,
            status=AuditStatus.ERROR,
            message="callable reference missing despite successful normalization",
        )

    try:
        call_args = _build_call_args(handle)
    except ExampleMaterializationError as exc:
        return AuditResult(
            transform=transform_fqn,
            status=AuditStatus.MISSING,
            message=str(exc),
        )
    except Exception as exc:  # pragma: no cover - 想定外のエラー
        traceback_text = traceback.format_exc()
        return AuditResult(
            transform=transform_fqn,
            status=AuditStatus.ERROR,
            message=f"failed to prepare inputs: {exc}",
            detail=traceback_text,
        )

    try:
        output = func(*call_args.args, **call_args.kwargs)
    except Exception as exc:
        traceback_text = traceback.format_exc()
        return AuditResult(
            transform=transform_fqn,
            status=AuditStatus.ERROR,
            message=f"execution raised {exc.__class__.__name__}: {exc}",
            detail=traceback_text,
        )

    status = AuditStatus.OK
    message: str | None = None
    detail: str | None = None

    try:
        _run_checks(handle, output)
    except CheckViolationError as exc:
        status = AuditStatus.VIOLATION
        message = f"{exc.target}: {exc}"
    except CheckExecutionError as exc:
        status = AuditStatus.ERROR
        message = f"{exc.target}: {exc}"
        detail = exc.detail
    except Exception as exc:  # pragma: no cover - 想定外の例外
        status = AuditStatus.ERROR
        message = f"unexpected check error: {exc}"
        detail = traceback.format_exc()

    return AuditResult(
        transform=transform_fqn,
        status=status,
        message=message,
        detail=detail,
    )


def _build_call_args(handle: TransformHandle) -> CallArgs:
    transform_fqn = handle.fqn
    entries = {entry.parameter: entry for entry in example_registry.get(transform_fqn)}

    assert handle.func is not None  # type narrowing for mypy
    signature = inspect.signature(handle.func)
    args: list[object] = []
    kwargs: Dict[str, object] = {}

    for param in signature.parameters.values():
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            continue
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            continue

        entry = entries.get(param.name)
        if entry is None:
            if param.default is inspect._empty:
                raise ExampleMaterializationError(
                    f"{transform_fqn}.{param.name}: missing Example metadata",
                    parameter=param.name,
                )
            continue

        try:
            value = materialize_entry(entry)
        except ExampleMaterializationError as exc:
            message = str(exc)
            raise ExampleMaterializationError(
                f"{transform_fqn}.{param.name}: {message}",
                parameter=param.name,
            ) from exc

        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            args.append(value)
        else:
            kwargs[param.name] = value

    return CallArgs(args=tuple(args), kwargs=kwargs)


def _run_checks(handle: TransformHandle, output: object) -> None:
    transform = handle.transform
    assert transform is not None  # type narrowing for mypy
    for target in transform.output_checks:
        check_func = check_registry.resolve(target)
        try:
            check_func(output)
        except (AssertionError, ValueError) as exc:
            raise CheckViolationError(target, str(exc)) from exc
        except Exception as exc:
            detail = traceback.format_exc()
            raise CheckExecutionError(
                target,
                f"{exc.__class__.__name__}: {exc}",
                detail,
            ) from exc


def _build_summary(results: Iterable[AuditResult]) -> AuditSummary:
    total = ok = violation = error = missing = 0
    for result in results:
        total += 1
        if result.status is AuditStatus.OK:
            ok += 1
        elif result.status is AuditStatus.VIOLATION:
            violation += 1
        elif result.status is AuditStatus.ERROR:
            error += 1
        elif result.status is AuditStatus.MISSING:
            missing += 1
    return AuditSummary(
        total=total,
        ok=ok,
        violation=violation,
        error=error,
        missing=missing,
    )


# allow_transform_errors is re-exported from xform_core and used directly
