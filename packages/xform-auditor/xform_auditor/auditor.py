"""xform-auditor のコアロジック。"""

from __future__ import annotations

import ast
import inspect
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Iterable, Sequence, Set, Tuple

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


_UNUSED_PARAMS_DETAIL = (
    "This likely indicates incomplete implementation. All parameters should be used."
)


def _evaluate_transform(handle: TransformHandle) -> AuditResult:
    transform_fqn = handle.fqn

    missing_result = _maybe_missing_transform(handle, transform_fqn)
    if missing_result is not None:
        return missing_result

    func = handle.func
    assert func is not None  # mypy narrowing

    unused_result = _maybe_unused_parameters(func, transform_fqn)
    if unused_result is not None:
        return unused_result

    call_args_or_error = _prepare_call_args(handle)
    if isinstance(call_args_or_error, AuditResult):
        return call_args_or_error
    call_args = call_args_or_error

    output, execution_error = _execute_transform_safely(handle, call_args)
    if execution_error is not None:
        return execution_error
    assert output is not None

    check_error = _evaluate_checks(handle, output)
    if check_error is not None:
        return check_error

    return _audit_ok(transform_fqn)


def _maybe_missing_transform(
    handle: TransformHandle, transform_fqn: str
) -> AuditResult | None:
    if handle.transform is None:
        error = handle.error
        message = str(error) if error is not None else "failed to normalize transform"
        detail = None
        if error is not None and not isinstance(error, ValueError):
            detail = repr(error)
        return _audit_error(transform_fqn, message, detail=detail)

    if handle.func is None:
        return _audit_error(
            transform_fqn,
            "callable reference missing despite successful normalization",
        )
    return None


def _maybe_unused_parameters(
    func: Callable[..., object], transform_fqn: str
) -> AuditResult | None:
    unused_params = _check_unused_parameters(func)
    if not unused_params:
        return None

    params_str = ", ".join(sorted(unused_params))
    message = f"Parameters defined but not used in function body: {params_str}"
    return _audit_error(transform_fqn, message, detail=_UNUSED_PARAMS_DETAIL)


def _prepare_call_args(handle: TransformHandle) -> CallArgs | AuditResult:
    try:
        return _build_call_args(handle)
    except ExampleMaterializationError as exc:
        return _audit_error(
            handle.fqn,
            str(exc),
            status=AuditStatus.MISSING,
        )
    except Exception as exc:  # pragma: no cover - 想定外のエラー
        detail = traceback.format_exc()
        message = f"failed to prepare inputs: {exc}"
        return _audit_error(handle.fqn, message, detail=detail)


def _execute_transform_safely(
    handle: TransformHandle, call_args: CallArgs
) -> tuple[object | None, AuditResult | None]:
    func = handle.func
    assert func is not None
    try:
        output = func(*call_args.args, **call_args.kwargs)
    except Exception as exc:
        detail = traceback.format_exc()
        message = f"execution raised {exc.__class__.__name__}: {exc}"
        return None, _audit_error(handle.fqn, message, detail=detail)
    return output, None


def _evaluate_checks(handle: TransformHandle, output: object) -> AuditResult | None:
    try:
        _run_checks(handle, output)
    except CheckViolationError as exc:
        message = f"{exc.target}: {exc}"
        return _audit_error(handle.fqn, message, status=AuditStatus.VIOLATION)
    except CheckExecutionError as exc:
        message = f"{exc.target}: {exc}"
        return _audit_error(handle.fqn, message, detail=exc.detail)
    except Exception as exc:  # pragma: no cover - 想定外の例外
        detail = traceback.format_exc()
        message = f"unexpected check error: {exc}"
        return _audit_error(handle.fqn, message, detail=detail)
    return None


def _audit_ok(transform_fqn: str) -> AuditResult:
    return AuditResult(
        transform=transform_fqn,
        status=AuditStatus.OK,
        message=None,
        detail=None,
    )


def _audit_error(
    transform_fqn: str,
    message: str,
    *,
    status: AuditStatus = AuditStatus.ERROR,
    detail: str | None = None,
) -> AuditResult:
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


def _check_unused_parameters(func: Callable[..., object]) -> Set[str]:
    """Detect parameters that are defined but unused within the function."""
    func_def = _resolve_function_def(func)
    if func_def is None:
        return set()

    params = _collect_parameter_names(func_def)
    used = _collect_used_parameters(func_def, params)
    return params - used


def _resolve_function_def(func: Callable[..., object]) -> ast.FunctionDef | None:
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        return None

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func.__name__:
            return node
    return None


def _collect_parameter_names(func_def: ast.FunctionDef) -> Set[str]:
    params: Set[str] = set()
    for arg in (
        *func_def.args.posonlyargs,
        *func_def.args.args,
        *func_def.args.kwonlyargs,
    ):
        if arg.arg not in ("self", "cls"):
            params.add(arg.arg)
    if func_def.args.vararg:
        params.add(func_def.args.vararg.arg)
    if func_def.args.kwarg:
        params.add(func_def.args.kwarg.arg)

    return params


def _collect_used_parameters(func_def: ast.FunctionDef, params: Set[str]) -> Set[str]:
    used_in_body: Set[str] = set()

    body_start = 0
    if (
        func_def.body
        and isinstance(func_def.body[0], ast.Expr)
        and isinstance(func_def.body[0].value, ast.Constant)
        and isinstance(func_def.body[0].value.value, str)
    ):
        body_start = 1

    for stmt in func_def.body[body_start:]:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name) and node.id in params:
                used_in_body.add(node.id)

    return used_in_body


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
