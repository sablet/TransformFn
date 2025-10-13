"""Example/Check メタデータのレジストリ。

注釈から抽出した Example/Check 情報をリポジトリ横断で共有するための
最小限の登録/参照 API を提供する。xform-auditor や proj-dtypes から
利用されることを想定している。

将来の拡張（例: ExampleList, ExampleFactory, CheckPair など）を見据えて、
登録レコードには "source" フィールドを持たせており、バリエーションを
柔軟に追加できる設計としている。
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence, Tuple

from .metadata import is_example_metadata


@dataclass(slots=True, frozen=True)
class ExampleEntry:
    """単一パラメータに紐づく Example メタデータ。"""

    transform: str
    parameter: str
    metadata: Tuple[Any, ...]
    source: str = "annotation"


@dataclass(slots=True, frozen=True)
class CheckEntry:
    """Transform と Check 関数の紐づけ情報。"""

    transform: str
    target: str
    func: Callable[..., Any]
    source: str = "annotation"


class ExampleRegistry:
    """Example メタデータを保持するレジストリ。"""

    def __init__(self) -> None:
        self._by_transform: Dict[str, Dict[str, ExampleEntry]] = {}
        self._lock = RLock()

    def register_many(
        self,
        transform: str,
        mapping: Mapping[str, Sequence[Any]],
        *,
        source: str = "annotation",
        overwrite: bool = False,
    ) -> Tuple[ExampleEntry, ...]:
        if not transform:
            raise ValueError("transform FQN must be a non-empty string")
        if not mapping:
            raise ValueError(
                "at least one parameter must be provided for example registration"
            )

        entries = []
        for parameter, metadata_seq in mapping.items():
            entry = self._build_entry(transform, parameter, metadata_seq, source)
            entries.append(entry)

        with self._lock:
            existing = self._by_transform.setdefault(transform, {})
            for entry in entries:
                prior = existing.get(entry.parameter)
                if not overwrite and prior is not None and prior != entry:
                    param_key = f"{transform}.{entry.parameter}"
                    raise ValueError(
                        f"examples for {param_key} already registered; "
                        "set overwrite=True to replace"
                    )
                existing[entry.parameter] = entry
        return tuple(entries)

    def get(self, transform: str) -> Tuple[ExampleEntry, ...]:
        with self._lock:
            items = self._by_transform.get(transform, {})
            return tuple(items.values())

    def get_for_parameter(self, transform: str, parameter: str) -> ExampleEntry | None:
        with self._lock:
            return self._by_transform.get(transform, {}).get(parameter)

    def iter_all(self) -> Tuple[ExampleEntry, ...]:
        with self._lock:
            results: list[ExampleEntry] = []
            for param_map in self._by_transform.values():
                results.extend(param_map.values())
            return tuple(results)

    def clear(self) -> None:
        with self._lock:
            self._by_transform.clear()

    @staticmethod
    def _build_entry(
        transform: str,
        parameter: str,
        metadata_seq: Sequence[Any],
        source: str,
    ) -> ExampleEntry:
        if not parameter:
            raise ValueError("parameter name must be a non-empty string")
        metadata = tuple(metadata_seq)
        if not metadata:
            target = f"{transform}.{parameter}"
            raise ValueError(
                f"{target} must provide at least one ExampleType/ExampleValue"
            )
        _validate_example_metadata(transform, parameter, metadata)
        return ExampleEntry(
            transform=transform, parameter=parameter, metadata=metadata, source=source
        )


class CheckRegistry:
    """Check 関数の解決と Transform との紐づけを担うレジストリ。"""

    def __init__(self) -> None:
        self._by_transform: Dict[str, Tuple[CheckEntry, ...]] = {}
        self._by_target: Dict[str, Callable[..., Any]] = {}
        self._lock = RLock()

    def register(
        self,
        transform: str,
        entries: Sequence[Tuple[str, Callable[..., Any]]],
        *,
        source: str = "annotation",
        overwrite: bool = False,
    ) -> Tuple[CheckEntry, ...]:
        if not transform:
            raise ValueError("transform FQN must be a non-empty string")
        if not entries:
            raise ValueError("at least one check target must be provided")

        records = [
            self._build_entry(transform, target, func, source)
            for target, func in entries
        ]

        with self._lock:
            if transform in self._by_transform and not overwrite:
                previous = self._by_transform[transform]
                if tuple(previous) != tuple(records):
                    raise ValueError(
                        f"checks for {transform} already registered; "
                        "set overwrite=True to replace"
                    )
            self._by_transform[transform] = tuple(records)

            for record in records:
                existing = self._by_target.get(record.target)
                if (
                    existing is not None
                    and existing is not record.func
                    and not overwrite
                ):
                    message = (
                        "check target {target} already registered; set overwrite=True"
                    )
                    message = message.format(target=record.target)
                    raise ValueError(message)
                self._by_target[record.target] = record.func
        return tuple(records)

    def resolve(self, target: str) -> Callable[..., Any]:
        with self._lock:
            try:
                return self._by_target[target]
            except KeyError as exc:
                raise KeyError(f"check target not registered: {target}") from exc

    def get(self, transform: str) -> Tuple[CheckEntry, ...]:
        with self._lock:
            return self._by_transform.get(transform, tuple())

    def clear(self) -> None:
        with self._lock:
            self._by_transform.clear()
            self._by_target.clear()

    @staticmethod
    def _build_entry(
        transform: str,
        target: str,
        func: Callable[..., Any],
        source: str,
    ) -> CheckEntry:
        if not target:
            raise ValueError(f"check target must be non-empty for {transform}")
        if not callable(func):
            raise TypeError(f"check target {target} for {transform} must be callable")
        return CheckEntry(transform=transform, target=target, func=func, source=source)


def _validate_example_metadata(
    transform: str, parameter: str, metadata: Iterable[Any]
) -> None:
    for meta in metadata:
        if not is_example_metadata(meta):
            raise TypeError(
                f"{transform}.{parameter} contains unsupported metadata: {meta!r}; "
                "expected ExampleType or ExampleValue"
            )


example_registry = ExampleRegistry()
check_registry = CheckRegistry()

__all__ = [
    "ExampleEntry",
    "CheckEntry",
    "ExampleRegistry",
    "CheckRegistry",
    "example_registry",
    "check_registry",
]
