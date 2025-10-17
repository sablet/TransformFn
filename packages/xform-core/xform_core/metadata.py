"""@transform で利用するメタ型の定義とユーティリティ。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

T = TypeVar("T")


@dataclass(slots=True)
class ExampleType(Generic[T]):
    """型情報のみを提供する Example メタデータ。"""

    type: type[T]
    description: Optional[str] = None


@dataclass(slots=True)
class ExampleValue(Generic[T]):
    """実際の値（または生成仕様）を提供する Example メタデータ。"""

    value: T
    description: Optional[str] = None


@dataclass(slots=True)
class Check:
    """出力検査用の関数参照を保持する。"""

    target: str

    def __post_init__(self) -> None:
        if not isinstance(self.target, str) or not self.target:
            raise ValueError("Check target must be a non-empty string literal.")

    @classmethod
    def __class_getitem__(cls, item: str) -> "Check":
        """型注釈での subscript 記法をサポート: Check["fqn"] → Check(target="fqn")"""
        if not isinstance(item, str):
            raise TypeError(f"Check subscript must be a string, got {type(item)}")
        return cls(target=item)


def is_example_metadata(meta: object) -> bool:
    return isinstance(meta, (ExampleType, ExampleValue))


def is_check_metadata(meta: object) -> bool:
    return isinstance(meta, Check)


def describe_example(meta: object) -> str:
    """ロギング用のシンプルな説明を返す。"""

    if isinstance(meta, ExampleType):
        return f"ExampleType({meta.type!r})"
    if isinstance(meta, ExampleValue):
        return f"ExampleValue({meta.value!r})"
    return repr(meta)
