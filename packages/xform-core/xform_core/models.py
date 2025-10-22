"""データモデル定義。

TransformFn 正規化の結果として生成されるデータクラス群をまとめる。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple
from uuid import UUID, uuid4


@dataclass(slots=True)
class SchemaField:
    """単一フィールドのスキーマ情報。"""

    name: str
    dtype: str
    nullable: bool = False
    description: Optional[str] = None


@dataclass(slots=True)
class Schema:
    """入出力スキーマ。"""

    name: str
    fields: Tuple[SchemaField, ...] = field(default_factory=tuple)
    primary_key: Optional[str] = None


@dataclass(slots=True)
class ParamField:
    """パラメータの型情報。"""

    name: str
    dtype: str
    required: bool
    default: Optional[Any] = None


@dataclass(slots=True)
class ParamSchema:
    """関数パラメータのスキーマ。"""

    params: Tuple[ParamField, ...] = field(default_factory=tuple)


@dataclass(slots=True)
class CodeRef:
    """関数のコード位置を表す。"""

    module: str
    qualname: str
    filepath: Optional[str]
    lineno: Optional[int]
    code_hash: str

    @property
    def module_path(self) -> str:
        return f"{self.module}:{self.qualname}"


@dataclass(slots=True)
class TransformFn:
    """正規化済み TransformFn レコード。"""

    name: str
    qualname: str
    module: str
    input_schema: Schema
    output_schema: Schema
    param_schema: ParamSchema
    code_ref: CodeRef
    engine: str = "python"
    is_pure: bool = True
    parametric: bool = True
    input_metadata: Tuple[Any, ...] = field(default_factory=tuple)
    output_checks: Tuple[str, ...] = field(default_factory=tuple)
    id: UUID = field(default_factory=uuid4)
    version: str = field(init=False)

    def __post_init__(self) -> None:
        # code_hash の短縮値を版管理用に利用
        self.version = self.code_ref.code_hash[:12]
