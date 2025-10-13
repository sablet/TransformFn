"""Minimal pipeline runner that demonstrates cache key generation."""

from __future__ import annotations

import hashlib
import json
import pickle
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Mapping, Sequence, cast

try:  # pandas はオプショナル依存だが、デモでは利用する
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - 環境によっては pandas が無い場合
    pd = None

if TYPE_CHECKING:  # pragma: no cover - 型チェック専用
    import pandas as pd_typing

from xform_core import TransformFn

from .dag import Node, Pipeline


@dataclass(frozen=True)
class ArtifactRecord:
    """Metadata about a persisted artifact produced by a pipeline node."""

    node: str
    cache_key: str
    artifact_path: Path
    transform: TransformFn


@dataclass(frozen=True)
class PipelineRunResult:
    """Aggregated outputs and artifacts from a pipeline execution."""

    outputs: Dict[str, object]
    records: Sequence[ArtifactRecord]


class ArtifactStore:
    """Very small artifact store that writes results under ``output/``."""

    def __init__(self, directory: Path | None = None) -> None:
        if directory is None:
            directory = Path(__file__).resolve().parents[3] / "output" / "pipeline-app"
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.records: list[ArtifactRecord] = []

    def save(self, node: Node, value: object, cache_key: str) -> ArtifactRecord:
        artifact_path = self._write_value(node.name, cache_key, value)
        record = ArtifactRecord(
            node=node.name,
            cache_key=cache_key,
            artifact_path=artifact_path,
            transform=node.transform,
        )
        self.records.append(record)
        return record

    def _write_value(self, node_name: str, cache_key: str, value: object) -> Path:
        stem = f"{node_name}-{cache_key[:12]}"
        base_path = self.directory / stem

        if _is_dataframe(value):
            artifact_path = base_path.with_suffix(".csv")
            dataframe = cast("pd_typing.DataFrame", value)
            dataframe.to_csv(artifact_path, index=False)
            return artifact_path

        try:
            artifact_path = base_path.with_suffix(".json")
            with artifact_path.open("w", encoding="utf-8") as handle:
                json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
            return artifact_path
        except TypeError:
            artifact_path = base_path.with_suffix(".txt")
            with artifact_path.open("w", encoding="utf-8") as handle:
                handle.write(repr(value))
            return artifact_path


class PipelineRunner:
    """Execute a :class:`~pipeline_app.dag.Pipeline` from its DAG definition."""

    def __init__(self, store: ArtifactStore | None = None) -> None:
        self.store = store or ArtifactStore()

    def run(self, pipeline: Pipeline) -> PipelineRunResult:
        outputs: Dict[str, object] = {}
        artifact_records: list[ArtifactRecord] = []

        for node in pipeline.topological_order():
            kwargs = node.build_kwargs(outputs)
            dependency_params = {param for param, _ in node.inputs}
            input_payload = {param: kwargs[param] for param in dependency_params}
            param_payload = {
                key: value
                for key, value in kwargs.items()
                if key not in dependency_params
            }

            cache_key = compute_cache_key(
                node.transform,
                inputs=input_payload,
                params=param_payload,
            )

            result = node.func(**kwargs)
            outputs[node.name] = result

            record = self.store.save(node, result, cache_key)
            artifact_records.append(record)

        return PipelineRunResult(outputs=outputs, records=tuple(artifact_records))


def compute_cache_key(
    transform: TransformFn,
    *,
    inputs: Mapping[str, object],
    params: Mapping[str, object],
) -> str:
    """Build a reproducible cache key from runtime inputs and metadata."""

    payload = {
        "transform_id": str(transform.id),
        "transform_version": transform.version,
        "code_hash": transform.code_ref.code_hash,
        "inputs": {key: _hash_value(value) for key, value in sorted(inputs.items())},
        "params": {key: _hash_value(value) for key, value in sorted(params.items())},
        "env_hash": platform.python_version(),
    }
    data = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _hash_value(value: object) -> str:
    try:
        blob = pickle.dumps(value)
    except Exception:
        blob = repr(value).encode("utf-8", errors="replace")
    return hashlib.sha256(blob).hexdigest()


def _is_dataframe(value: object) -> bool:
    if pd is None:
        return False
    return isinstance(value, pd.DataFrame)


__all__ = [
    "ArtifactRecord",
    "ArtifactStore",
    "PipelineRunResult",
    "PipelineRunner",
    "compute_cache_key",
]
