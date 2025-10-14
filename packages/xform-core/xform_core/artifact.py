"""Artifact storage for pipeline execution results."""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence, cast

try:  # pandas はオプショナル依存
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover
    pd = None

if TYPE_CHECKING:
    import pandas as pd_typing

    from .models import TransformFn
    from .pipeline import Node


@dataclass(frozen=True)
class ArtifactRecord:
    """Metadata about a persisted artifact produced by a pipeline node.

    Attributes:
        node: Node name that produced this artifact
        cache_key: Cache key for reproducibility tracking
        artifact_path: File system path where artifact is stored
        transform: TransformFn metadata for the producing function
    """

    node: str
    cache_key: str
    artifact_path: Path
    transform: TransformFn


@dataclass(frozen=True)
class PipelineRunResult:
    """Aggregated outputs and artifacts from a pipeline execution.

    Attributes:
        outputs: Dictionary mapping node names to their output values
        records: Sequence of artifact records for persisted outputs
    """

    outputs: dict[str, object]
    records: Sequence[ArtifactRecord]


class ArtifactStore:
    """Artifact store that writes pipeline results to disk.

    Automatically resolves output directory based on caller location,
    placing artifacts under ``output/{app_name}/`` by default.

    Attributes:
        directory: Root directory for artifact storage
        records: List of saved artifact records
    """

    def __init__(self, directory: Path | None = None) -> None:
        """Initialize artifact store.

        Args:
            directory: Custom output directory. If None, auto-detects from caller
        """
        if directory is None:
            directory = self._resolve_default_directory()
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.records: list[ArtifactRecord] = []

    def _resolve_default_directory(self) -> Path:
        """Auto-detect output directory based on calling module location.

        Searches for ``apps/{app_name}/`` in caller stack and creates
        corresponding ``output/{app_name}/`` directory.

        Returns:
            Path to output directory
        """
        for frame_info in inspect.stack():
            frame_path = Path(frame_info.filename).resolve()
            # Look for apps/{app_name}/ pattern
            if "apps" in frame_path.parts:
                apps_idx = frame_path.parts.index("apps")
                if apps_idx + 1 < len(frame_path.parts):
                    app_name = frame_path.parts[apps_idx + 1]
                    # Find project root (where apps/ directory exists)
                    for parent in frame_path.parents:
                        if (parent / "apps").exists():
                            return parent / "output" / app_name
        # Fallback to current working directory
        return Path.cwd() / "output" / "default"

    def save(self, node: Node, value: object, cache_key: str) -> ArtifactRecord:
        """Save a node's output to disk and record metadata.

        Args:
            node: Pipeline node that produced the value
            value: Output value to persist
            cache_key: Cache key for reproducibility

        Returns:
            ArtifactRecord describing the saved artifact
        """
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
        """Write value to disk with appropriate serialization format.

        Args:
            node_name: Name of the producing node
            cache_key: Cache key (first 12 chars used in filename)
            value: Value to serialize

        Returns:
            Path to written file
        """
        stem = f"{node_name}-{cache_key[:12]}"
        base_path = self.directory / stem

        # DataFrame → CSV
        if _is_dataframe(value):
            artifact_path = base_path.with_suffix(".csv")
            dataframe = cast("pd_typing.DataFrame", value)
            dataframe.to_csv(artifact_path, index=False)
            return artifact_path

        # JSON-serializable → JSON
        try:
            artifact_path = base_path.with_suffix(".json")
            with artifact_path.open("w", encoding="utf-8") as handle:
                json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
            return artifact_path
        except TypeError:
            # Fallback → repr as text
            artifact_path = base_path.with_suffix(".txt")
            with artifact_path.open("w", encoding="utf-8") as handle:
                handle.write(repr(value))
            return artifact_path


def _is_dataframe(value: object) -> bool:
    """Check if value is a pandas DataFrame.

    Args:
        value: Object to check

    Returns:
        True if value is a DataFrame, False otherwise
    """
    if pd is None:
        return False
    return isinstance(value, pd.DataFrame)


__all__ = [
    "ArtifactRecord",
    "ArtifactStore",
    "PipelineRunResult",
]
