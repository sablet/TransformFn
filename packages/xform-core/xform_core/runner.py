"""Pipeline execution runner with cache key tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .artifact import ArtifactStore, PipelineRunResult
    from .pipeline import Pipeline


class PipelineRunner:
    """Execute a Pipeline from its DAG definition with artifact storage.

    Attributes:
        store: ArtifactStore for persisting node outputs
    """

    def __init__(self, store: ArtifactStore | None = None) -> None:
        """Initialize pipeline runner.

        Args:
            store: ArtifactStore instance. If None, creates default store
        """
        from .artifact import ArtifactStore

        self.store = store or ArtifactStore()

    def run(self, pipeline: Pipeline) -> PipelineRunResult:
        """Execute pipeline nodes in topological order.

        For each node:
        1. Resolve dependencies from previous outputs
        2. Compute cache key from inputs and parameters
        3. Execute the transformation function
        4. Save output to artifact store

        Args:
            pipeline: Pipeline to execute

        Returns:
            PipelineRunResult containing all outputs and artifact records
        """
        from .artifact import PipelineRunResult
        from .cache import compute_cache_key

        outputs: dict[str, object] = {}
        artifact_records: list = []

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


__all__ = [
    "PipelineRunner",
]
