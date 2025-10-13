from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_DIR = REPO_ROOT / "apps" / "pipeline-app"
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from pipeline_app import dag, runner  # noqa: E402
from proj_dtypes.hlocv_spec import HLOCVSpec  # noqa: E402


def test_pipeline_topological_order() -> None:
    order = [node.name for node in dag.PIPELINE.topological_order()]
    assert order == ["price_bars", "feature_map", "top_features"]


def test_pipeline_runner_executes(tmp_path: Path) -> None:
    store = runner.ArtifactStore(directory=tmp_path)
    pipeline_runner = runner.PipelineRunner(store=store)
    result = pipeline_runner.run(dag.PIPELINE)

    assert set(result.outputs) == {"price_bars", "feature_map", "top_features"}
    expected_nodes = len(dag.PIPELINE.nodes)
    assert len(store.records) == expected_nodes
    assert all(record.artifact_path.exists() for record in store.records)

    selections = result.outputs["top_features"]
    assert isinstance(selections, list)
    assert selections


def test_cache_key_changes_when_parameters_differ() -> None:
    transform = dag.PIPELINE.get("price_bars").transform
    params_a = {"spec": HLOCVSpec(n=16, seed=1)}
    params_b = {"spec": HLOCVSpec(n=16, seed=2)}

    key_a = runner.compute_cache_key(transform, inputs={}, params=params_a)
    key_b = runner.compute_cache_key(transform, inputs={}, params=params_b)

    assert key_a != key_b
