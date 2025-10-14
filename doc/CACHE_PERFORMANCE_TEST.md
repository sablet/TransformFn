# キャッシュ機能のパフォーマンステスト実装タスク

## 調査日
2025-10-14

## 調査結果サマリ

### 現状
**キャッシュのパフォーマンステストは未実装。**
また、キャッシュキーの生成は実装済みだが、**キャッシュヒット時の読み込み・再利用ロジックが未実装**。

### 実装状況

| コンポーネント | 状態 | ファイル |
|--------------|------|---------|
| キャッシュキー生成 | ✅ 実装済み | `xform_core/cache.py` |
| アーティファクト保存 | ✅ 実装済み | `xform_core/artifact.py` |
| キャッシュヒット判定 | ❌ 未実装 | `xform_core/runner.py` |
| キャッシュからの読み込み | ❌ 未実装 | `xform_core/artifact.py` |
| パフォーマンステスト | ❌ 未実装 | `packages/xform-core/tests/` |

### 既存のキャッシュ関連テスト

#### `apps/pipeline-app/tests/test_runner.py:36-44`
```python
def test_cache_key_changes_when_parameters_differ() -> None:
    """パラメータ変更時にキャッシュキーが変わることを検証"""
    transform = dag.PIPELINE.get("price_bars").transform
    params_a = {"spec": HLOCVSpec(n=16, seed=1)}
    params_b = {"spec": HLOCVSpec(n=16, seed=2)}

    key_a = compute_cache_key(transform, inputs={}, params=params_a)
    key_b = compute_cache_key(transform, inputs={}, params=params_b)

    assert key_a != key_b
```

**このテストの目的**: キャッシュキー生成ロジックの正しさを検証
**このテストの限界**: キャッシュヒット時のパフォーマンス向上は検証していない

---

## 問題点の詳細

### 1. キャッシュ再利用ロジックの欠如

**現在の `PipelineRunner.run()` 実装 (`xform_core/runner.py:29-72`)**:
```python
def run(self, pipeline: Pipeline) -> PipelineRunResult:
    for node in pipeline.topological_order():
        kwargs = node.build_kwargs(outputs)
        # ... 省略 ...

        cache_key = compute_cache_key(
            node.transform,
            inputs=input_payload,
            params=param_payload,
        )

        # ⚠️ キャッシュヒット判定なし！常に実行される
        result = node.func(**kwargs)
        outputs[node.name] = result

        record = self.store.save(node, result, cache_key)
        artifact_records.append(record)
```

**期待される動作**:
```python
# キャッシュヒット判定を追加
cached_result = self.store.load(cache_key)
if cached_result is not None:
    result = cached_result  # キャッシュから取得
else:
    result = node.func(**kwargs)  # 実行して保存
    self.store.save(node, result, cache_key)
```

### 2. `ArtifactStore` に読み込みメソッドがない

**現在の実装 (`xform_core/artifact.py:53-153`)**:
- `save(node, value, cache_key)`: ✅ 実装済み
- `load(cache_key)`: ❌ 未実装

---

## 実装タスク

### タスク1: `ArtifactStore.load()` メソッドの追加

**目的**: キャッシュキーから保存済みアーティファクトを読み込む

**実装箇所**: `packages/xform-core/xform_core/artifact.py`

**実装内容**:
```python
def load(self, cache_key: str) -> object | None:
    """Load artifact from cache by cache key.

    Args:
        cache_key: Cache key to look up

    Returns:
        Cached value if found, None otherwise
    """
    # キャッシュキーの最初の12文字でファイル検索
    key_prefix = cache_key[:12]
    candidates = list(self.directory.glob(f"*-{key_prefix}.*"))

    if not candidates:
        return None

    # 最初の候補を読み込み（複数ある場合は最新）
    artifact_path = max(candidates, key=lambda p: p.stat().st_mtime)

    # 拡張子に応じて読み込み
    if artifact_path.suffix == ".csv":
        import pandas as pd
        return pd.read_csv(artifact_path)
    elif artifact_path.suffix == ".json":
        with artifact_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    elif artifact_path.suffix == ".txt":
        with artifact_path.open("r", encoding="utf-8") as f:
            return f.read()

    return None
```

**テストケース** (`packages/xform-core/tests/test_artifact_store.py`):
- キャッシュミス時に `None` を返す
- キャッシュヒット時に正しい値を返す
- JSON/CSV/TXT 各形式で正しく復元できる

---

### タスク2: `PipelineRunner.run()` にキャッシュヒット判定を追加

**目的**: キャッシュがあれば関数実行をスキップ

**実装箇所**: `packages/xform-core/xform_core/runner.py`

**実装内容**:
```python
def run(self, pipeline: Pipeline) -> PipelineRunResult:
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

        # キャッシュヒット判定を追加
        cached_result = self.store.load(cache_key)
        if cached_result is not None:
            result = cached_result
            # 既存のレコードを参照（または新規作成）
            record = self._find_or_create_record(node, cache_key)
        else:
            result = node.func(**kwargs)
            record = self.store.save(node, result, cache_key)

        outputs[node.name] = result
        artifact_records.append(record)

    return PipelineRunResult(outputs=outputs, records=tuple(artifact_records))
```

**テストケース** (`packages/xform-core/tests/test_pipeline_runner.py`):
- キャッシュミス時に関数が実行される
- キャッシュヒット時に関数が実行されない（スキップされる）
- 同一パラメータで2回実行した場合、結果が一致する

---

### タスク3: パフォーマンステストの追加

**目的**: キャッシュヒット時に劇的な高速化が確認できることを検証

**実装箇所**: `packages/xform-core/tests/test_cache_performance.py` (新規作成)

**テスト設計**:

#### 3.1 基本パターン: `time.sleep()` による重い処理のシミュレーション

```python
import time
from typing import Annotated

from xform_core import (
    ArtifactStore,
    Check,
    ExampleValue,
    Node,
    Pipeline,
    PipelineRunner,
    transform,
)


@transform
def slow_computation(
    X: Annotated[int, ExampleValue(42)],
    delay: float = 1.0
) -> Annotated[int, Check("tests.checks.check_positive")]:
    """Simulates heavy computation with intentional delay."""
    time.sleep(delay)
    return X * 2


def test_cache_hit_improves_performance(tmp_path):
    """キャッシュヒット時に90%以上高速化されることを検証"""
    # パイプライン作成
    node = Node(
        name="slow_node",
        func=slow_computation,
        transform=slow_computation.__transform_fn__,
        inputs=(),
        parameters=(("delay", 1.0),),
    )
    pipeline = Pipeline(nodes=(node,))

    store = ArtifactStore(directory=tmp_path)
    runner = PipelineRunner(store=store)

    # 初回実行 (キャッシュミス)
    start = time.time()
    result1 = runner.run(pipeline)
    first_duration = time.time() - start

    # 2回目実行 (キャッシュヒット)
    start = time.time()
    result2 = runner.run(pipeline)
    second_duration = time.time() - start

    # 結果が一致することを確認
    assert result1.outputs["slow_node"] == result2.outputs["slow_node"]

    # 2回目が90%以上高速化されていることを確認
    assert second_duration < first_duration * 0.1, (
        f"Cache hit should be 10x faster: "
        f"first={first_duration:.3f}s, second={second_duration:.3f}s"
    )

    # デバッグ情報
    print(f"First run: {first_duration:.3f}s")
    print(f"Second run: {second_duration:.3f}s")
    print(f"Speedup: {first_duration / second_duration:.1f}x")
```

#### 3.2 応用パターン: 複数ノードでのキャッシュ効果

```python
def test_partial_cache_hit_in_pipeline(tmp_path):
"""一部のノードだけキャッシュヒットした場合の動作を検証"""

    from dataclasses import replace

    @transform
    def step_a(X: Annotated[int, ExampleValue(1)]) -> Annotated[int, Check("...")]:
        time.sleep(0.5)
        return X + 1

    @transform
    def step_b(
        X: Annotated[int, ExampleValue(2)],
        factor: int = 2,
    ) -> Annotated[int, Check("...")]:
        time.sleep(0.5)
        return X * factor

    pipeline = Pipeline(
        nodes=(
            Node(
                name="a",
                func=step_a,
                transform=step_a.__transform_fn__,
                inputs=(),
            ),
            Node(
                name="b",
                func=step_b,
                transform=step_b.__transform_fn__,
                inputs=(("X", "a"),),
                parameters=(("factor", 2),),
            ),
        ),
    )

    store = ArtifactStore(directory=tmp_path)
    runner = PipelineRunner(store=store)

    # 初回実行 (両方キャッシュミス)
    start = time.time()
    result1 = runner.run(pipeline)
    first_duration = time.time() - start

    # 2回目実行 (両方キャッシュヒット)
    start = time.time()
    result2 = runner.run(pipeline)
    second_duration = time.time() - start

    assert second_duration < first_duration * 0.1

    # 3回目: step_b のパラメータを変更 (step_a はキャッシュヒット、step_b はミス)
    updated_step_b = replace(
        pipeline.nodes[1],
        parameters=(("factor", 3),),
    )
    pipeline = Pipeline(nodes=(pipeline.nodes[0], updated_step_b))
    start = time.time()
    result3 = runner.run(pipeline)
    third_duration = time.time() - start

    # step_a がキャッシュから取得されるため、約0.5秒で完了
    assert 0.4 < third_duration < 0.7, (
        f"Expected ~0.5s (only step_b executed), got {third_duration:.3f}s"
    )
```

#### 3.3 エッジケース: キャッシュ無効化のテスト

```python
def test_cache_invalidation_on_code_change(tmp_path):
"""関数コードが変わるとキャッシュが無効化されることを検証"""

    from dataclasses import replace

    # 初回の関数定義
    @transform
    def compute_v1(X: Annotated[int, ExampleValue(10)]) -> Annotated[int, Check("...")]:
        time.sleep(0.5)
        return X * 2

    node = Node(
        name="compute",
        func=compute_v1,
        transform=compute_v1.__transform_fn__,
        inputs=(),
    )
    pipeline = Pipeline(nodes=(node,))

    store = ArtifactStore(directory=tmp_path)
    runner = PipelineRunner(store=store)

    # 初回実行
    result1 = runner.run(pipeline)
    assert result1.outputs["compute"] == 20

    # 関数の実装を変更
    @transform
    def compute_v2(X: Annotated[int, ExampleValue(10)]) -> Annotated[int, Check("...")]:
        time.sleep(0.5)
        return X * 3  # ロジック変更

    # ノードを更新
    updated_node = replace(
        pipeline.nodes[0],
        func=compute_v2,
        transform=compute_v2.__transform_fn__,
    )
    pipeline = Pipeline(nodes=(updated_node,))

    # 再実行 (code_hash が変わるのでキャッシュミス)
    result2 = runner.run(pipeline)
    assert result2.outputs["compute"] == 30  # 新しいロジックが適用される
```

---

## 実装優先度

| タスク | 優先度 | 理由 |
|--------|--------|------|
| タスク1: `ArtifactStore.load()` | **HIGH** | キャッシュ再利用の基盤 |
| タスク2: `PipelineRunner` へのキャッシュヒット判定 | **HIGH** | タスク1に依存、パフォーマンス向上の本体 |
| タスク3.1: 基本パフォーマンステスト | **HIGH** | キャッシュ効果の検証 |
| タスク3.2: 複数ノードテスト | **MEDIUM** | 実用的なシナリオの検証 |
| タスク3.3: キャッシュ無効化テスト | **MEDIUM** | 再現性担保の検証 |

---

## 実装時の注意事項

### キャッシュキーのファイル名マッピング

**現在の保存形式** (`artifact.py:131`):
```python
stem = f"{node_name}-{cache_key[:12]}"
```

**問題点**:
- 同一ノード名で異なるキャッシュキーの場合、最初の12文字が衝突する可能性がある
- ファイル検索時に誤ったアーティファクトを取得するリスク

**対策**:
1. **完全なキャッシュキーでファイル名を生成** (推奨):
   ```python
   stem = f"{node_name}-{cache_key}"
   ```
2. **メタデータファイルを作成**:
   ```python
   # {node_name}-{cache_key[:12]}.meta.json
   {
       "full_cache_key": "abcdef1234567890...",
       "created_at": "2025-10-14T12:00:00Z",
       "node_name": "price_bars"
   }
   ```

### 並行実行とキャッシュの競合

**現在の実装では考慮不要**:
- `PipelineRunner` はシングルスレッド実行
- 将来的に並行実行を導入する場合、ファイルロックや AtomicWrite が必要

---

## 関連ドキュメント

- `doc/ARCHITECTURE.md`: キャッシュの設計思想
- `doc/TICKET-ARCHITECTURE-IMPLEMENTATION.md`: 実装タスク一覧
- `packages/xform-core/xform_core/cache.py`: キャッシュキー生成実装
- `packages/xform-core/xform_core/runner.py`: パイプライン実行ロジック
- `packages/xform-core/xform_core/artifact.py`: アーティファクト保存実装

---

## 今後の拡張性

### キャッシュストレージのバックエンド抽象化

現在はファイルシステムのみだが、将来的に以下を考慮:
- Redis/Memcached (インメモリキャッシュ)
- S3/GCS (クラウドストレージ)
- DuckDB/SQLite (クエリ可能なアーティファクトストア)

**インタフェース設計例**:
```python
class CacheBackend(Protocol):
    def get(self, key: str) -> object | None: ...
    def set(self, key: str, value: object) -> None: ...
    def exists(self, key: str) -> bool: ...
```

---

## まとめ

- ✅ キャッシュキー生成は実装済み
- ❌ キャッシュヒット判定・読み込みが未実装
- ❌ パフォーマンステストが未実装

**次のステップ**: タスク1 → タスク2 → タスク3 の順に実装
