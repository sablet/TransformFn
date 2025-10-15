# TransformFn

TransformFn は、普通の Python 関数を型注釈とメタデータから正規化し、データ変換ノードとして扱う Python モノレポプラットフォームです。静的解析（mypy）、透過的キャッシュ、実行監査（CLI）により、最小限の記述で高品質なデータパイプラインを構築できます。

## 主要機能

- **型注釈ベースの正規化**: `@transform` デコレータと `Annotated` で関数を TransformFn レコードへ変換
- **静的検査**: mypy プラグイン（TR001-TR009）が型注釈・Example・Check・docstring を強制
- **自動テスト**: `xform-auditor` CLI が型注釈から入力を生成し、関数を実行・検証
- **透過キャッシュ**: 完全キャッシュキー（関数 + 入力 + パラメータ + コード + 環境）による再現性担保
- **レイヤード設計**: `xform-core`（共通基盤）→ `proj-dtypes`（型・例・検査）→ `apps`（Transform 関数・DAG）

## クイックスタート

### セットアップ

```bash
make setup        # uv sync --all-groups
```

### 品質チェック

```bash
make check        # duplication, format, lint, typecheck, complexity
make test         # pytest 実行
```

### Audit CLI

```bash
# 全 Transform を監査
uv run python -m xform_auditor apps/pipeline-app/pipeline_app

# JSON 出力
uv run python -m xform_auditor apps/pipeline-app/pipeline_app --format json
```

---

## apps 層の開発ガイド（audit CLI 前提）

`xform-auditor` CLI は、型注釈のみから **入力生成 → 関数実行 → 出力検証** を自動化します。以下のガイドに従うことで、**pytest を最小限に抑え**、型注釈ベースのテストで品質を担保できます。

---

## apps のコードに必要な要素

`xform-auditor` で自動テストを実行するには、以下の **5 要素** が必須です（mypy プラグインで強制）：

### 1. `@transform` デコレータ

```python
from xform_core.transforms_core import transform

@transform
def my_transform(...):
    ...
```

### 2. 第1引数が `Annotated[...]`（TR001-TR002）

```python
from typing import Annotated

@transform
def tokenize(X: Annotated[In, ...], ...):  # 第1引数が Annotated
    ...
```

### 3. 入力に `ExampleValue[T]` または `ExampleType[T]`（TR003-TR004）

```python
from xform_core import ExampleValue

class In(TypedDict):
    text: str

@transform
def tokenize(
    X: Annotated[In, ExampleValue[{"text": "hello world"}]],  # 入力例
    lower: bool = True
):
    ...
```

- **`ExampleValue[T]`**: 具体的な値または仕様オブジェクト（例: `HLOCVSpec(n=1000)`）。TransformFn では数行の定数や簡単な仕様オブジェクトで十分であり、複雑な再計算をする必要はない。
- **`ExampleType[T]`**: 型から自動生成（例: `ExampleType[pd.DataFrame]`）

### 4. 返り値が `Annotated[...]` with `Check["pkg.func"]`（TR005-TR008）

```python
from xform_core import Check

@transform
def tokenize(
    X: Annotated[In, ExampleValue[{"text": "hello world"}]],
    lower: bool = True
) -> Annotated[Out, Check["algo_trade_dtype.checks.check_tokens"]]:  # 出力検証
    ...
```

- **`Check["pkg.func"]`**: FQN で検査関数を指定
- 検査関数は `def check_tokens(output: Out) -> None:` のシグネチャで実装

### 5. docstring（TR009）

```python
@transform
def tokenize(
    X: Annotated[In, ExampleValue[{"text": "hello world"}]],
    lower: bool = True
) -> Annotated[Out, Check["algo_trade_dtype.checks.check_tokens"]]:
    """入力テキストをトークン列に分割する。"""  # docstring 必須
    return {"tokens": X["text"].lower().split() if lower else X["text"].split()}
```

---

## pytest が **不要** なケース

以下のケースでは、**`xform-auditor` CLI のみでテスト可能**であり、pytest を書く必要はありません：

### ✅ 1. 単純な Transform 関数の正常系テスト

```python
@transform
def add_feature(
    X: Annotated[pd.DataFrame, ExampleValue[HLOCVSpec(n=100)]],
    window: int = 5
) -> Annotated[pd.DataFrame, Check["algo_trade_dtype.checks.check_ohlcv"]]:
    """移動平均を追加する。"""
    X["ma"] = X["close"].rolling(window).mean()
    return X
```

**audit CLI でカバー**:
- 入力例（`HLOCVSpec(n=100)`）から DataFrame 生成
- `add_feature()` を実行
- `check_ohlcv()` で出力検証

### ✅ 2. 境界値・異常系テスト（将来拡張後）

```python
@transform
def filter_rows(
    X: Annotated[
        pd.DataFrame,
        ExampleValue[HLOCVSpec(n=1000)],  # 正常ケース
        ExampleValue[HLOCVSpec(n=0)],     # 境界値: 空 DataFrame（将来対応）
    ]
) -> Annotated[pd.DataFrame, Check["algo_trade_dtype.checks.check_ohlcv"]]:
    """閾値以上の行をフィルタリング。"""
    return X[X["close"] > 100]
```

**audit CLI でカバー**（複数 `ExampleValue` のサポート後）:
- 正常系・境界値を網羅的にテスト

### ✅ 3. 出力の型・構造検証

```python
@transform
def extract_features(
    X: Annotated[pd.DataFrame, ExampleValue[HLOCVSpec(n=100)]]
) -> Annotated[dict[str, float], Check["algo_trade_dtype.checks.check_feature_map"]]:
    """特徴量を抽出し dict で返す。"""
    return {"mean": X["close"].mean(), "std": X["close"].std()}
```

**audit CLI でカバー**:
- `check_feature_map()` で出力の型・キーの存在・値の範囲を検証

### ✅ 4. 単一パラメータでの動作確認

```python
@transform
def resample_ohlcv(
    X: Annotated[pd.DataFrame, ExampleValue[HLOCVSpec(n=1000, freq="1min")]],
    freq: str = "1H"  # デフォルト値のみテスト
) -> Annotated[pd.DataFrame, Check["algo_trade_dtype.checks.check_ohlcv"]]:
    """OHLCV をリサンプリング。"""
    return X.resample(freq).agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    })
```

**audit CLI でカバー**:
- デフォルト引数（`freq="1H"`）で実行・検証

---

## pytest が **必要** なケース

以下のケースでは、**audit CLI では不十分**であり、pytest による明示的なテストが必要です：

### ❌ 1. 複数パラメータの組み合わせテスト

```python
@pytest.mark.parametrize("freq", ["1H", "1D", "1W"])
@pytest.mark.parametrize("agg_method", ["mean", "median"])
def test_resample_combinations(freq, agg_method):
    """複数パラメータの組み合わせをテスト。"""
    X = gen_hlocv(HLOCVSpec(n=1000))
    result = resample_ohlcv(X, freq=freq, agg_method=agg_method)
    assert check_ohlcv(result) is None
```

**理由**: audit CLI は **デフォルトパラメータのみ**で実行（将来 `ParamCombinations` で対応予定）

### ❌ 2. DAG 実行・統合テスト

```python
def test_pipeline_dag():
    """DAG 全体を実行し、最終出力を検証。"""
    dag = load_dag("apps/pipeline-app/pipeline_app/dag.py")
    runner = SimpleRunner()
    artifacts = runner.execute(dag, inputs={"raw_data": ...})

    assert artifacts["final_output"].status == "success"
    assert len(artifacts["final_output"].data) > 0
```

**理由**: audit CLI は **単一 Transform の独立実行のみ**（将来 `--dag` モードで対応予定）

### ❌ 3. モック・スタブを用いた外部依存の制御

```python
@patch("pipeline_app.transforms.external_api_call")
def test_fetch_with_mock(mock_api):
    """外部 API を mock して Transform をテスト。"""
    mock_api.return_value = {"data": [1, 2, 3]}
    result = fetch_and_transform(X, api_key="test")
    assert result["processed"] == [1, 2, 3]
```

**理由**: audit CLI は **純粋関数のみ想定**、モック機構は未サポート

### ❌ 4. パフォーマンス・ベンチマーク

```python
def test_performance_add_feature():
    """実行時間が 1 秒以内であることを検証。"""
    X = gen_hlocv(HLOCVSpec(n=100000))
    start = time.perf_counter()
    result = add_feature(X, window=50)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0
```

**理由**: audit CLI は **実行時間計測なし**（将来 `--benchmark` で対応予定）

### ❌ 5. スナップショット・リグレッションテスト

```python
def test_extract_features_snapshot(snapshot):
    """過去の出力と差分を検証。"""
    X = gen_hlocv(HLOCVSpec(n=100, seed=42))
    result = extract_features(X)
    snapshot.assert_match(result, "features_v1.json")
```

**理由**: audit CLI は **スナップショット機能なし**（将来 `ExampleSnapshot` で対応予定）

### ❌ 6. プロパティベース・ファジングテスト

```python
@given(arrays=st.arrays(dtype=float, shape=(100,)))
def test_normalize_hypothesis(arrays):
    """ランダム入力でプロパティを検証。"""
    result = normalize(arrays)
    assert abs(result.mean()) < 1e-6  # 平均が 0 に近い
```

**理由**: audit CLI は **固定 Example のみ**（将来 `ExampleStrategy` で対応予定）

---

## まとめ：pytest の使い分け

| テスト種別 | audit CLI | pytest |
|-----------|-----------|--------|
| **単一 Transform の正常系** | ✅ 十分 | 不要 |
| **境界値・異常系**（単一入力） | ✅ 十分 | 不要 |
| **出力の型・構造検証** | ✅ 十分 | 不要 |
| **複数パラメータの組み合わせ** | ❌ 不十分 | ✅ 必要 |
| **DAG 実行・統合テスト** | ❌ 不十分 | ✅ 必要 |
| **モック・スタブ** | ❌ 不十分 | ✅ 必要 |
| **パフォーマンステスト** | ❌ 不十分 | ✅ 必要 |
| **スナップショット** | ❌ 不十分 | ✅ 必要 |
| **ファジング** | ❌ 不十分 | ✅ 必要 |

---

## 開発フロー

### 1. Transform 関数の実装

```python
@transform
def my_transform(
    X: Annotated[In, ExampleValue[spec]],
    param: T = default
) -> Annotated[Out, Check["pkg.check_func"]]:
    """docstring"""
    ...
```

### 2. 静的検査（mypy）

```bash
make typecheck  # TR001-TR009 を検証
```

### 3. 自動テスト（audit CLI）

```bash
uv run python -m xform_auditor apps/pipeline-app/pipeline_app
```

### 4. 必要に応じて pytest を追加

```bash
# パラメータ組み合わせ・DAG・モック等
uv run pytest apps/pipeline-app/tests/
```

---

## プロジェクト構成

```
TransformFn/
├── packages/            # 再利用可能ライブラリ
│   ├── xform-core/     # @transform, メタ型, mypy plugin
│   ├── xform-auditor/  # 監査 CLI
│   └── proj-dtypes/    # ドメイン型, Example, Check
├── apps/               # Transform 関数・DAG
│   ├── pipeline-app/    # サンプルアプリケーション
│   └── algo-trade-app/ # アルゴリズムトレーディングパイプライン（algo_trade_v3移植）
├── doc/                # 設計ドキュメント
└── output/             # 生成物（gitignored）
```

---

## 関連ドキュメント

- **[ARCHITECTURE.md](doc/ARCHITECTURE.md)**: 設計・仕様の詳細
- **[CLAUDE.md](CLAUDE.md)**: プロジェクト固有の開発規約
- **[TICKET-ARCHITECTURE-IMPLEMENTATION.md](doc/TICKET-ARCHITECTURE-IMPLEMENTATION.md)**: 実装タスクチケット
- **[ALGO_TRADE_V3_PORTING.md](doc/ALGO_TRADE_V3_PORTING.md)**: algo_trade_v3 移植調査メモ

---

## ライセンス

（未定）
