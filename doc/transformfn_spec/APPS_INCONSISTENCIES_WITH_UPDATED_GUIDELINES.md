# apps/** パッケージの transformfn-spec-writer 指針更新に対する不整合事項

## 概要

`@.claude/agents/transformfn-spec-writer.md` の方針更新（Type-First Approach）に対応するため、`apps/**` 配下の実装における不整合・未対応事項をまとめた。

## 発見された主な問題点

### 1. Type-First Approach への対応不足（最大の問題）

**問題点**: 
- 現在の実装では、`@transform` デコレータで直接 `ExampleValue` や `Check` を記述している箇所が多々存在
- `RegisteredType` による一元管理ではなく、関数個別にアノテーションが記述されている
- これにより、型メタデータの再利用性やDRY原則が損なわれている

**具体的なファイル**:
- `/Users/mikke/git_dir/TransformFn/apps/algo-trade/algo_trade_transforms/simulation.py`
  - `rank_predictions`, `select_top_currency`, `simulate_buy_scenario`, `calculate_performance_metrics` で直接 `Annotated[Type, ExampleValue(...)]` を使用
  - これらの関数で `ExampleValue` を直接記述する代わりに、型レベルで `RegisteredType` から自動補完されるべき

**理想的な修正**:
```python
# 修正前
@transform
def rank_predictions(
    predictions: Annotated[
        List[PredictionData],
        ExampleValue(gen_prediction_data()),
    ],
) -> Annotated[
    List[RankedPredictionData],
    Check("algo_trade_dtypes.checks.check_ranked_predictions"),
]:
    # ...

# 修正後
@transform
def rank_predictions(
    predictions: List[PredictionData],  # RegisteredTypeからExample/Checkが補完される
) -> List[RankedPredictionData]:       # RegisteredTypeからExample/Checkが補完される
    # ...
```

### 2. 中間データ型の網羅性の不足（重要）

**問題点**:
- `RegisteredType` 指針の「CRITICAL - 中間データ型の網羅性」に従っていない
- パイプライン A→B→C→D を構成する場合、A, B, C, D 全ての型が `RegisteredType` として定義されている必要がある
- 現在の実装では、中間データ型（B, C）の登録が漏れている可能性がある

**検証が必要なパイプライン**:
- `/Users/mikke/git_dir/TransformFn/apps/algo-trade/algo_trade_transforms/simulation.py`
  - `rank_predictions` (入力: `List[PredictionData]`, 出力: `List[RankedPredictionData]`)
  - `select_top_currency` (入力: `List[RankedPredictionData]`, 出力: `List[SelectedCurrencyData]`)
  - `simulate_buy_scenario` (入力: `List[SelectedCurrencyData]`, 出力: `SimulationResult`)
  - `calculate_performance_metrics` (入力: `SimulationResult`, 出力: `PerformanceMetrics`)

**具体的に不足している登録**:
- `PredictionData` 型: `types.py` には定義されているが `registry.py` に `RegisteredType` として登録されていない
- `RankedPredictionData` 型: `types.py` には定義されているが `registry.py` に `RegisteredType` として登録されていない
- `SelectedCurrencyData` 型: `types.py` には定義されているが `registry.py` に `RegisteredType` として登録されていない
- `SimulationResult` 型: これは既に `SimulationResultReg` として登録されている
- `PerformanceMetrics` 型: これは既に `PerformanceMetricsReg` として登録されている

**検出方法**:
- `types.py` で定義されている `PredictionData`, `RankedPredictionData`, `SelectedCurrencyData` は `RegisteredType` で宣言されていない
- `registry.py` にはこれらの型に対応する登録 (`PredictionDataReg`, `RankedPredictionDataReg`, `SelectedCurrencyDataReg`) が存在しない
- しかし、`generators.py` には対応する生成関数 (`gen_prediction_data`, `gen_ranked_prediction_data`, `gen_selected_currency_data`) が存在する

### 3. ExampleValue の複雑化（Simple is better 原則の逸脱）

**問題点**:
- `simulation.py` で ExampleValue に生成関数を直接使用している箇所がある (`gen_prediction_data()`, `gen_ranked_prediction_data()` など)
- 指針では「Simple is better」原則により、ハードコードされた具体値を使用することが推奨されている
- 複雑な生成関数ではなく、必要最小限の静的値を使用すべき

### 4. 例外的使用の誤判断

**問題点**:
- `simulation.py` の `@transform` 関数で、型固有の検証ではなく transformer 固有の検証である場合にのみ個別に `Check` を記述すべき
- 現在の実装では、すべての出力型に対して `Check` を指定しているが、型固有の検証であれば `RegisteredType` で一元管理すべき

## 具体的な修正提案

### 1. simulation.py の修正

**修正対象**:
1. `rank_predictions` 関数の入力・出力型を `RegisteredType` で宣言し、関数内のアノテーションを削除
2. `select_top_currency` 関数の入力・出力型を `RegisteredType` で宣言し、関数内のアノテーションを削除
3. `simulate_buy_scenario` 関数の入力・出力型を `RegisteredType` で宣言し、関数内のアノテーションを削除
4. `calculate_performance_metrics` 関数の入力・出力型を `RegisteredType` で宣言し、関数内のアノテーションを削除

**実施手順**:
1. `algo_trade_dtypes/registry.py` に `PredictionData`, `RankedPredictionData`, `SelectedCurrencyData` の各型を `RegisteredType` で登録
2. `simulation.py` の各関数から関数個別の `Annotated[Type, ExampleValue(...)]` を削除
3. `simulation.py` の各関数から関数個別の `Annotated[Type, Check("...")]` を削除（型固有の検証であれば）

### 2. registry.py の修正

**追加が必要な RegisteredType 宣言**:
```python
# generators.py から必要な生成関数をインポート
from .generators import (
    gen_prediction_data,
    gen_ranked_prediction_data,
    gen_selected_currency_data,
    # ... 既存のインポート
)

# types.py から必要な型をインポート
from .types import (
    # ... 既存のインポート
    PredictionData,
    RankedPredictionData,
    SelectedCurrencyData,
)

# 新たに登録する型
PredictionDataReg: RegisteredType[PredictionData] = (
    RegisteredType(PredictionData)
    .with_example(gen_prediction_data()[0], "sample_prediction_data")
    .with_example(gen_prediction_data()[1], "sample_prediction_data2")
    # 他のExampleも必要に応じて追加
    # Check関数が適切に定義されていれば以下のように追加:
    # .with_check(check_prediction_data)  # type: ignore[arg-type]
)

RankedPredictionDataReg: RegisteredType[RankedPredictionData] = (
    RegisteredType(RankedPredictionData)
    .with_example(gen_ranked_prediction_data()[0], "sample_ranked_prediction_data")
    .with_example(gen_ranked_prediction_data()[1], "sample_ranked_prediction_data2")
    # 他のExampleも必要に応じて追加
    # Check関数が適切に定義されていれば以下のように追加:
    # .with_check(check_ranked_prediction_data)  # type: ignore[arg-type]
)

SelectedCurrencyDataReg: RegisteredType[SelectedCurrencyData] = (
    RegisteredType(SelectedCurrencyData)
    .with_example(gen_selected_currency_data()[0], "sample_selected_currency_data")
    .with_example(gen_selected_currency_data()[1], "sample_selected_currency_data2")
    # 他のExampleも必要に応じて追加
    # Check関数が適切に定義されていれば以下のように追加:
    # .with_check(check_selected_currency_data)  # type: ignore[arg-type]
)
```

また、`ALL_REGISTERED_TYPES` にはこれらの新しい型を追加し、`__all__` にも含める必要がある。

### 3. transforms.py の部分的修正

一部の関数（`resample_ohlcv`, `calculate_rsi`, `calculate_adx` など）はすでに型ヒントのみを使用しているため、この部分は問題なし。
ただし、`pandas.DataFrame` は型として `RegisteredType` で登録されているが、具体的な金融データ用の DataFrame 型を定義することを検討。

### 4. training.py の部分的修正

一部の関数（`get_cv_splits`, `calculate_rmse` など）は `Check` を関数個別に指定しているが、型固有の検証であれば `RegisteredType` で管理する方が良い。

## audit コマンド実行時の影響

現在の実装では audit が通過しているが、Type-First Approach に完全に準拠することで以下の恩恵が期待できる:

1. 型メタデータの再利用性向上（DRY原則）
2. 一元管理による保守性の向上
3. パイプライン構成時の型整合性の保証
4. ExampleValue の簡潔化によるテストの明確化

## 検証ステータス

- [ ] simulation.py の修正と audit 通過確認
- [ ] 中間データ型の `RegisteredType` 登録状況確認
- [ ] Example データの静的値への置換
- [ ] Check 関数の型固有 vs transformer 固有の分類と修正
- [ ] pipeline-app の同様の修正確認

## 結論

現在の `apps/**` 実装は audit コマンドパスを達成しているが、`transformfn-spec-writer` の最新指針（特に Type-First Approach）への準拠が不十分である。特に `simulation.py` で `ExampleValue` と `Check` を関数個別に記述している構造は、更新された指針に反している。

また、`PredictionData`, `RankedPredictionData`, `SelectedCurrencyData` の3つの型が `RegisteredType` として登録されておらず、`transformfn-spec-writer` 指針の「CRITICAL - 中間データ型の網羅性」に違反している。

これらを修正することで、より型中心・再利用可能な TransformFn 実装になることが期待される。