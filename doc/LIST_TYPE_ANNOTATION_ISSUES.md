# List[T] 型に対する TransformFn アノテーション解決の問題

## 概要

TransformFn の `xform-auditor` CLI が `List[T]` 型 (例: `List[SomeCustomType]`) に対して ExampleValue/Check の自動補完を正しく行えない問題が発生している。この問題は、特に中間データ型をリストで受け渡すパイプライン（例: `simulation.py` の `rank_predictions` 関数など）で顕著に現れる。

## 問題の詳細

### 1. TR003 エラーの発生

`xform-auditor` を実行すると、以下のような TR003 エラーが発生する：

```
[ERROR] algo_trade_transforms.simulation.rank_predictions - rank_predictions.predictions: TR003: ExampleValue for builtins.list が未登録です
```

このエラーは、`List[T]` 型の入力引数に対して `ExampleValue` が見つからないことを示している。

### 2. 型解決の不整合

現在のアーキテクチャでは、`RegisteredType` で個別型 `T` を登録しても、`List[T]` に対する自動補完が正しく行われない。

- `T` は `RegisteredType` で登録されている
- しかし `List[T]` は `builtins.list` として型解決され、`T` の登録が活かせない
- 結果として、`transform` 関数で `List[T]` を使用してもエラーとなる

## 現在のワークアラウンド

`apps/algo-trade/algo_trade_transforms/simulation.py` では、以下のようなワークアラウンドを実施している：

```python
@transform
def rank_predictions(
    predictions: Annotated[
        List[PredictionData],
        ExampleValue([
            {
                "date": "2024-01-01",
                "currency_pair": "USD_JPY",
                "prediction": 0.01,
                "actual_return": 0.005,
            },
            # ...
        ])
    ],
) -> Annotated[
    List[RankedPredictionData],
    Check("algo_trade_dtypes.checks.check_ranked_predictions"),
]:
```

## 期待される理想動作

### Type-First Approach での期待

`doc/APPS_INCONSISTENCIES_WITH_UPDATED_GUIDELINES.md` で示されている理想形：

```python
# 修正後
@transform
def rank_predictions(
    predictions: List[PredictionData],  # RegisteredTypeからExample/Checkが補完される
) -> List[RankedPredictionData]:       # RegisteredTypeからExample/Checkが補完される
    # ...
```

この動作を実現するためには、以下が必要：
1. `RegisteredType` で `PredictionData` が登録されている
2. システムが `List[PredictionData]` を検出した際、`PredictionData` の登録情報を活用
3. `List[RegisteredType]` に対しても `RegisteredType` が持つ Example/Check を自動的に適用

## 根本的な問題点

### 1. ジェネリック型への対応不足

現在の型解決システムは、`List[T]` のようなジェネリック型に対して、要素型 `T` から Example/Check を導出する機構を持たない。

### 2. mypy プラグインでの解決不能

TR003-TR006 の検証は mypy プラグインで行われるが、`List[T]` に対する登録情報の活用ロジックが実装されていない。

### 3. 型の正規化処理の欠如

`List[T]` 型を正規化する際に、要素型 `T` の登録情報を参照し、`ExampleValue[List[T]]` を構築する処理が不足。

## 影響範囲

- `List[T]`, `Dict[str, T]` などのジェネリック型を使用するすべての `@transform` 関数
- 特に金融データ処理など、データがリスト形式で渡されるパイプライン
- `TypedDict` を要素とするリスト型で顕著

## 解決案

### 方案1: ジェネリック型に対する登録システム強化

`RegisteredType[T]` がある場合、`List[T]` や `Dict[str, T]` といった型が現れた際にも自動的に該当型の Example/Check を適用可能にする。

```python
# ジェネリック型登録の概念
RegisteredType.register_list_type(PredictionData, "list_prediction_data")
# これにより、List[PredictionData] にも ExampleValue が適用可能に
```

### 方案2: 型解決アルゴリズムの改善

型解決処理を拡張し、以下のようなロジックを追加：

1. `List[T]` を検出
2. `T` が `RegisteredType` で登録されているか確認
3. 登録されている場合、`T` の Example を複数適用して `List[T]` 用の Example を構築
4. `T` の Check を `List[T]` 全体に適用（各要素検証）

### 方案3: 新しいメタ型の導入

ジェネリック型に対応する新しいメタ型を導入：

```python
from xform_core import ListExample, ListCheck

@transform
def func(
    data: Annotated[List[MyType], ListExample(registered_type="MyType")]
) -> Annotated[List[ResultType], ListCheck(registered_check="check_result")]
```

## 実装タスク

### タスク1: 型解決ロジックの拡張

- `packages/xform-core/xform_core/type_resolver.py` にジェネリック型解決ロジック追加
- `List[T]`, `Dict[K,V]`, `Tuple[T, ...]` などへの対応
- 登録済み型 `T` から `List[T]` に対応する Example/Check を構築

### タスク2: mypy プラグインの更新

- `packages/xform-core/xform_core/dtype_rules/plugin.py` にジェネリック型対応を追加
- TR003-TR006 で `List[T]` に対する登録情報を考慮

### タスク3: テストケースの追加

- `packages/xform-core/tests/test_type_resolver.py` にジェネリック型解決テスト
- 各ジェネリック型に対する Example/Check 自動補完の検証

## 期待される改善効果

1. **コード簡潔性**: `Annotated[List[T], ExampleValue(...)]` のような冗長記述が不要に
2. **DRY原則準拠**: 型定義1回で `List[T]`, `Dict[str, T]` など複数形式で利用可能に
3. **保守性向上**: 型定義更新が自動的にすべてのジェネリック構成に反映
4. **一貫性**: `RegisteredType` による型中心設計の理念と整合

## 関連ドキュメント

- `doc/transformfn_spec/APPS_INCONSISTENCIES_WITH_UPDATED_GUIDELINES.md` - Type-First Approach の指針
- `doc/AUTO_ANNOTATION_RESOLUTION.md` - 自動注釈解決設計書
- `doc/ARCHITECTURE.md` - TR001-TR009 の詳細仕様
- `doc/transformfn_spec/algo-trade-phase4-simulation-spec.md` - List[T] 使用例

## 完了条件

- `xform-auditor` が `List[T]` 型を含む関数でも TR003-TR006 を満たす
- `RegisteredType` で登録された型が `List[T]` などジェネリック型でも自動適用される
- `simulation.py` のような関数で `Annotated[List[T], ...]` 記述が不要になる