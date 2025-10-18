# DataFrame型の制約と検証方針

## 背景

TransformFnアーキテクチャでは、`pd.DataFrame`を入出力とするtransformerが多数存在するが、Pythonの型システムでは実行時のカラム構成を静的に表現できない。

## 現状の設計

### 型定義の役割分担

1. **型ヒント（TypedDict/TypeAlias）**: 大まかな構造を示す
   - 例: `PriceBarsFrame: TypeAlias = pd.DataFrame`
   - 例: `FXDataSchema(TypedDict)` - 行スキーマとして定義

2. **Check関数**: 実行時に詳細な制約を検証
   - カラムの存在確認
   - データ型の検証
   - 値の範囲や関係性の検証

### 具体例: OHLCV DataFrame

**型定義** (`algo_trade_dtypes/types.py`):
```python
# DataFrame型（静的型チェック用）
PriceBarsFrame: TypeAlias = pd.DataFrame

# 行スキーマ（ドキュメント目的、一部の型チェックで利用可能）
class FXDataSchema(TypedDict):
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
```

**Check関数** (`algo_trade_dtypes/checks.py`):
```python
def check_hlocv_dataframe(frame: pd.DataFrame) -> None:
    """HLOCV DataFrameが全ての制約を満たすか検証する。"""
    check_hlocv_dataframe_length(frame)    # カラム存在確認
    check_hlocv_dataframe_notnull(frame)   # 欠損値確認
    _validate_price_columns(frame)         # 価格列の値検証
    _validate_price_relationships(frame)   # 価格の関係性検証
    _validate_volume(frame)                # 出来高検証
```

## DataFrame型定義のパターン

### パターン1: TypeAlias（最もシンプル）

```python
PriceBarsFrame: TypeAlias = pd.DataFrame
```

- **用途**: 型ヒントでの使用、大まかな意図の表明
- **静的チェック**: DataFrameであることのみ保証
- **実行時検証**: Check関数で詳細を検証

### パターン2: TypedDict（行スキーマ定義）

```python
class FXDataSchema(TypedDict):
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
```

- **用途**: 行の構造をドキュメント化、一部の型チェッカーで活用可能
- **静的チェック**: 限定的（DataFrameとは別の型として扱われる）
- **実行時検証**: Check関数で詳細を検証

### パターン3: TypedDict with total=False（オプショナルフィールド）

volumeの有無など、カラムがオプショナルな場合:

```python
from typing import TypedDict

class OHLCSchemaBase(TypedDict):
    """必須フィールド"""
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float

class OHLCVSchemaOptional(TypedDict, total=False):
    """オプショナルフィールド"""
    volume: float

class OHLCVSchema(OHLCSchemaBase, OHLCVSchemaOptional):
    """OHLC(V)データの行スキーマ（volumeはオプショナル）"""
    pass
```

## 推奨事項

### 型定義時

1. **TypeAlias**: DataFrame型として使う場合は簡潔に定義
2. **コメント**: カラム構成の概要をコメントで記載
   ```python
   # OHLCV列を持つDataFrame (timestamp, open, high, low, close, volume)
   PriceBarsFrame: TypeAlias = pd.DataFrame
   ```
3. **TypedDict**: 行スキーマのドキュメント化が必要な場合に使用

### Check関数実装時

1. **xform-core汎用関数**: `check_dataframe_has_columns`, `check_dataframe_not_empty` などを活用
2. **ドメイン固有検証**: アプリ固有のチェックロジックを実装
3. **段階的検証**: 存在確認 → 型確認 → 値の検証 → 関係性の検証

### 仕様書記述時（transformfn-spec-writer）

Mermaid図のExample欄に、DataFrameの構造情報を記載:

```markdown
Example:
DataFrame with columns:
timestamp, open, high,
low, close, volume
```

## 将来の拡張案

より厳密な静的型チェックが必要になった場合の選択肢:

1. **Pandera**: DataFrameスキーマ検証ライブラリ
2. **Pydantic**: データ検証ライブラリ（DataFrame対応版）
3. **カスタムmypyプラグイン拡張**: DataFrame構造の静的チェック

現時点では、型ヒント + Check関数の組み合わせで十分な検証が可能。
