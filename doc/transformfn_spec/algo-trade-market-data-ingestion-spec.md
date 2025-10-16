# algo-trade-app Market Data Ingestion 仕様

## 概要

Yahoo Finance（株式/ETF）と CCXT 経由の暗号資産 OHLCV データを共通フォーマットへ正規化し、下流の特徴量生成・学習パイプラインへ渡す Transform 群の仕様を定義する。中間の *Request* 型は設けず、`MarketDataIngestionConfig` から各プロバイダの取得・正規化を直接行う構成とし、型メタデータは `RegisteredType` で集約管理する。

## パイプライン構造

```mermaid
graph TD
    C0["<b>MarketDataIngestionConfig</b><br/>────────<br/>Example:<br/>yahoo: AAPL,MSFT<br/>ccxt: BTC/USDT,ETH/USDT<br/>期間: 2024-01-01~01-10<br/>────────<br/>Check:<br/>少なくとも1プロバイダ存在<br/>日付範囲・シンボル妥当性"]

    Y1["<b>ProviderBatchCollection</b><br/>(Yahoo)<br/>────────<br/>Example:<br/>AAPL 48時間分の<br/>OHLCV DataFrame<br/>────────<br/>Check:<br/>プロバイダ取得結果の<br/>最低要件検証"]

    X1["<b>ProviderBatchCollection</b><br/>(CCXT)<br/>────────<br/>Example:<br/>BTC/USDT 48時間分の<br/>OHLCV DataFrame<br/>────────<br/>Check:<br/>プロバイダ取得結果の<br/>最低要件検証"]

    B1["<b>ProviderBatchAggregate</b><br/>────────<br/>Example:<br/>Yahoo+CCXT<br/>統合バッチリスト<br/>────────<br/>Check:<br/>複数プロバイダ<br/>集約結果の整合性検証"]

    D1["<b>NormalizedOHLCVBundle</b><br/>────────<br/>Example:<br/>AAPL & BTC/USDT<br/>正規化レコード2件+<br/>メタデータ<br/>────────<br/>Check:<br/>正規化済みレコードの<br/>一貫性検証"]

    D2["<b>MultiAssetOHLCVFrame</b><br/>────────<br/>Example:<br/>(timestamp,symbol)<br/>MultiIndex<br/>AAPL & BTC/USDT<br/>統合DataFrame<br/>────────<br/>Check:<br/>MultiIndex構造と<br/>OHLCV列型の検証"]

    D3["<b>MarketDataSnapshotMeta</b><br/>────────<br/>Example:<br/>96レコード, S3パス,<br/>snapshot_id,<br/>作成日時<br/>────────<br/>Check:<br/>永続化メタ情報の<br/>整合性検証"]

    C0 -->|"@transform<br/>fetch_yahoo_finance_ohlcv"| Y1
    C0 -->|"@transform<br/>fetch_ccxt_ohlcv"| X1
    Y1 -->|"@transform<br/>combine_provider_batches"| B1
    X1 -->|"@transform<br/>combine_provider_batches"| B1
    B1 -->|"@transform<br/>normalize_provider_batches<br/>(target_frequency='1H')"| D1
    D1 -->|"@transform<br/>merge_market_data_bundle<br/>(join_policy='outer')"| D2
    D2 -->|"@transform<br/>persist_market_data_snapshot<br/>(destination='s3://...snapshots')"| D3

    style C0 fill:#e8f5ff,stroke:#333,stroke-width:2px
    style Y1 fill:#f2faff,stroke:#333,stroke-width:2px
    style X1 fill:#f2faff,stroke:#333,stroke-width:2px
    style B1 fill:#e8f5ff,stroke:#333,stroke-width:2px
    style D1 fill:#e8f5ff,stroke:#333,stroke-width:2px
    style D2 fill:#e8f5ff,stroke:#333,stroke-width:2px
    style D3 fill:#e8f5ff,stroke:#333,stroke-width:2px
```

## 作成する型定義（`apps/algo-trade-app/algo_trade_dtype/types.py`）

既存の `Frequency` や `CurrencyPair` を再利用しつつ、以下の新規 TypedDict / StrEnum を追加する。

```python
from typing import List, Dict, Optional
from typing_extensions import TypedDict
import pandas as pd

class CCXTExchange(StrEnum):
    """CCXT で利用する取引所識別子。"""

    BINANCE = "binance"
    BYBIT = "bybit"
    KRAKEN = "kraken"


class YahooFinanceConfig(TypedDict):
    """Yahoo Finance データ取得設定。"""

    tickers: List[str]       # 例: ["AAPL", "MSFT"]
    start_date: str          # ISO8601 (YYYY-MM-DD)
    end_date: str            # ISO8601 (YYYY-MM-DD)
    frequency: Frequency     # 最小粒度: 1日 (Yahoo Finance の制約)
    use_adjusted_close: bool


class CCXTConfig(TypedDict):
    """CCXT データ取得設定。"""

    symbols: List[str]       # 例: ["BTC/USDT", "ETH/USDT"]
    start_date: str          # ISO8601 (YYYY-MM-DD)
    end_date: str            # ISO8601 (YYYY-MM-DD)
    frequency: Frequency     # 最小粒度: 1分 (取引所による)
    exchange: CCXTExchange
    rate_limit_ms: int       # レート制限 (ミリ秒)


class MarketDataIngestionConfig(TypedDict, total=False):
    """プロバイダ横断の取得条件（使用するプロバイダのみ指定）。"""

    yahoo: YahooFinanceConfig     # オプショナル
    ccxt: CCXTConfig              # オプショナル


class ProviderOHLCVBatch(TypedDict):
    """単一シンボルの取得結果とメタ情報。"""

    provider: str            # "yahoo" or "ccxt"
    symbol: str
    frame: pd.DataFrame      # 各行は FXDataSchema (OHLCVSchema) に準拠
    frequency: Frequency


class ProviderBatchCollection(TypedDict):
    """特定プロバイダの一括取得結果。"""

    provider: str                    # "yahoo" / "ccxt"
    batches: List[ProviderOHLCVBatch]


class ProviderBatchAggregate(TypedDict):
    """複数プロバイダのバッチを結合した集合。"""

    providers: List[str]
    batches: List[ProviderOHLCVBatch]


class NormalizedOHLCVRecord(TypedDict):
    """正規化済み 1 レコード。"""

    timestamp: str
    provider: str
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class NormalizedOHLCVBundle(TypedDict):
    """正規化済みレコードとメタデータ。"""

    records: List[NormalizedOHLCVRecord]
    metadata: Dict[str, str]


class MultiAssetOHLCVFrame(TypedDict):
    """MultiIndex DataFrame をラップした構造。"""

    frame: pd.DataFrame      # index=(timestamp, symbol), 列は FXDataSchema (OHLCVSchema) 準拠
    symbols: List[str]
    providers: List[str]


class MarketDataSnapshotMeta(TypedDict):
    """永続化済みスナップショットのメタ情報。"""

    snapshot_id: str
    record_count: int
    storage_path: str
    created_at: str          # ISO8601 (UTC)
```

### RegisteredType 登録方針

- 上記 TypedDict すべてを `RegisteredType` で登録し、Example / Check を型レベルで付与する。
- `ProviderOHLCVBatch.frame` は既存の `DataFrameReg`（OHLCV チェック付き）を再利用。
- `ProviderBatchCollection.batches` や `ProviderBatchAggregate.batches` の例は `gen_provider_aggregate` など静的値を返すジェネレータで供給する。

## 作成する Example（`apps/algo-trade-app/algo_trade_dtype/generators.py`）

```python
def gen_ingestion_config() -> MarketDataIngestionConfig:
    """両方のプロバイダを使用する例。"""
    return {
        "yahoo": {
            "tickers": ["AAPL", "MSFT"],
            "start_date": "2024-01-01",
            "end_date": "2024-01-10",
            "frequency": Frequency.HOUR_1,
            "use_adjusted_close": True,
        },
        "ccxt": {
            "symbols": ["BTC/USDT", "ETH/USDT"],
            "start_date": "2024-01-01",
            "end_date": "2024-01-10",
            "frequency": Frequency.HOUR_1,
            "exchange": CCXTExchange.BINANCE,
            "rate_limit_ms": 1000,
        },
    }


def gen_yahoo_only_config() -> MarketDataIngestionConfig:
    """Yahoo Finance のみを使用する例（日足データ）。"""
    return {
        "yahoo": {
            "tickers": ["AAPL", "MSFT", "GOOGL"],
            "start_date": "2024-01-01",
            "end_date": "2024-01-10",
            "frequency": Frequency.DAY_1,  # Yahoo Finance は日足が最小粒度
            "use_adjusted_close": True,
        },
    }


def gen_ccxt_only_config() -> MarketDataIngestionConfig:
    """CCXT のみを使用する例（分足データ）。"""
    return {
        "ccxt": {
            "symbols": ["BTC/USDT", "ETH/USDT"],
            "start_date": "2024-01-01",
            "end_date": "2024-01-10",
            "frequency": Frequency.MIN_15,  # CCXT は分足から取得可能
            "exchange": CCXTExchange.BINANCE,
            "rate_limit_ms": 1000,
        },
    }


def gen_mixed_frequency_config() -> MarketDataIngestionConfig:
    """異なる粒度のデータソースを混在させる例。"""
    return {
        "yahoo": {
            "tickers": ["AAPL", "MSFT"],
            "start_date": "2024-01-01",
            "end_date": "2024-01-10",
            "frequency": Frequency.DAY_1,  # 日足
            "use_adjusted_close": True,
        },
        "ccxt": {
            "symbols": ["BTC/USDT", "ETH/USDT"],
            "start_date": "2024-01-01",
            "end_date": "2024-01-10",
            "frequency": Frequency.HOUR_1,  # 時間足
            "exchange": CCXTExchange.BINANCE,
            "rate_limit_ms": 1000,
        },
    }


def gen_yahoo_batch_collection() -> ProviderBatchCollection:
    frame = gen_sample_ohlcv(n=48, start_price=150.0, seed=7)
    frame.reset_index(inplace=True)
    batch = {
        "provider": "yahoo",
        "symbol": "AAPL",
        "frame": frame,
        "frequency": Frequency.HOUR_1,
    }
    return {"provider": "yahoo", "batches": [batch]}


def gen_ccxt_batch_collection() -> ProviderBatchCollection:
    frame = gen_sample_ohlcv(n=48, start_price=45000.0, seed=11)
    frame.reset_index(inplace=True)
    batch = {
        "provider": "ccxt",
        "symbol": "BTC/USDT",
        "frame": frame,
        "frequency": Frequency.HOUR_1,
    }
    return {"provider": "ccxt", "batches": [batch]}


def gen_provider_aggregate() -> ProviderBatchAggregate:
    yahoo = gen_yahoo_batch_collection()["batches"][0]
    ccxt = gen_ccxt_batch_collection()["batches"][0]
    return {
        "providers": ["yahoo", "ccxt"],
        "batches": [yahoo, ccxt],
    }


def gen_normalized_bundle() -> NormalizedOHLCVBundle:
    return {
        "records": [
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "provider": "yahoo",
                "symbol": "AAPL",
                "open": 150.0,
                "high": 151.0,
                "low": 149.5,
                "close": 150.5,
                "volume": 1_200_000.0,
            },
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "provider": "ccxt",
                "symbol": "BTC/USDT",
                "open": 45000.0,
                "high": 45200.0,
                "low": 44850.0,
                "close": 45120.0,
                "volume": 320.5,
            },
        ],
        "metadata": {"target_frequency": "1H", "source_count": "2"},
    }


def gen_multiasset_frame() -> MultiAssetOHLCVFrame:
    normalized = gen_normalized_bundle()
    frame = pd.DataFrame(normalized["records"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame.set_index(["timestamp", "symbol"], inplace=True)
    return {
        "frame": frame,
        "symbols": ["AAPL", "BTC/USDT"],
        "providers": ["yahoo", "ccxt"],
    }


def gen_snapshot_meta() -> MarketDataSnapshotMeta:
    return {
        "snapshot_id": "snapshot_20240110T000000Z",
        "record_count": 96,
        "storage_path": "output/data/snapshots/2024/01/10/market.parquet",
        "created_at": "2024-01-10T00:15:00Z",
    }
```

## 作成する Check 関数（`apps/algo-trade-app/algo_trade_dtype/checks.py`）

```python
def check_ingestion_config(config: MarketDataIngestionConfig) -> None:
    """日付範囲、対象シンボル、周波数設定の妥当性を検証。"""
    # 少なくとも yahoo または ccxt のいずれかが存在することを確認。
    # 各プロバイダ設定で start_date < end_date、シンボル/ticker重複なしを検証。


def check_batch_collection(collection: ProviderBatchCollection) -> None:
    """各プロバイダの取得結果が最低要件を満たすか検証。"""
    # provider 名の正当性、batches の各要素に対して check_provider_batch を再利用。


def check_provider_batch(batch: ProviderOHLCVBatch) -> None:
    """個別バッチの DataFrame 構造を検証。"""
    # timestamp 昇順、OHLCV 列の存在、volume の有限性など。


def check_provider_aggregate(aggregate: ProviderBatchAggregate) -> None:
    """複数プロバイダの集約結果を検証。"""
    # providers と batches の長さ整合性、provider 名の一意性。


def check_normalized_bundle(bundle: NormalizedOHLCVBundle) -> None:
    """正規化済みレコードの一貫性を検証。"""
    # 各レコードの価格 > 0、timestamp ISO8601、provider/symbol の非空を確認。


def check_multiasset_frame(frame_info: MultiAssetOHLCVFrame) -> None:
    """MultiIndex DataFrame が想定構造を満たすか検証。"""
    # index=(timestamp,symbol)、列セットが OHLCV のみ、dtype が float64/int64。


def check_snapshot_meta(meta: MarketDataSnapshotMeta) -> None:
    """永続化メタ情報の整合性を検証。"""
    # snapshot_id フォーマット、record_count > 0、storage_path prefix を検証。
```

## RegisteredType 追加一覧（`apps/algo-trade-app/algo_trade_dtype/registry.py`）

| 型 | Example | Check |
| --- | --- | --- |
| `MarketDataIngestionConfig` | `gen_ingestion_config()` / `gen_yahoo_only_config()` / `gen_ccxt_only_config()` | `check_ingestion_config` |
| `YahooFinanceConfig` | `gen_ingestion_config()["yahoo"]` | 既存のチェックで対応 |
| `CCXTConfig` | `gen_ingestion_config()["ccxt"]` | 既存のチェックで対応 |
| `ProviderOHLCVBatch` | `gen_yahoo_batch_collection()["batches"][0]` | `check_provider_batch` |
| `ProviderBatchCollection` | `gen_yahoo_batch_collection()` / `gen_ccxt_batch_collection()` | `check_batch_collection` |
| `ProviderBatchAggregate` | `gen_provider_aggregate()` | `check_provider_aggregate` |
| `NormalizedOHLCVRecord` | `gen_normalized_bundle()["records"][0]` | `check_normalized_bundle` |
| `NormalizedOHLCVBundle` | `gen_normalized_bundle()` | `check_normalized_bundle` |
| `MultiAssetOHLCVFrame` | `gen_multiasset_frame()` | `check_multiasset_frame` |
| `MarketDataSnapshotMeta` | `gen_snapshot_meta()` | `check_snapshot_meta` |
| `CCXTExchange` | `CCXTExchange.BINANCE` など列挙値 | 既存の列挙チェック（新設不要なら `None`） |

## 作成する Transformer（`apps/algo-trade-app/algo_trade_app/market_data.py` 想定）

```python
@transform
def fetch_yahoo_finance_ohlcv(
    config: YahooFinanceConfig,
) -> ProviderBatchCollection:
    """yfinance から ticker 群を取得し、ProviderBatchCollection として返す。"""
```
- `config["tickers"]` をループし、`yfinance.download` で取得。
- `config["use_adjusted_close"]` に応じて `Adj Close` を `close` に置き換え。
- 欠損行を削除し、`provider="yahoo"` を付与した `ProviderOHLCVBatch` を構築。

```python
@transform
def fetch_ccxt_ohlcv(
    config: CCXTConfig,
) -> ProviderBatchCollection:
    """CCXT 取引所から暗号資産の OHLCV を取得。"""
```
- `config["exchange"]` を利用して `ccxt` クライアントを初期化。
- `config["symbols"]` ごとに `fetch_ohlcv` を実行し、`provider="ccxt"` として格納。
- `config["rate_limit_ms"]` に従ってレート制限を適用、失敗時は指数バックオフで再試行。

```python
@transform
def combine_provider_batches(
    yahoo_batches: ProviderBatchCollection,
    ccxt_batches: ProviderBatchCollection,
) -> ProviderBatchAggregate:
    """複数プロバイダの取得結果を 1 つの集合にまとめる。"""
```
- providers 配列をマージし重複を排除。
- `batches` リストを連結し、provider と symbol の組が重複しないことを検証。

```python
@transform
def normalize_provider_batches(
    aggregate: ProviderBatchAggregate,
    *,
    target_frequency: Frequency = Frequency.HOUR_1,
    resample_method: str = "forward_fill",  # "forward_fill" or "upsample"
) -> NormalizedOHLCVBundle:
    """取得した DataFrame 群を統一スキーマへ変換。"""
```
- 各 DataFrame を UTC にそろえ、`target_frequency` へリサンプリング。
- **リサンプリング方針**:
  - **ダウンサンプリング** (分足→時間足、時間足→日足): OHLCV の集約ルールを適用 (open=first, high=max, low=min, close=last, volume=sum)
  - **アップサンプリング** (日足→時間足):
    - `resample_method="forward_fill"`: 前方補完（リーク防止のため、未来のデータは使わない）
    - `resample_method="upsample"`: エラーを発生（粒度の粗いデータを細かくすることは推奨しない）
- **リーク防止**: リサンプリング時に未来のデータを参照しないよう、`label='left'`, `closed='left'` を使用。
- OHLCV 列を float64 にキャストし、`NormalizedOHLCVRecord` のリストを生成。

```python
@transform
def merge_market_data_bundle(
    bundle: NormalizedOHLCVBundle,
    *,
    join_policy: str = "outer",
) -> MultiAssetOHLCVFrame:
    """正規化レコードを MultiIndex DataFrame に変換。"""
```
- `pivot_table` で `(timestamp, symbol)` を index に、OHLCV を列に配置。
- `join_policy` に応じて `outer` / `inner` を制御し、欠損を前方補完する。

```python
@transform
def persist_market_data_snapshot(
    multiasset_frame: MultiAssetOHLCVFrame,
    *,
    destination: str = "output/data/snapshots",
) -> MarketDataSnapshotMeta:
    """Parquet へ書き出し、スナップショット ID とメタ情報を返却。"""
```
- `destination` と最新 timestamp から `snapshot_id` を生成。
- 実際の書き出しは I/O 層に委譲し、メタ情報を TypedDict で返す。
- `destination` はローカルパスをデフォルトとし、S3パス（`s3://...`）にも対応。

## Audit 実行

```bash
uv run python -m xform_auditor apps/algo-trade-app/algo_trade_app/market_data.py
```

期待結果: `6 transforms, 6 OK, 0 VIOLATION, 0 ERROR, 0 MISSING`

## 実装メモ

### データソースごとの時間粒度制約

- **Yahoo Finance**: 最小粒度は**日足** (`Frequency.DAY_1`)。分足・時間足は取得不可。
- **CCXT**: 取引所により異なるが、**分足** (`Frequency.MIN_1`) から取得可能。
- `check_ingestion_config` で各プロバイダの `frequency` が最小粒度制約を満たすか検証する。

### リサンプリングとリーク防止

- **ダウンサンプリング** (高頻度→低頻度): 安全。OHLCV集約ルールを適用。
  - 例: 1分足 → 1時間足、1時間足 → 1日足
- **アップサンプリング** (低頻度→高頻度): 注意が必要。
  - `resample_method="forward_fill"`: 前方補完（過去のデータのみ使用、リークなし）
  - `resample_method="upsample"`: エラー発生（粒度の粗いデータを細かくすることは非推奨）
- **リーク防止の実装例**:
  ```python
  # ダウンサンプリング（1時間足 → 1日足）
  df_daily = df_hourly.resample('1D', label='left', closed='left').agg({
      'open': 'first',
      'high': 'max',
      'low': 'min',
      'close': 'last',
      'volume': 'sum'
  })

  # アップサンプリング（1日足 → 1時間足、前方補完）
  df_hourly = df_daily.resample('1H', label='left', closed='left').ffill()
  ```
- `label='left'` と `closed='left'` により、各時間窓の**左端**（開始時刻）のデータのみを使用し、未来のデータを参照しない。

### 異なる粒度のデータソース混在時の注意

- `gen_mixed_frequency_config()` のように、日足と時間足を混在させる場合:
  - `normalize_provider_batches` で統一粒度へリサンプリング
  - `target_frequency` は**最も粗い粒度**に合わせるのが安全（例: 日足に統一）
  - または、`merge_market_data_bundle` で `join_policy="outer"` を使い、欠損を前方補完

### その他

- `MarketDataIngestionConfig` は使用するプロバイダのみを含み、将来的なプロバイダ追加に対応。
- `ProviderBatchCollection` の Example は 1 シンボルのみとし、Audit 実行時のデータ量を抑える。
- `normalize_provider_batches` 内で `provider` / `symbol` の組合せをキーにキャッシュキーを構築し、取得データの再現性を担保する。
- `persist_market_data_snapshot` の Check では `storage_path` が `destination` で始まること、`record_count` が `MultiAssetOHLCVFrame.frame` の行数と一致することを検証する。
- ローカルパスの場合、親ディレクトリが存在しない場合は自動作成する。S3パスの場合はboto3などのSDKを利用してアップロード。
