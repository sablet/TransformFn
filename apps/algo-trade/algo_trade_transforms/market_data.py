"""Market Data Ingestion用のTransform群。

Yahoo FinanceとCCXT経由のOHLCVデータを正規化し、
下流パイプラインへ渡すTransform群を実装する。
"""

from __future__ import annotations


from xform_core.transforms_core import transform

from algo_trade_dtypes.types import (
    CCXTConfig,
    YahooFinanceConfig,
    MarketDataIngestionConfig,
    MarketDataProvider,
    MultiAssetOHLCVFrame,
    NormalizedOHLCVBundle,
    ProviderBatchCollection,
    ProviderOHLCVBatch,
    Frequency,
    MarketDataSnapshotMeta,
)
from algo_trade_dtypes.generators import gen_sample_ohlcv


@transform
def fetch_yahoo_finance_ohlcv(
    config: YahooFinanceConfig,
) -> ProviderBatchCollection:
    """yfinance から ticker 群を取得し、ProviderBatchCollection として返す。"""
    # 仮実装: 実際のAPI呼び出し部分は省略し、Exampleデータを返す
    tickers = config["tickers"]
    # 最初のティッカーのみ処理
    symbol = tickers[0]
    frame = gen_sample_ohlcv(n=48, start_price=150.0, seed=7)
    frame.reset_index(inplace=True)
    batch: ProviderOHLCVBatch = {
        "provider": MarketDataProvider.YAHOO,
        "symbol": symbol,
        "frame": frame,
        "frequency": config["frequency"],
    }
    return {"provider": MarketDataProvider.YAHOO, "batches": [batch]}


@transform
def fetch_ccxt_ohlcv(
    config: CCXTConfig,
) -> ProviderBatchCollection:
    """CCXT 取引所から暗号資産の OHLCV を取得。"""
    # 仮実装: 実際のAPI呼び出し部分は省略し、Exampleデータを返す
    symbols = config["symbols"]
    # 最初のシンボルのみ処理
    symbol = symbols[0]
    frame = gen_sample_ohlcv(n=48, start_price=45000.0, seed=11)
    frame.reset_index(inplace=True)
    batch: ProviderOHLCVBatch = {
        "provider": MarketDataProvider.CCXT,
        "symbol": symbol,
        "frame": frame,
        "frequency": config["frequency"],
    }
    return {"provider": MarketDataProvider.CCXT, "batches": [batch]}


@transform
def normalize_multi_provider(
    *provider_batches: ProviderBatchCollection,
    target_frequency: Frequency = Frequency.HOUR_1,
    resample_method: str = "forward_fill",  # "forward_fill" or "upsample"
) -> NormalizedOHLCVBundle:
    """複数プロバイダの取得結果を統一スキーマへ変換。

    可変長引数で複数のプロバイダバッチを受け取り、内部で集約してから正規化する。
    provider と symbol の組が重複しないことを検証。
    """
    # 仮実装: Exampleデータを返す
    from algo_trade_dtypes.generators import gen_normalized_bundle

    return gen_normalized_bundle()


@transform
def merge_market_data_bundle(
    bundle: NormalizedOHLCVBundle,
    *,
    join_policy: str = "outer",
) -> MultiAssetOHLCVFrame:
    """正規化 DataFrame を MultiIndex DataFrame に変換。"""
    # 仮実装: Exampleデータを返す
    from algo_trade_dtypes.generators import gen_multiasset_frame

    return gen_multiasset_frame()


@transform
def persist_market_data_snapshot(
    multiasset_frame: MultiAssetOHLCVFrame,
    config: MarketDataIngestionConfig,
    *,
    base_dir: str = "output/data/snapshots",
) -> MarketDataSnapshotMeta:
    """Parquet へ書き出し、config から一意のパスを自動生成。"""
    # 仮実装: Exampleデータを返す
    from algo_trade_dtypes.generators import gen_snapshot_meta

    return gen_snapshot_meta()


@transform
def load_market_data(
    storage_path: str,
    *,
    format: str = "auto",
) -> MultiAssetOHLCVFrame:
    """ファイルパスから市場データを読み込み。"""
    # 仮実装: Exampleデータを返す
    from algo_trade_dtypes.generators import gen_multiasset_frame

    return gen_multiasset_frame()
