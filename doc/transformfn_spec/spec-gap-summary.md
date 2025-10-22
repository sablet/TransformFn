---
title: "TransformFn 仕様との差分サマリー"
description: "doc/transformfn_spec 下の各仕様と現状実装の乖離ポイントまとめ"
---

# TransformFn 仕様との差分サマリー

各仕様ドキュメント（`doc/transformfn_spec/` 配下）と `apps/algo-trade` 実装コードを突き合わせ、優先度が高いギャップを整理した。

## Phase 1: Market Data Ingestion

### normalize_multi_provider（`apps/algo-trade/algo_trade_transforms/market_data.py:100`）
- **仕様期待**: ProviderBatchCollection の正規化・重複検証・タイムゾーン統一を行う。
- **現状**: Example ジェネレータを返すスタブで、入力検証がゼロ。
- **影響**: 以降の統合・永続化フェーズが全てサンプルデータ依存。
- **対応メモ**: `normalize_provider_batch` 等の既存ヘルパーを呼ぶ実装へ差し替え、検証失敗時は `TransformError`。

### merge_market_data_bundle（`apps/algo-trade/algo_trade_transforms/market_data.py:120`）
- **仕様期待**: 正規化済みデータを MultiIndex DataFrame（symbol/date）へ集約。
- **現状**: スタブで Example DataFrame を返すのみ。
- **影響**: 複数プロバイダ統合の検証が未着手。
- **対応メモ**: `concat`・`pivot` と欠損補完ロジックを実装し、dtype を統一。

### persist_market_data_snapshot（`apps/algo-trade/algo_trade_transforms/market_data.py:131`）
- **仕様期待**: `MarketDataIngestionConfig` からパス生成し Parquet 永続化。
- **現状**: 引数を一切使わず Example を返す。
- **影響**: 再現性・キャッシュ機構が成立しない。
- **対応メモ**: `base_dir` と `config` からファイル名を決定し、`pathlib.Path` で保存処理を実装。

### load_market_data（`apps/algo-trade/algo_trade_transforms/market_data.py:148`）
- **仕様期待**: 永続化済みスナップショットからフォーマット検知→読み込み。
- **現状**: Example データを返却。
- **影響**: 実データでのパイプライン検証が不可能。
- **対応メモ**: スナップショットメタを参照し Parquet/CSV 読み込みを実装。

参照: `doc/transformfn_spec/algo-trade-market-data-ingestion-spec.md:512`

## Phase 2: Feature Engineering

### calculate_future_return（`apps/algo-trade/algo_trade_transforms/transforms.py:305`）
- **仕様期待**: `future_return_{forward}` 形式の列名を生成し、複数 forward 期間を共存させる。
- **現状**: 常に `"target"` 列へ上書きし、パラメータ情報が欠落。
- **影響**: `("USDJPY", "future_return", 5)` のような仕様前提の参照が出来ない。
- **対応メモ**: シンボル＋ forward を列名へ埋め込み、互換用 `"target"` 列は任意で維持。

### select_features / extract_target / clean_and_align（`apps/algo-trade/algo_trade_transforms/transforms.py:372,416,488`）
- **仕様期待**: いずれもヘルパー関数扱い（@transform 非適用）。
- **現状**: 3 関数とも @transform が付与され DAG ノード化。
- **影響**: 技術的ユーティリティまで Transform 化され、監査ノイズと不要なキャッシュが増大。
- **対応メモ**: @transform を外し、Transform 内部での再利用に留める。

参照: `doc/transformfn_spec/algo-trade-phase2-feature-engineering-spec.md:682`

## Phase 3: Training & Prediction

### aggregate_cv_results（`apps/algo-trade/algo_trade_transforms/training.py:217`）
- **仕様期待**: fold ごとの `oos_actuals` と index を連結し、`extract_predictions` へ引き渡す。
- **現状**: `mean_score` / `std_score` を返す構造へ変更され、実測値配列が欠落。
- **影響**: Phase 3→4 のデータ受け渡しが途切れ、評価フローが成立しない。
- **対応メモ**: 仕様フィールドを復元しつつ統計値は付加情報として保持。

参照: `doc/transformfn_spec/algo-trade-phase3-training-prediction-spec.md:503`

## Phase 4: Simulation & Evaluation

### SimulationResult 型（`apps/algo-trade/algo_trade_dtypes/types.py:264`, `apps/algo-trade/algo_trade_transforms/simulation.py:312`）
- **仕様期待**: `pd.DataFrame` を返し、`date`/`portfolio_return`/`n_positions` 列を持つ。
- **現状**: `TypedDict` でリストを返す構造。DataFrame API と互換性なし。
- **影響**: 評価系 Transform が DataFrame 前提のままでは動作しない。
- **対応メモ**: RegisteredType を DataFrame 指定へ切り替え、Check も DataFrame ベースに更新。

### calculate_trading_costs（`apps/algo-trade/algo_trade_transforms/simulation.py:210`）
- **仕様期待**: `TradingCostConfig`・`SwapDataSource`・`ohlcv_frame` を使いスワップ／スプレッドを計算。
- **現状**: `NotImplementedError` を即返却し、実引数を利用しない。
- **影響**: 取引コスト考慮パイプラインが停止。
- **対応メモ**: `trading_cost_implementation.md` のステップに沿って実装、FRED キャッシュと BID/ASK 処理を追加。

### rank_predictions（`apps/algo-trade/algo_trade_transforms/simulation.py:37`）
- **仕様期待**: どの `method` でも `prediction_rank_pct` が 0-1 の `RankPercent` を維持。
- **現状**: `"ordinal"` / `"zscore"` が 0-1 範囲外の値を返す。
- **影響**: `RankPercent` チェックが破綻し、後続ロジックが期待前提で動かない。
- **対応メモ**: 0-1 正規化列を別途生成するか、RankPercent 用フィールドを分離。

参照: `doc/transformfn_spec/algo-trade-phase4-simulation-spec.md:598`

---

このサマリーをベースに、該当 Transform／ユーティリティ関数の修正計画を検討すること。
