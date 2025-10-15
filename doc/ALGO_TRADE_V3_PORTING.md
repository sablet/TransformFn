# algo_trade_v3 移植調査メモ

TransformFn の `datatype` / `transformer` へ段階的に移植する際に、`algo_trade_v3` 配下の 3 つのサブプロジェクトから再利用できそうな構造・処理を整理した。データ読み込み → 前処理 → 特徴量・ターゲット生成 → 学習と予測 → 評価の流れに沿って候補をまとめる。

## 移植ポリシー
- **IO 主体の処理は TransformFn へ移植しない**: `algo_trade_v3` でのファイル操作はほぼキャッシュ目的であり、TransformFn レポジトリ側には透過的なキャッシュ機構を別途実装予定。IO やキャッシュ管理そのものは `apps` 層へ持ち込まず、必要に応じて「中間データの型」だけ `datatype` として再定義する。
- **Selenium を用いた取得処理は今回対象外**: `ohlcv_loader` の GMO クリック証券ダウンロードは環境依存性が高いため、当面は TransformFn DAG の外に置く。
- **pyarrow / parquet 書き出しは TransformFn 側キャッシュで代替**: 既存コードの parquet 生成はキャッシュ用途。TransformFn では公式キャッシュで同等機能を提供し、必要なら parquet 相当の型定義のみ残す。
- **LightGBM は素直に対応**: 学習データが小規模であれば計算負荷は問題にならないため、`train_predict_server` の学習・推論ロジックはシンプルに移植する前提とする。
- **評価・シミュレーションは `experiments/` を参照**: `@algo_trade_v3/train_predict_server/experiments/` にある複数通貨（インデックス指数を含む）から最大期待収益の通貨を選んで BUY するシミュレーションなどを再現できるよう、評価関数やパイプライン設計に注意する。
- **Example / Check は軽量に維持**: TransformFn での `ExampleValue` や `Check` は最小限のデータ提示と値域確認に留め、`algo_trade_v3` 由来の重い再計算を持ち込まない。複雑な検証は Transform 本体やテストで扱う。
- `xform-auditor` CLI は、型注釈のみから **入力生成 → 関数実行 → 出力検証** を自動化します。以下のガイドに従うことで、**pytest を最小限に抑え**、型注釈ベースのテストで品質を担保する


## 調査スコープ
- `algo_trade_v3/ohlcv_loader`
- `algo_trade_v3/ohlcv_preprocessor`
- `algo_trade_v3/train_predict_server`

---

## 1. データ読み込み (`ohlcv_loader`)
### datatype 候補
- `algo_trade_v3/ohlcv_loader/schema.py`
  - `OHLCVDataRequest`: 取得対象通貨・保存先・リトライなどを保持する要求スキーマ。TransformFn では入力 `Annotated` の `ExampleValue` に対応させやすく、キャッシュキー生成にも流用可能。
  - `OHLCVDataResponse`: 取得結果（成功/失敗通貨、処理時間、警告）を集約。返り値 TypedDict のたたき台になる。
  - `DirectoryConfig` / `DownloadConfig` / `AuthConfig`: 取得パラメータを分離した補助設定。TransformFn `datatype` として再定義し CLI から渡しやすくする。
  - `DataProvider`, `CurrencyPair`, `CurrencyIndex`: 列挙型。`ExampleType` での制約に使える。
  - `validate_currency_list`, `validate_directory_path` などのバリデータ: TransformFn 側では `datatype` に付随する検証ロジックとして活用。

### transformer 候補
- **IO 付き関数は移植対象外**: `get_ohlcv_data_paths`, `OHLCVDataService.process_request`, `_download_gmo_data`, `_unzip_files`, `_save_dataframes`, `_load_ohlc_df`, `_collect_currency_paths` は副作用の大半がキャッシュ目的のため、そのまま TransformFn へは持ち込まない。必要なら `apps` 層外部のユーティリティとして残し、TransformFn には取得済みデータを渡す想定。
- `currency_index_calculator.py` の純粋計算関数群（例: `calc_currency_index`, `create_currency_index_ohlc`）は、入力 DataFrame さえあればファイル出力なしで利用できるため TransformFn `transformer` 候補。

### 移植メモ
- `OHLCVDataResponse` など Path 情報を持つ型は、TransformFn ではファイル生成を伴わない形にリファイン（例: データフレーム ID や統計量のみにする）。
- Selenium ベースのダウンロードを扱う場合は別モジュールとしてラップし、TransformFn DAG からは切り離す。

---

## 2. 前処理 (`ohlcv_preprocessor/src`)
### datatype 候補
- `algo_trade_v3/ohlcv_preprocessor/src/schema.py`
  - `FXDataSchema`: 必須/任意カラムと型検証をまとめたスキーマ。`DataValidationResult` とセットで TransformFn の入力検証へ転用。
  - `DataValidationResult`: バリデーション結果の Pydantic モデル。`Check` 側での利用も想定。
  - `Frequency`, `Currency`, `ConvertType`, `CyclicParameter`: 列挙型。`Annotated` メタ情報に組み込み可能。
  - `DataSpec`, `TargetSetting`, `OutputConfig`, `FeatureGenerationRequest`: 特徴量生成の設定モデル。TransformFn DAG パラメータのベース。

### transformer 候補
- **IO を伴う関数は除外**: `load_currency_data_from_dir`, `get_resampled_ohlcv` などファイル読み込み＋キャッシュ前提の処理は TransformFn では利用しない。代わりに、既に手元にある DataFrame を受け取り処理する純粋関数を抜き出す。
- `resample_ohlcv` / `resample_ohlcv_df`: DataFrame を受け取ってリサンプリングする純粋処理。TransformFn に適合しやすい。
- `validate_input_data_df` / `validate_input_data`: DataFrame / ファイルバリデーション。ファイル読み込み部分を切り離し、DataFrame 版を TransformFn に取り込む。
- `get_currency_index_info`: インデックス構成情報を返す純粋計算。`datatype` / `transformer` 両面で活用可能。
- `clean_data_and_align`: 特徴量・ターゲットを共通インデックスに揃え統計を返す。複数入力を束ねる TransformFn として重要。

### 移植メモ
- グローバルキャッシュ `_ohlcv_cache` / `_currency_index_cache` への依存は排除し、TransformFn のキャッシュ機構へ置き換える。
- リサンプリングで parquet 書き出しは行わず、DataFrame をそのまま戻す形にする。

---

## 3. 特徴量 + ターゲット生成 (`ohlcv_preprocessor/src/service.py`)
### datatype 候補
- `DataSpec.feature_name`: 特徴量命名規則をカプセル化。TransformFn 設計時に列名仕様を一元管理。
- `GenerationResult`: 出力情報を保持。TransformFn ではファイルパスを外し、生成済みデータ ID やメタ情報に寄せる。

### transformer 候補
- テクニカル指標・派生値計算群（`calculate_rsi`, `calculate_adx`, `calculate_recent_return`, `calculate_future_return`, `calculate_cyclic_feature`, `generate_spread`, `generate_swap_rate`）。いずれも純粋計算なので TransformFn の小さな `transformer` として直接移植可能。
- `generate_target`: `ConvertType` に応じてターゲット/方向ラベルを生成。
- `clean_data_and_align`: 前処理と同様に重要な結合ステップ。
- `create_data`, `generate_features_with_cache`, `process_feature_generation_request`: これらはファイル IO + キャッシュを伴うラッパーのため TransformFn へは直接移植しない。内包されている純粋計算部分を分解して `transformer` 化する。

### 移植メモ
- `FeatureCacheManager` 依存部分は TransformFn キャッシュへ移行し不要化。
- 戻り値は DataFrame や Series ではなく TypedDict + `Check` を組み合わせた構造で表現する想定。

---

## 4. 学習と予測 (`train_predict_server/src`)
### datatype 候補
- `algo_trade_v3/train_predict_server/src/schema.py`
  - `TrainPredictRequest`: データパス・CV・LGBM 設定まとめ。TransformFn ではファイルパスを直接扱わず、データ ID や構造を受け取る設定モデルにリファイン。
  - `SimpleCVConfig`, `SimpleLGBMParams`, `SimpleOutputConfig`: デフォルトを保持した設定モデル。`ExampleValue` を付けやすい。
  - `DataPaths`: TransformFn ではファイル実体を扱わないため、データフェッチ用の識別子型に読み替える想定。
  - `PredictionResult`, `TrainPredictResponse`, `ValidationResult`: 学習結果の構造を表すモデル。戻り値 TypedDict のベース。

### transformer 候補
- `convert_nullable_dtypes`: pandas の nullable dtype を通常 dtype へ変換する純粋処理。そのまま移植。
- `TimeSeriesCrossValidator.get_cv_splits` および `_time_series_split` / `_expanding_window_split` / `_sliding_window_split`: インデックス配列を返す純粋計算。TransformFn 上で再利用可能。
- `LightGBMTrainer.train_fold`: Fold 単位の学習・評価（RMSE）。モデル保存は行わず、`Booster` とスコア/重要度を戻すように調整する。
- `TrainPredictService._execute_cross_validation`: 特徴量・ターゲットを受けて CV を実行し、OOS 予測と fold 結果を返す。ファイル書き出し `_save_oos_predictions` は削除し、DataFrame/TypedDict で返す。
- `TrainPredictService.process_train_predict_request`: IO を排除し、事前に提供されたデータ構造をもとに学習 → 評価 → シミュレーションに繋がる高レベル TransformFn へ改修。

### 移植メモ
- 学習結果を `output/` に保存する代わりに、TransformFn のキャッシュ/監査機能へ記録。
- `_calculate_score` は RMSE を返す。必要に応じて目的関数別に切り替え可能な評価モジュールへ昇格。

---

## 5. 評価・シミュレーション
### datatype / transformer 候補
- `ValidationResult` (`train_predict_server/src/schema.py`): 欠損数・相関などデータ品質指標。`Check` で活用。
- `DataValidator.validate_dataframe` / `.analyze_feature_target_correlation`: 特徴量・ターゲットの品質検証。TransformFn で評価ノード化。
- `TrainPredictResponse.mean_cv_score` / `.std_cv_score`: CV 結果のサマリ計算。`Check` 用の薄いラッパーとして利用。
- `reversal_modeling_validation.py` の評価ユーティリティ（`create_future_returns`, `create_lagged_gaps` など）: シミュレーション用途の純粋計算。TransformFn の評価 DAG へ組み込み、`experiments/` にある「複数通貨から最大期待収益の通貨を BUY,SELL する」シナリオを再現できるようにする。
- シミュレーション系 Transform は、複数通貨・通貨インデックスを入力とし、予測結果をランキング → 上位買い付け → リターン計算 のステップに分解して設計する。

### 移植メモ
- 評価指標（R², RMSE, MAE, Spearman, Kendall など）は `reversal_modeling_validation.py` の dataclass を参考に TypedDict 化。
- シミュレーション結果には、対象通貨、購入タイミング、想定リターン、評価ウィンドウなどを含める。

---

## 横断的な考慮事項
- **キャッシュと副作用の分離**: 既存コードが担っていたキャッシュ/ファイル保存は TransformFn の公式キャッシュへ集約し、`transformer` は純粋計算（あるいは最小限の副作用）に留める。
- **入出力ディレクトリ**: どうしてもファイルを扱う必要がある場合は `output/` 配下に限定し、TransformFn の監査機能と整合させる。
- **外部依存**: `selenium` は対象外、`pyarrow` は TransformFn キャッシュで代替、`lightgbm` は軽量データ前提でそのまま利用。
- **型付け**: Pydantic / Enum を `datatype` に落とし、`Annotated` へ `ExampleType` / `Check` を付与。評価関数は `Check` として組み込む。
- **評価パイプラインの拡張性**: `experiments/` のシミュレーションケースを想定し、複数通貨・通貨インデックスを同時に扱える DAG を構築する。BUY シミュレーションの戻り値は、取引履歴・P/L・ドローダウン等を含む構造にする。

この整理をベースに、TransformFn の DAG 設計では「副作用を持たない純粋計算ノード」と「評価/シミュレーションノード」を中心に据え、IO 依存はリポジトリ外の仕組みへ切り出す方針を徹底する。EOF

## 完了条件
- `make check` が成功する
- `uv run python -m xform_auditor apps/pipeline-app/pipeline_app` を実行し、移植した Transform 群に対する監査が `OK` となる。
- `uv run python -m xform_auditor apps/pipeline-app/pipeline_app --format json` の結果にシミュレーション Transform の評価が含まれ、BUY シナリオに対する `Check` が全て成功する
- 上記 `audit` コマンドの出力例（成功ログ）を `doc/` もしくは PR 説明に添付し、客観的に確認できる状態にする。
- `doc/` に移植範囲・評価パイプライン・制約が記載され、オンボーディング時に 30 分以内でサンプル DAG を監査・実行できる手順が明示されている。
