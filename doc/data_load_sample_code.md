##

ccxt, binance

```
import ccxt
import pandas as pd
import time

# Binance インスタンスを生成（認証不要）
exchange = ccxt.binance()

# 通貨ペアと時間足を指定
symbol = 'BTC/USDT'
timeframe = '1d'  # 1d=日足, 1h=1時間足, 15m=15分足 など

# 取得範囲の指定（例：2017年から現在まで）
since = exchange.parse8601('2017-01-01T00:00:00Z')

# OHLCV データを格納するリスト
all_candles = []
limit = 1000  # 一度に取得できる最大数（取引所制限）

while True:
    # ヒストリカルデータ取得
    candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
    
    if not candles:
        break  # データがなくなったら終了
    
    all_candles += candles
    
    # 次回の取得開始時間を更新
    since = candles[-1][0] + 1  # 最後のデータの次から開始
    
    # API制限対策で少し待機
    time.sleep(exchange.rateLimit / 1000)

# pandas DataFrame に変換
df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

# 結果を表示
print(df.head())
print(f"Total candles: {len(df)}")

# CSVに保存
df.to_csv('binance_btcusdt_daily.csv', index=False)
```


```sample code
      timestamp     open     high      low    close       volume   datetime
0  1502928000000  4261.48  4485.39  4200.74  4285.08   795.150377 2017-08-17
1  1503014400000  4285.08  4371.52  3938.77  4108.37  1199.888264 2017-08-18
```


AAPL + USDJPY

```
import yfinance as yf
import pandas as pd

# 取得対象を定義
tickers = ['AAPL', 'JPY=X']  # "JPY=X" が USD/JPY のYahoo!Finance表記

# 期間とデータ頻度を指定
start_date = '2015-01-01'
end_date = '2025-01-01'
interval = '1d'  # 1d=日足, 1h=1時間足, 1wk=週足, 1mo=月足 など

# yfinanceで複数ティッカーを一括取得
data = yf.download(tickers, start=start_date, end=end_date, interval=interval)

# カラム整理（Closeだけを抽出）
close_prices = data['Close']

# 結果を表示
print(close_prices.head())
print(f"取得データ件数: {len(close_prices)}")

# CSV出力
close_prices.to_csv('aapl_usdjpy_daily.csv')

```

```sample data
Ticker           AAPL       JPY=X
Date                             
2015-01-01        NaN  119.672997
2015-01-02  24.261045  119.870003
2015-01-05  23.577578  120.433998
```
