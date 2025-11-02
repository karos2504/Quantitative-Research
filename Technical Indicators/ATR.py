import numpy as np
import pandas as pd
import yfinance as yf

# Configuration
TICKERS = ['AAPL', 'MSFT', 'GOOG']
PERIOD = '1mo'
INTERVAL = '5m'

# Fetch OHLCV data for each ticker
def fetch_ohlcv_data(tickers, period, interval):
    return {
        ticker: yf.download(ticker, period=period, interval=interval, auto_adjust=False).dropna()
        for ticker in tickers
    }

# Calculate ATR
def calculate_atr(df, period=14):
    df = df.copy()
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    
    df['ATR'] = tr.ewm(span=period, adjust=False).mean()
    return df

# Main
def main():
    ohlcv_data = fetch_ohlcv_data(TICKERS, PERIOD, INTERVAL)
    for ticker in TICKERS:
        ohlcv_data[ticker] = calculate_atr(ohlcv_data[ticker], ticker, period=14)
    print(ohlcv_data['AAPL'][['Close', 'ATR']].tail())

if __name__ == '__main__':
    main()
