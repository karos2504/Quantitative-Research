import numpy as np
import pandas as pd
import yfinance as yf
import ATR

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

# Calculate ADX
def calculate_adx(df, period=20):
    df = df.copy()
    atr = ATR.calculate_atr(df, period)
    up_move = df['High'].diff()
    down_move = df['Low'].diff()
    
    # Apply parentheses around conditions for logical AND
    positive_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    negative_dm = np.where((up_move < down_move) & (down_move > 0), down_move, 0)
    
    positive_di = 100 * (positive_dm / atr).ewm(span=period, min_periods=period).mean()
    negative_di = 100 * (negative_dm / atr).ewm(span=period, min_periods=period).mean()

    # Calculate ADX
    adx = 100 * abs((positive_di - negative_di) / (positive_di + negative_di)).ewm(span=period, min_periods=period).mean()
    df['ADX'] = adx.iloc[:, 0]
    return df

# Main execution
def main():
    ohlcv_data = fetch_ohlcv_data(TICKERS, PERIOD, INTERVAL)

    for ticker in TICKERS:
        ohlcv_data[ticker] = calculate_adx(ohlcv_data[ticker])

    # Display last few rows for AAPL
    print(ohlcv_data['AAPL'].tail())

if __name__ == '__main__':
    main()
