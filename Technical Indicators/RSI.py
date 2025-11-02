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

# Calculate RSI
def calculate_rsi(df, ticker, period=14):
    delta = df['Adj Close'].diff().squeeze()

    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    gain_series = pd.Series(gain, index=df.index)
    loss_series = pd.Series(loss, index=df.index)

    avg_gain = gain_series.ewm(span=period, min_periods=period).mean()
    avg_loss = loss_series.ewm(span=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    df[('RSI', ticker)] = rsi
    return df

# Main execution
def main():
    ohlcv_data = fetch_ohlcv_data(TICKERS, PERIOD, INTERVAL)

    for ticker in TICKERS:
        ohlcv_data[ticker] = calculate_rsi(ohlcv_data[ticker], ticker)

    # Display last few rows for AAPL
    print(ohlcv_data['AAPL'].tail())

if __name__ == '__main__':
    main()
