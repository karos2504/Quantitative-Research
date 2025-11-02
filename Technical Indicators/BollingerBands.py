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


# Calculate BB
def calculate_bollinger_bands(df, ticker, n=20, std_dev=2):
    df = df.copy()
    sma = df['Adj Close'].rolling(window=n).mean()
    std = df['Adj Close'].rolling(window=n).std(ddof=0)

    df[('Middle_Band', ticker)] = sma
    df[('Upper_Band', ticker)] = sma + (std_dev * std)
    df[('Lower_Band', ticker)] = sma - (std_dev * std)
    df[('Bollinger_Band_Width', ticker)] = df[('Upper_Band', ticker)] - df[('Lower_Band', ticker)]
    return df


# Main execution
def main():
    ohlcv_data = fetch_ohlcv_data(TICKERS, PERIOD, INTERVAL)

    for ticker in TICKERS:
        ohlcv_data[ticker] = calculate_bollinger_bands(ohlcv_data[ticker], ticker)

    # Display last few rows for AAPL
    print(ohlcv_data['AAPL'][['Adj Close', 'Upper_Band', 'Lower_Band']].tail())


if __name__ == '__main__':
    main()
