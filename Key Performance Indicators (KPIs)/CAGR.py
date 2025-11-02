import numpy as np
import pandas as pd
import yfinance as yf

# Configuration
TICKERS = ['AAPL', 'MSFT', 'GOOG']
PERIOD = '7mo'
INTERVAL = '1d'
NUM_OF_TRADING = 252

# Fetch OHLCV data for each ticker
def fetch_ohlcv_data(tickers, period, interval):
    return {
        ticker: yf.download(ticker, period=period, interval=interval, auto_adjust=False).dropna()
        for ticker in tickers
    }

# Calculate CAGR
def calculate_cagr(df, periods_per_year):
    start = df['Adj Close'].iloc[0]
    end = df['Adj Close'].iloc[-1]
    years = len(df) / periods_per_year
    cagr = (end / start) ** (1 / years) - 1
    return cagr.item()

# Main execution
def main():
    ohlcv_data = fetch_ohlcv_data(TICKERS, PERIOD, INTERVAL)

    for ticker in TICKERS:
        print(f"\nData for {ticker}:")
        print(ohlcv_data[ticker].head())

    print('\nCompounded Annual Growth Rate (CAGR):')
    for ticker in TICKERS:
        cagr = calculate_cagr(ohlcv_data[ticker], NUM_OF_TRADING)
        print(f'CAGR of {ticker} = {cagr:.4f}')

if __name__ == '__main__':
    main()
