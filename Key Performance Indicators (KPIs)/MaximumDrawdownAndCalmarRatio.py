import numpy as np
import pandas as pd
import yfinance as yf
from CAGR import calculate_cagr

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

# Calculate Maximum Drawdown
def calculate_maximum_drawdown(df):
    cum_max = df['Adj Close'].cummax()
    drawdown = (df['Adj Close'] - cum_max) / cum_max
    max_drawdown = drawdown.min()
    return max_drawdown.item()

# Calculate Calmar Ratio
def calculate_calmar_ratio(df, periods_per_year):
    cagr = calculate_cagr(df, periods_per_year)
    max_drawdown = abs(calculate_maximum_drawdown(df))
    calmar = cagr / max_drawdown
    return calmar

# Main execution
def main():
    ohlcv_data = fetch_ohlcv_data(TICKERS, PERIOD, INTERVAL)
    
    for ticker in TICKERS:
        df = ohlcv_data[ticker]
        cagr = calculate_cagr(df, NUM_OF_TRADING)
        mdd = calculate_maximum_drawdown(df)
        calmar = calculate_calmar_ratio(df, NUM_OF_TRADING)
        print(f"{ticker}: CAGR={cagr:.4f}, Max Drawdown={mdd:.4f}, Calmar Ratio={calmar:.4f}")

if __name__ == '__main__':
    main()
