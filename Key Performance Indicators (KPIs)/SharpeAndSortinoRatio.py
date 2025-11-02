import numpy as np
import pandas as pd
import yfinance as yf
from CAGR import calculate_cagr
from VolatilityMeasures import calculate_volatility_returns

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

# Calculate Sharpe Ratio
def calculate_sharpe_ratio(df, rf):
    df = df.copy()
    sharpe_ratio = (calculate_cagr(df, periods_per_year=NUM_OF_TRADING) - rf) / calculate_volatility_returns(df, periods_per_year=NUM_OF_TRADING)
    return sharpe_ratio

# Calculate Sortino Ratio
def calculate_sortino_ratio(df, rf):
    returns = df['Adj Close'].pct_change().dropna()
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(NUM_OF_TRADING)
    sortino_ratio = (calculate_cagr(df, periods_per_year=NUM_OF_TRADING) - rf) / downside_std
    return sortino_ratio.item()

# Main execution
def main():
    ohlcv_data = fetch_ohlcv_data(TICKERS, PERIOD, INTERVAL)

    for ticker in TICKERS:
        df = ohlcv_data[ticker]
        sharpe_ratio = calculate_sharpe_ratio(ohlcv_data[ticker], rf=0.03)
        sortino_ratio = calculate_sortino_ratio(ohlcv_data[ticker], rf=0.03)
        print(f"{ticker}: Sharpe Ratio={sharpe_ratio:.4f}, Sortino Ratio={sortino_ratio:.4f}")

if __name__ == '__main__':
    main()
