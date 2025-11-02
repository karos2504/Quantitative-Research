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

# Calculate Volatility
def calculate_volatility_returns(df, periods_per_year):
    df = df.copy()
    returns = df['Adj Close'].pct_change()
    volatility = returns.std() * np.sqrt(periods_per_year)
    return volatility.item()

# Main execution
def main():
    ohlcv_data = fetch_ohlcv_data(TICKERS, PERIOD, INTERVAL)
    print('\nVolatility Returns:')
    for ticker in TICKERS:
        volatility = calculate_volatility_returns(ohlcv_data[ticker], periods_per_year=NUM_OF_TRADING)
        print(f'Volatility of {ticker} = {volatility:.4f}')

if __name__ == '__main__':
    main()
