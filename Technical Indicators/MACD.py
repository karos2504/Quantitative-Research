import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
TICKERS = ['AAPL', 'MSFT', 'GOOG']
PERIOD = '1mo'
INTERVAL = '15m'

# Fetch OHLCV data for each ticker
def fetch_ohlcv_data(tickers, period, interval):
    ohlcv_data = {}
    for ticker in tickers:
        print(f"Fetching {ticker}...")
        data = yf.download(ticker, period=period, interval=interval, auto_adjust=False).dropna()
        ohlcv_data[ticker] = data
    return ohlcv_data

# Calculate MACD and Signal Line
def calculate_macd(df, fast=12, slow=26, signal=9):
    df = df.copy()
    ema_fast = df['Adj Close'].ewm(span=fast, min_periods=fast).mean()
    ema_slow = df['Adj Close'].ewm(span=slow, min_periods=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, min_periods=signal).mean()
    hist = macd - signal_line

    df['MACD'] = macd
    df['Signal'] = signal_line
    df['Hist'] = hist
    return df

# Plot MACD, Signal, and Histogram
def plot_macd(df, ticker):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['MACD'], label='MACD', color='blue')
    plt.plot(df.index, df['Signal'], label='Signal', color='red')
    plt.bar(df.index, df['Hist'], label='Histogram', color='gray', alpha=0.4)
    plt.title(f"{ticker} MACD (15m Interval)")
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main execution
def main():
    ohlcv_data = fetch_ohlcv_data(TICKERS, PERIOD, INTERVAL)

    for ticker in TICKERS:
        ohlcv_data[ticker] = calculate_macd(ohlcv_data[ticker], ticker)
        print(f"\n{ticker} Latest Data:")
        print(ohlcv_data[ticker][['Adj Close', 'MACD', 'Signal', 'Hist']].tail())
        plot_macd(ohlcv_data[ticker], ticker)

if __name__ == '__main__':
    main()
