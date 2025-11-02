import pandas as pd
import numpy as np
import yfinance as yf
import datetime

def fetch_data(ticker, start, end):
    # Download stock data including volume and adjusted close
    data = yf.download(tickers=ticker, start=start, end=end, auto_adjust=False)
    return data[['Adj Close', 'Volume']]

def calculate_obv(df, ticker):
    df = df.copy()
    # Calculate the daily returns
    returns = df['Adj Close'].pct_change()
    
    # Assign 1 for positive returns and -1 for negative returns
    direction = np.where(returns >= 0, 1, -1)
    direction[0] = 0
    
    # Adjust volume based on price movement direction
    vol_adj = df['Volume'] * direction
    
    # Calculate OBV (On-Balance Volume)
    obv = vol_adj.cumsum()

    df[('OBV', ticker)] = obv
    return df

def main():
    ticker = 'AAPL'
    start = datetime.date.today() - datetime.timedelta(days=365)
    end = datetime.date.today()

    # Fetch data
    stock_data = fetch_data(ticker, start, end)
    
    # Calculate OBV
    obv = calculate_obv(stock_data, ticker)

    # Output the OBV
    print(obv)

if __name__ == '__main__':
    main()
