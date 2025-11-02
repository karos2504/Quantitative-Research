import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

# Fetch stock data
def fetch_data(ticker, start, end):
    data = yf.download(tickers=ticker, start=start, end=end, auto_adjust=False)
    return data[['Adj Close']]

# Calculate slopes of the normalized data over the last 'n' days
def calculate_slope(ser, n):
    slopes = []  # Initialize slopes as an empty list
    for i in range(n, len(ser)):
        y = ser['Adj Close'].iloc[i - n:i]  # Get the last 'n' closing prices
        x = np.arange(n)  # Create an array of indices for the 'n' days
        y_scaled = (y - y.min()) / (y.max() - y.min())  # Normalize the 'y' values
        x_scaled = (x - x.min()) / (x.max() - x.min())  # Normalize the 'x' values
        x_scaled = sm.add_constant(x_scaled)  # Add constant to 'x' for OLS
        model = sm.OLS(y_scaled, x_scaled)  # Fit the OLS model
        results = model.fit()  # Get the results of the regression
        slopes.append(results.params.iloc[-1])  # Append the slope (gradient) value
    
    # Convert slope to angle in degrees
    slopes_angle = np.rad2deg(np.arctan(np.array(slopes)))
    
    # Align slopes with the correct dates
    slope_dates = ser.index[n:]  # Dates start from the n-th data point
    slope_df = pd.DataFrame(slopes_angle, index=slope_dates, columns=['Slope'])

    # Merge slope values directly into the original dataframe
    ser['Slope'] = slope_df['Slope']  # Directly assign slope values to the dataframe
    return ser

# Calculate slopes using sklearn's LinearRegression
def calculate_slope_sklearn(df, n):
    slopes = []
    for i in range(n, len(df)):
        y = df['Adj Close'].iloc[i - n:i].values.reshape(-1, 1)  # Reshape for sklearn
        x = np.arange(n).reshape(-1, 1)  # Reshape for sklearn
        model = LinearRegression().fit(x, y)
        slope = model.coef_[0][0]  # Extract the slope (coefficient)
        slopes.append(np.degrees(np.arctan(slope)))  # Convert slope to angle in degrees
    slope_df = pd.DataFrame(slopes, index=df.index[n:], columns=['Slope'])
    df['Slope'] = slope_df['Slope']
    return df

# Plot the slope as an angle over time
def plot_slope(df):
    # Plot the slope (angle) over time using the 'Slope' column from df
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Slope'], label='Slope Angle', color='b')
    plt.title('Slope of Stock Price (Angle over Time)')
    plt.xlabel('Date')
    plt.ylabel('Slope (Degrees)')
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()

def main():
    ticker = 'AAPL'
    start = dt.datetime.today() - dt.timedelta(days=365)  # One year of data
    end = dt.datetime.today()
    n = 5  # Number of days to calculate slope over

    # Fetch stock data
    stock_data = fetch_data(ticker=ticker, start=start, end=end)

    # Calculate the slope (angle)
    # stock_data_with_slope = calculate_slope(stock_data, n=n)

    # Calculate the slope (angle) use sklearn
    stock_data_with_slope = calculate_slope_sklearn(stock_data, n=n)

    # Plot the results
    plot_slope(stock_data_with_slope)

if __name__ == "__main__":
    main()
