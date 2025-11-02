import yfinance as yf
import pandas as pd

# Define tickers to process
tickers = ["AAPL", "MSFT"]

# Dictionary to store financial data for each ticker
financial_dir = {}

# Function to fetch financial data for each ticker
def fetch_financial_data(tickers):
    for ticker in tickers:
        print(f"Processing ticker: {ticker}")
        
        # Fetch the stock data
        stock = yf.Ticker(ticker)
        
        # Retrieve financial data (balance sheet, income statement, cash flow, and key stats)
        financial_data = {
            'balance_sheet': stock.balance_sheet,
            'income_statement': stock.financials,
            'cash_flow': stock.cashflow,
            'key_stats': stock.info
        }
        
        # Store data in dictionary
        financial_dir[ticker] = financial_data

# Fetch financial data for all tickers
fetch_financial_data(tickers)

# Function to construct DataFrames from the collected financial data
def create_financial_dataframes(financial_dir):
    # Concatenate data for each financial statement type across tickers
    df_balance_sheet = pd.concat({ticker: data['balance_sheet'] for ticker, data in financial_dir.items()}, axis=1)
    df_income_statement = pd.concat({ticker: data['income_statement'] for ticker, data in financial_dir.items()}, axis=1)
    df_cash_flow = pd.concat({ticker: data['cash_flow'] for ticker, data in financial_dir.items()}, axis=1)

    # Convert key stats to a DataFrame (if it's a dictionary)
    df_key_stats = pd.DataFrame({ticker: data['key_stats'] for ticker, data in financial_dir.items()}).T
    
    return df_balance_sheet, df_income_statement, df_cash_flow, df_key_stats

# Create the DataFrames
df_balance_sheet, df_income_statement, df_cash_flow, df_key_stats = create_financial_dataframes(financial_dir)

# Print the DataFrames for inspection
print("\nFinal Balance Sheets:")
print(df_balance_sheet)

print("\nFinal Income Statements:")
print(df_income_statement)

print("\nFinal Cash Flow Statements:")
print(df_cash_flow)

print("\nFinal Key Statistics:")
print(df_key_stats)
