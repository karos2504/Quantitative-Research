# Import necessary libraries
import yfinance as yf
import pandas as pd
import ATR
from stocktrends import Renko

# --- Configuration ---
TICKERS = ["AAPL", "MSFT", "GOOG"]
RENKO_PERIOD = '1mo'
RENKO_INTERVAL = '5m'
HOURLY_PERIOD = '1y'
HOURLY_INTERVAL = '1h'
ATR_CALC_PERIOD = 120  # Period for ATR calculation on hourly data

# --- Data Fetching ---
def fetch_ohlcv_data(tickers, period, interval):
    return {
        ticker: yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False
        ).dropna(how="any")
        for ticker in tickers
    }

def convert_to_renko(ohlc_df, hourly_df):
    # --- 1. Prepare ATR Input (hourly_df) ---
    hourly_data = hourly_df.copy()
    
    if isinstance(hourly_data.columns, pd.MultiIndex):
        hourly_data.columns = hourly_data.columns.droplevel(1) 
    
    # Calculate ATR using the Title Case columns
    atr_df = ATR.calculate_atr(hourly_data, period=ATR_CALC_PERIOD) 
    
    # ATR is calculated on hourly_data. We extract the last ATR value for brick size.
    # The ATR column name is likely ('ATR', ticker), so we access it that way.
    brick_size = 3 * round(atr_df['ATR'].iloc[-1], 0) 
    
    
    # --- 2. Prepare Renko Input (ohlc_df) ---
    df = ohlc_df.copy()
    
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1) 
        
    df.reset_index(inplace=True)
    date_col_name = 'Datetime' if 'Datetime' in df.columns else 'Date'
    
    # Rename columns to the exact **LOWERCASE** format required by stocktrends (Renko)
    df.rename(columns={
        date_col_name: "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Adj Close": "close",
        "Volume": "volume"
    }, inplace=True)
    
    # Select only the required columns for the Renko conversion
    required_cols = ["date", "open", "high", "low", "close", "volume"]
    df_renko_input = df[required_cols]

    # Create Renko object and get the Renko DataFrame
    renko_converter = Renko(df_renko_input)
    renko_converter.brick_size = brick_size
    
    return renko_converter.get_ohlc_data()

# --- Main Execution ---
def main():
    print("Fetching 5-minute data for Renko charts...")
    ohlcv_5m_data = fetch_ohlcv_data(TICKERS, RENKO_PERIOD, RENKO_INTERVAL)
    
    print("Fetching hourly data for ATR-based brick size...")
    ohlcv_1h_data = fetch_ohlcv_data(TICKERS, HOURLY_PERIOD, HOURLY_INTERVAL)
    
    print("\nConverting OHLCV data to Renko bricks for all tickers...")
    renko_data = {
        ticker: convert_to_renko(ohlcv_5m_data[ticker], ohlcv_1h_data[ticker])
        for ticker in TICKERS
    }
    
    print("\n--- Renko Conversion Complete ---")
    
    # Display last few rows for a sample ticker to verify the output
    sample_ticker = TICKERS[0]
    if sample_ticker in renko_data:
        print(f"\nSample Renko Data for {sample_ticker}:")
        print(renko_data[sample_ticker].tail())

if __name__ == "__main__":
    main()

