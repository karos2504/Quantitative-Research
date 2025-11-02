import numpy as np
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
import datetime as dt
import copy
import time 
import matplotlib.pyplot as plt


# ---------------------------- CONFIGURATION ---------------------------- #
TICKERS = ["MSFT", "AAPL", "FB", "AMZN", "INTC", "CSCO", "VZ", "IBM", "TSLA", "AMD"]
# Alpha Vantage offers 'full' data which can be extensive.
# Start date is no longer strictly controlled by yfinance limits.
# We'll keep the interval as '5min'.
INTERVAL = '5min' 
# Placeholder for your key path
ALPHA_VANTAGE_KEY_PATH = "/Users/karos/Documents/Alpha Vantage/key.txt" 
# Trading periods per year remains the same for 5-min intervals
PERIODS_PER_YEAR = 252 * 78 
RISK_FREE_RATE = 0.025


# ---------------------------- KPIs ---------------------------- #
def calculate_atr(df, n):
    """Function to calculate Average True Range (ATR)"""
    df_ = df.copy()
    df_['H-L'] = abs(df_['High'] - df_['Low'])
    df_['H-PC'] = abs(df_['High'] - df_['Close'].shift(1))
    df_['L-PC'] = abs(df_['Low'] - df_['Close'].shift(1))
    df_['TR'] = df_[['H-L', 'H-PC', 'L-PC']].max(axis=1, skipna=False)
    df_['ATR'] = df_['TR'].rolling(n).mean()
    return df_['ATR']

def calculate_cagr(returns_series, periods_per_year):
    """
    Calculate CAGR given a series of periodic returns.
    """
    cumulative_return = (1 + returns_series).prod()
    years = len(returns_series) / periods_per_year 
    if years == 0: return 0.0
    cagr = cumulative_return ** (1 / years) - 1
    return cagr

def calculate_volatility(returns_series, periods_per_year):
    """
    Annualized volatility from a returns series.
    """
    return returns_series.std() * np.sqrt(periods_per_year)

def calculate_sharpe_ratio(returns_series, risk_free_rate, periods_per_year):
    """
    Annualized Sharpe ratio from a returns series.
    """
    cagr = calculate_cagr(returns_series, periods_per_year)
    vol = calculate_volatility(returns_series, periods_per_year)
    return (cagr - risk_free_rate) / vol if vol != 0 else np.nan

def calculate_maximum_drawdown(returns_series):
    """
    Maximum drawdown from a returns series.
    """
    cumulative = (1 + returns_series).cumprod()
    peak = cumulative.cummax()
    drawdown_pct = (peak - cumulative) / peak
    return drawdown_pct.max()


# ---------------------------- DATA FETCHING (ALPHA VANTAGE) ---------------------------- #
def fetch_ohlcv_data_av(tickers, key_path, interval):
    """Fetch OHLCV data for given tickers using Alpha Vantage, respecting rate limits."""
    ohlc_intraday = {}
    api_call_count = 0
    # Initialize TimeSeries instance
    try:
        ts = TimeSeries(key=open(key_path,'r').read(), output_format='pandas')
    except Exception as e:
        print(f"Error reading API key or initializing TimeSeries: {e}")
        return {}
        
    start_time = time.time()
    
    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        try:
            # Alpha Vantage call for intraday data
            # outputsize='full' fetches the maximum available data (usually 1 month for 5min free)
            data_tuple = ts.get_intraday(symbol=ticker, interval=interval, outputsize='full')
            
            # The result is a tuple: (DataFrame, Metadata). We need the DataFrame [0]
            data = data_tuple[0]
            
            # The original Alpha Vantage data columns need renaming and reversal:
            data.columns = ["Open", "High", "Low", "Close", "Volume"]
            data = data.iloc[::-1]  # Reverse index (Alpha Vantage returns latest first)
            
            # Filter trading hours (9:35 AM to 4:00 PM EST/NYC time)
            data = data.between_time('09:35', '16:00')
            
            ohlc_intraday[ticker] = data.dropna()
            
        except Exception as e:
            print(f"Could not fetch data for {ticker} from Alpha Vantage: {e}")
            continue
            
        api_call_count += 1
        
        # Rate Limiting: Free API is 5 calls per minute (or 500 per day).
        # We enforce a wait after every 5 calls to ensure compliance.
        if api_call_count == 5:
            api_call_count = 0
            # Wait until 60 seconds have passed since the first API call of the batch
            wait_time = 60 - ((time.time() - start_time) % 60.0)
            if wait_time > 0:
                print(f"Rate limit reached. Waiting for {wait_time:.1f} seconds...")
                time.sleep(wait_time)
            start_time = time.time() # Reset start time for the next batch

    return ohlc_intraday


# ---------------------------- STRATEGY LOGIC ---------------------------- #
def run_breakout_strategy(ohlc_dict, atr_period=20, roll_period=20, vol_factor=1.5):
    """
    Implements the Intraday Resistance Breakout Strategy.
    """
    ohlc_dict_ = copy.deepcopy(ohlc_dict)
    tickers = list(ohlc_dict_.keys())
    tickers_ret = {}
    
    # Pre-calculate indicators
    for ticker in tickers:
        df = ohlc_dict_[ticker]
        df["ATR"] = calculate_atr(df, atr_period)
        df["roll_max_cp"] = df["High"].rolling(roll_period).max().shift(1) # max of previous 20 periods
        df["roll_min_cp"] = df["Low"].rolling(roll_period).min().shift(1)  # min of previous 20 periods
        df["roll_max_vol"] = df["Volume"].rolling(roll_period).max().shift(1) # max vol of previous 20 periods
        df.dropna(inplace=True)
        tickers_ret[ticker] = [0] * len(df) # Initialize returns
    
    # Backtesting loop
    for ticker in tickers:
        df = ohlc_dict_[ticker]
        tickers_signal = "" # Signal for the *next* period
        
        for i in range(1, len(df)):
            # Use .item() to ensure scalar extraction
            current_close = df["Close"].iloc[i].item()
            prev_close = df["Close"].iloc[i-1].item()
            current_high = df["High"].iloc[i].item()
            current_low = df["Low"].iloc[i].item()
            current_volume = df["Volume"].iloc[i].item()
            prev_atr = df["ATR"].iloc[i-1].item()
            roll_max = df["roll_max_cp"].iloc[i].item()
            roll_min = df["roll_min_cp"].iloc[i].item()
            roll_max_vol = df["roll_max_vol"].iloc[i].item()

            # --- Signal Generation ---
            if tickers_signal == "":
                tickers_ret[ticker][i] = 0
                # Buy signal: Breakout above rolling max AND volume surge
                if current_high >= roll_max and current_volume > vol_factor * roll_max_vol:
                    tickers_signal = "Buy"
                # Sell signal: Breakout below rolling min AND volume surge
                elif current_low <= roll_min and current_volume > vol_factor * roll_max_vol:
                    tickers_signal = "Sell"
            
            # --- Position Management (Buy) ---
            elif tickers_signal == "Buy":
                stop_loss = prev_close - prev_atr
                # Stop loss hit
                if current_low < stop_loss:
                    tickers_signal = ""
                    # Return if stopped out
                    tickers_ret[ticker][i] = (stop_loss / prev_close) - 1
                # Reverse signal
                elif current_low <= roll_min and current_volume > vol_factor * roll_max_vol:
                    tickers_signal = "Sell"
                    # Return from previous long position
                    tickers_ret[ticker][i] = (current_close / prev_close) - 1 
                else:
                    # Hold position
                    tickers_ret[ticker][i] = (current_close / prev_close) - 1
            
            # --- Position Management (Sell) ---
            elif tickers_signal == "Sell":
                stop_loss = prev_close + prev_atr
                # Stop loss hit
                if current_high > stop_loss:
                    tickers_signal = ""
                    # Return if stopped out
                    tickers_ret[ticker][i] = (prev_close / stop_loss) - 1
                # Reverse signal
                elif current_high >= roll_max and current_volume > vol_factor * roll_max_vol:
                    tickers_signal = "Buy"
                    # Return from previous short position
                    tickers_ret[ticker][i] = (prev_close / current_close) - 1
                else:
                    # Hold position
                    tickers_ret[ticker][i] = (prev_close / current_close) - 1
        
        # Assign returns to the DataFrame
        ohlc_dict_[ticker]["ret"] = np.array(tickers_ret[ticker])
    
    return ohlc_dict_


# ---------------------------- MAIN ---------------------------- #
def main():
    print("Fetching intraday OHLCV data using Alpha Vantage...")
    ohlcv_data = fetch_ohlcv_data_av(TICKERS, ALPHA_VANTAGE_KEY_PATH, INTERVAL)
    tickers = list(ohlcv_data.keys())

    if not tickers:
        print("No data fetched. Exiting.")
        return

    print("Running breakout strategy and calculating returns...")
    backtested_data = run_breakout_strategy(ohlcv_data)

    # Consolidate returns for overall strategy
    strategy_df = pd.DataFrame()
    for ticker in backtested_data:
        strategy_df[ticker] = backtested_data[ticker]["ret"]
    
    # Strategy return is the equal-weighted average of all stock returns for each 5-min period
    strategy_df["ret"] = strategy_df.mean(axis=1)

    ## Overall Strategy KPIs
    print("\n" + "="*30)
    print("--- Overall Strategy KPIs ---")
    print("="*30)
    
    # Calculate KPIs
    cagr_strategy = calculate_cagr(strategy_df["ret"], PERIODS_PER_YEAR)
    sharpe_strategy = calculate_sharpe_ratio(strategy_df["ret"], RISK_FREE_RATE, PERIODS_PER_YEAR)
    maxdd_strategy = calculate_maximum_drawdown(strategy_df["ret"])
    
    print(f"CAGR: {cagr_strategy:.4f}")
    print(f"Sharpe Ratio: {sharpe_strategy:.4f}")
    print(f"Max Drawdown: {maxdd_strategy:.4f}")
    
    # Visualization of overall strategy return
    plt.figure(figsize=(12, 6))
    (1 + strategy_df["ret"]).cumprod().plot(title="Cumulative Returns: Intraday Breakout Strategy (Alpha Vantage)")
    plt.xlabel("5-Minute Periods")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\n" + "="*30)
    print("--- Individual Stock KPIs ---")
    print("="*30)
    
    # Calculate individual stock's KPIs
    kpi_results = {}
    for ticker in tickers:
        ret_series = backtested_data[ticker]["ret"]
        cagr = calculate_cagr(ret_series, PERIODS_PER_YEAR)
        sharpe_ratios = calculate_sharpe_ratio(ret_series, RISK_FREE_RATE, PERIODS_PER_YEAR)
        max_drawdown = calculate_maximum_drawdown(ret_series)
        kpi_results[ticker] = [cagr, sharpe_ratios, max_drawdown]

    KPI_df = pd.DataFrame.from_dict(
        kpi_results, 
        orient='index', 
        columns=["CAGR", "Sharpe Ratio", "Max Drawdown"]
    )
    
    # Requires 'tabulate' library
    try:
        print(KPI_df.to_markdown(floatfmt=".4f"))
    except ImportError:
        print("\nNote: Install 'tabulate' (pip install tabulate) to display the table in markdown format.")
        print(KPI_df)

if __name__ == '__main__':
    main()
    