import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import copy
import matplotlib.pyplot as plt

# ---------------------------- CONFIGURATION ---------------------------- #
TICKERS = ["MSFT", "AAPL", "GOOGL", "META", "AMZN", "INTC", "CSCO", "VZ", "IBM", "TSLA", "AMD"]
# Note: For intraday data, yfinance free API has limits. We use a shorter recent period.
# Max 60 days for 5m interval is typical for yfinance.
START_DATE = dt.datetime.today() - dt.timedelta(days=59)
END_DATE = dt.datetime.today()
INTERVAL = '5m'
# Number of 5-minute periods in a trading day (9:35 to 16:00 is 386 minutes / 5 = 77.2 periods).
# We'll use 78 periods per day for calculation simplicity (covers 9:35 to 16:00 inclusive).
# Trading days per year: 252.
PERIODS_PER_YEAR = 252 * 78 
RISK_FREE_RATE = 0.025

# ---------------------------- KPIs ---------------------------- #
def calculate_atr(df, period=14):
    """Function to calculate Average True Range (ATR)"""
    df = df.copy()
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    
    atr = tr.ewm(span=period, adjust=False).mean()
    return atr

def calculate_cagr(returns_series, periods_per_year):
    """
    Calculate CAGR given a series of periodic returns.
    """
    cumulative_return = (1 + returns_series).prod()
    # A simple way to approximate years, though a more precise date calculation is better for daily/intraday
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

# ---------------------------- DATA FETCHING ---------------------------- #
def fetch_ohlcv_data(tickers, start, end, interval):
    """Fetch OHLCV data for given tickers and preprocess it."""
    ohlc_intraday = {}
    for ticker in tickers:
        try:
            data = yf.download(
                ticker, 
                start=start, 
                end=end, 
                interval=interval,
                auto_adjust=True
            )[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            
            # Filter trading hours (9:35 AM to 4:00 PM EST)
            # The timezone from yfinance depends on the stock, typically UTC or exchange time.
            # We'll rely on the index time if it's already localized/aware.
            # For simplicity, we filter based on time string assuming EST or NYSE-equivalent time.
            data = data.between_time('09:35', '16:00')
            
            # The original code reverses the data index, yfinance is typically oldest-first.
            # We will use the yfinance order (oldest first) and adjust the main loop index.
            ohlc_intraday[ticker] = data
        except Exception as e:
            print(f"Could not fetch data for {ticker}: {e}")
            continue
            
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
                    # Return if stopped out: (Exit Price / Entry Price) - 1. Entry price is Prev Close.
                    # Exit price is the stop loss level.
                    tickers_ret[ticker][i] = (stop_loss / prev_close) - 1
                # Reverse signal: Breakout below rolling min AND volume surge
                elif current_low <= roll_min and current_volume > vol_factor * roll_max_vol:
                    tickers_signal = "Sell"
                    # Return from previous long position: (Current Close / Prev Close) - 1
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
                    # Return if stopped out: (Entry Price / Exit Price) - 1. Entry price is Prev Close.
                    # Exit price is the stop loss level.
                    tickers_ret[ticker][i] = (prev_close / stop_loss) - 1
                # Reverse signal: Breakout above rolling max AND volume surge
                elif current_high >= roll_max and current_volume > vol_factor * roll_max_vol:
                    tickers_signal = "Buy"
                    # Return from previous short position: (Prev Close / Current Close) - 1
                    tickers_ret[ticker][i] = (prev_close / current_close) - 1
                else:
                    # Hold position
                    tickers_ret[ticker][i] = (prev_close / current_close) - 1
        
        # Assign returns to the DataFrame
        ohlc_dict_[ticker]["ret"] = np.array(tickers_ret[ticker])
    
    return ohlc_dict_

def plot_strategy(strategy_df):
    # Visualization of overall strategy return
    plt.figure(figsize=(12, 6))
    (1 + strategy_df["ret"]).cumprod().plot(title="Cumulative Returns: Intraday Breakout Strategy")
    plt.xlabel("5-Minute Periods")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ---------------------------- MAIN ---------------------------- #
def main():
    print("Fetching intraday OHLCV data...")
    ohlcv_data = fetch_ohlcv_data(TICKERS, START_DATE, END_DATE, INTERVAL)
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
    
    plot_strategy(strategy_df)

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
    
    print(KPI_df.to_markdown(floatfmt=".4f"))

if __name__ == '__main__':
    main()
