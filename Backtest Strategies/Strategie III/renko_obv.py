import numpy as np
import pandas as pd
import statsmodels.api as sm
import copy
import yfinance as yf

# --- TECHNICAL INDICATOR FUNCTIONS ---

def ATR(DF, n):
    """Function to calculate True Range and Average True Range"""
    df = DF.copy()
    df['H-L'] = abs(df['High'] - df['Low'])
    df['H-PC'] = abs(df['High'] - df['Adj Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Adj Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1, skipna=False)
    df['ATR'] = df['TR'].rolling(n).mean()
    df2 = df.drop(['H-L', 'H-PC', 'L-PC'], axis=1)
    return df2

def slope(ser, n):
    """Function to calculate the slope of n consecutive points on a plot"""
    slopes = [i * 0 for i in range(n - 1)]

    for i in range(n, len(ser) + 1):
        y = ser.iloc[i - n:i] 
        x = np.array(range(n))
        y_scaled = (y - y.min()) / (y.max() - y.min())
        x_scaled = (x - x.min()) / (x.max() - x.min())
        x_scaled = sm.add_constant(x_scaled)
        model = sm.OLS(y_scaled, x_scaled)
        results = model.fit()
        slopes.append(results.params.iloc[-1])
            
    slope_angle = (np.rad2deg(np.arctan(np.array(slopes))))
    return np.array(slope_angle)

def renko_df(DF):
    """Custom function to convert OHLC data into renko bricks using ATR for brick size."""
    df = DF.copy()
    
    # 1. Calculate Brick Size (using ATR)
    df_atr = ATR(df, 120)
    try:
        brick_size = max(0.5, round(df_atr["ATR"].iloc[-1], 0))
    except (IndexError, KeyError):
        brick_size = 0.5

    # 2. Renko generation logic
    data = df["Close"].to_numpy()
    dates = df.index.to_numpy()
    
    renko_dates, renko_closes, renko_uptrend = [], [], []
    current_brick_close = data[0]
    
    for i in range(1, len(data)):
        change = data[i] - current_brick_close
        
        if abs(change) >= brick_size:
            num_bricks = int(change / brick_size)
            trend = num_bricks > 0

            for _ in range(abs(num_bricks)):
                current_brick_close += np.sign(num_bricks) * brick_size
                renko_dates.append(dates[i])
                renko_closes.append(current_brick_close)
                renko_uptrend.append(trend)

    # 3. Create the Renko DataFrame
    renko_df = pd.DataFrame({
        'date': renko_dates,
        'close': renko_closes,
        'uptrend': renko_uptrend
    })
    
    if renko_df.empty:
        return pd.DataFrame(columns=['date', 'uptrend', 'bar_num']) 

    # 4. Calculate Bar Number (Consecutive Bricks)
    renko_df["bar_num"] = np.where(renko_df["uptrend"] == True, 1, -1)
    bar_num_series = renko_df["bar_num"]
    for i in range(1, len(bar_num_series)):
        current_index = bar_num_series.index[i]
        is_same_trend = (bar_num_series.iloc[i] > 0) == (bar_num_series.iloc[i-1] > 0)
        
        if is_same_trend:
            renko_df.loc[current_index, "bar_num"] = bar_num_series.iloc[i] + bar_num_series.iloc[i-1]
            
    # 5. Final cleanup
    renko_df.drop_duplicates(subset="date", keep="last", inplace=True)
    renko_df = renko_df.loc[:, ["date", "bar_num"]] 
    
    return renko_df

def OBV(DF):
    """function to calculate On Balance Volume"""
    df = DF.copy()
    df['daily_ret'] = df['Adj Close'].pct_change()
    df['direction'] = np.where(df['daily_ret'].fillna(0) >= 0, 1, -1).astype(int)
    df.loc[df.index[0], 'direction'] = 0 
    df['vol_adj'] = df['Volume'] * df['direction']
    df['obv'] = df['vol_adj'].cumsum()
    return df['obv']

# --- KPI CALCULATION FUNCTIONS ---

def CAGR(DF):
    """function to calculate the Cumulative Annual Growth Rate of a trading strategy"""
    df = DF.copy()
    df["cum_return"] = (1 + df["ret"]).cumprod()
    n = len(df) / (252 * 78) 
    cagr = (df["cum_return"].iloc[-1])**(1 / n) - 1
    return cagr

def volatility(DF):
    """function to calculate annualized volatility of a trading strategy"""
    df = DF.copy()
    vol = df["ret"].std() * np.sqrt(252 * 78)
    return vol

def sharpe(DF, rf):
    """function to calculate sharpe ratio ; rf is the risk free rate"""
    df = DF.copy()
    sr = (CAGR(df) - rf) / volatility(df)
    return sr

def max_dd(DF):
    """function to calculate max drawdown"""
    df = DF.copy()
    df["cum_return"] = (1 + df["ret"]).cumprod()
    df["cum_roll_max"] = df["cum_return"].cummax()
    df["drawdown"] = df["cum_roll_max"] - df["cum_return"]
    df["drawdown_pct"] = df["drawdown"] / df["cum_roll_max"]
    max_dd = df["drawdown_pct"].max()
    return max_dd

# --- DATA FETCHING AND BACKTESTING ---

tickers = ["MSFT", "AAPL", "GOOGL", "META", "AMZN", "INTC", "CSCO", "VZ", "IBM", "TSLA", "AMD"]
ohlc_intraday = {}

print("--- Starting data download using yfinance ---")

for ticker in tickers:
    try:
        data = yf.download(ticker, interval='5m', period='60d', progress=False, auto_adjust=True)
        
        data.columns = ["Open", "High", "Low", "Adj Close", "Volume"]
        data['Close'] = data['Adj Close'] 
        
        data.dropna(inplace=True) 

        ohlc_intraday[ticker] = data
        print(f"✅ Downloaded data for {ticker}. Rows: {len(data)}")
    except Exception as e:
        print(f"❌ Failed to fetch data for {ticker}. Error: {e}")

tickers = list(ohlc_intraday.keys()) 

if not tickers:
    raise ValueError("No data was successfully downloaded. Please check ticker symbols or connection.")
else:
    print(f"\nData prepared for backtesting tickers: {tickers}")

# -----------------------------------------------------------------------------
## Backtesting Logic
# -----------------------------------------------------------------------------

ohlc_renko = {}
df = copy.deepcopy(ohlc_intraday)
tickers_signal = {}
tickers_ret = {}

# 1. Generate Renko/OBV and Merge
for ticker in tickers:
    print(f"📊 Merging Renko and OBV for {ticker}")
    
    renko = renko_df(df[ticker])
    
    df[ticker]["Date"] = df[ticker].index
    
    df[ticker]['Date'] = pd.to_datetime(df[ticker]['Date']).dt.tz_localize(None)
    
    if not renko.empty:
        renko['date'] = pd.to_datetime(renko['date']).dt.tz_localize(None)
        renko.rename(columns={'date': 'Date'}, inplace=True) 

        ohlc_renko[ticker] = df[ticker].merge(renko.loc[:, ["Date", "bar_num"]], how="outer", on="Date")
    else:
        ohlc_renko[ticker] = df[ticker].copy()
        ohlc_renko[ticker]["bar_num"] = np.nan

    ohlc_renko[ticker]["bar_num"] = ohlc_renko[ticker]["bar_num"].ffill()
    
    ohlc_renko[ticker]["obv"] = OBV(ohlc_renko[ticker])
    ohlc_renko[ticker]["obv_slope"] = slope(ohlc_renko[ticker]["obv"], 5)
    
    tickers_signal[ticker] = ""
    tickers_ret[ticker] = []

# 2. Strategy Execution and Return Calculation
for ticker in tickers:
    
    df_ticker = ohlc_renko[ticker]
    for i in range(len(df_ticker)):
        
        if i == 0:
            tickers_ret[ticker].append(0)
            continue

        row = df_ticker.iloc[i]
        prev_close = df_ticker["Adj Close"].iloc[i-1]
        current_close = row["Adj Close"]
        
        # No position
        if tickers_signal[ticker] == "":
            tickers_ret[ticker].append(0)
            if row["bar_num"] >= 2 and row["obv_slope"] > 30:
                tickers_signal[ticker] = "Buy"
            elif row["bar_num"] <= -2 and row["obv_slope"] < -30:
                tickers_signal[ticker] = "Sell"
        
        # Long position
        elif tickers_signal[ticker] == "Buy":
            tickers_ret[ticker].append((current_close / prev_close) - 1)
            
            if row["bar_num"] <= -2 and row["obv_slope"] < -30:
                tickers_signal[ticker] = "Sell"
            elif row["bar_num"] < 2: 
                tickers_signal[ticker] = ""
                
        # Short position
        elif tickers_signal[ticker] == "Sell":
            tickers_ret[ticker].append((prev_close / current_close) - 1) 
            
            if row["bar_num"] >= 2 and row["obv_slope"] > 30:
                tickers_signal[ticker] = "Buy"
            elif row["bar_num"] > -2: 
                tickers_signal[ticker] = ""
                
    ohlc_renko[ticker]["ret"] = np.array(tickers_ret[ticker])

# 3. KPI Calculation
strategy_df = pd.DataFrame()
for ticker in tickers:
    strategy_df[ticker] = ohlc_renko[ticker]["ret"]

strategy_df["ret"] = strategy_df.mean(axis=1) 

print("\n--- 🎯 Overall Strategy KPIs (Equal-Weighted Portfolio) ---")
print(f"CAGR: {CAGR(strategy_df) * 100:.2f}%")
print(f"Sharpe Ratio (RF=2.5%): {sharpe(strategy_df, 0.025):.2f}")
print(f"Max Drawdown: {max_dd(strategy_df) * 100:.2f}%")
print("---------------------------------------------------------")

# Individual Stock KPIs
cagr, sharpe_ratios, max_drawdown = {}, {}, {}
for ticker in tickers:
    cagr[ticker] = CAGR(ohlc_renko[ticker])
    sharpe_ratios[ticker] = sharpe(ohlc_renko[ticker], 0.025)
    max_drawdown[ticker] = max_dd(ohlc_renko[ticker])

KPI_df = pd.DataFrame([cagr, sharpe_ratios, max_drawdown], 
                      index=["Return", "Sharpe Ratio", "Max Drawdown"])      

print("\n--- 📈 Individual Stock KPIs ---")
print(KPI_df.T)
