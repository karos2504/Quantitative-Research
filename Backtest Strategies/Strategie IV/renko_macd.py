import numpy as np
import pandas as pd
from stocktrends import Renko
import statsmodels.api as sm
import copy
import yfinance as yf

# --- TECHNICAL INDICATOR FUNCTIONS ---

def MACD(DF,a,b,c):
    """Function to calculate MACD
       typical values a = 12; b =26, c =9"""
    df = DF.copy()
    # Ensure 'Adj Close' is a float type for EWM calculation
    df["Adj Close"] = pd.to_numeric(df["Adj Close"], errors='coerce') 
    df["MA_Fast"]=df["Adj Close"].ewm(span=a,min_periods=a).mean()
    df["MA_Slow"]=df["Adj Close"].ewm(span=b,min_periods=b).mean()
    df["MACD"]=df["MA_Fast"]-df["MA_Slow"]
    df["Signal"]=df["MACD"].ewm(span=c,min_periods=c).mean()
    df.dropna(inplace=True)
    return (df["MACD"],df["Signal"])

def ATR(DF,n):
    "Function to calculate True Range and Average True Range"
    df = DF.copy()
    df['H-L']=abs(df['High']-df['Low'])
    df['H-PC']=abs(df['High']-df['Adj Close'].shift(1))
    df['L-PC']=abs(df['Low']-df['Adj Close'].shift(1))
    df['TR']=df[['H-L','H-PC','L-PC']].max(axis=1,skipna=False)
    df['ATR'] = df['TR'].rolling(n).mean()
    df2 = df.drop(['H-L','H-PC','L-PC'],axis=1)
    return df2

def slope(ser,n):
    "Function to calculate the slope of n consecutive points on a plot"
    slopes = [i*0 for i in range(n-1)]
    
    # Ensure series indices are handled for correct slicing/access
    ser_values = ser.values 
    for i in range(n,len(ser_values)+1):
        y = ser_values[i-n:i]
        x = np.array(range(n))
        y_scaled = (y - y.min())/(y.max() - y.min())
        x_scaled = (x - x.min())/(x.max() - x.min())
        x_scaled = sm.add_constant(x_scaled)
        model = sm.OLS(y_scaled,x_scaled)
        results = model.fit()
        slopes.append(results.params[-1])
    slope_angle = (np.rad2deg(np.arctan(np.array(slopes))))
    return np.array(slope_angle)

def renko_df(DF):
    """function to convert ohlc data into renko bricks using stocktrends and ATR"""
    df = DF.copy()
    df.reset_index(inplace=True)
    # Standardize column names for stocktrends Renko object (yfinance index is datetime)
    df = df.iloc[:,[0,1,2,3,4,5]]
    df.columns = ["date","open","high","low","close","volume"]
    
    # Calculate Brick Size (using ATR)
    df_atr = ATR(DF, 120) 
    try:
        brick_size = max(0.5, round(df_atr["ATR"].iloc[-1], 0))
    except (IndexError, KeyError):
        brick_size = 0.5
        
    df2 = Renko(df)
    df2.brick_size = brick_size
    # --- FIX 1: Using get_ohlc_data() as it was the last successful attempt before the final error
    # NOTE: If this fails, try get_bricks() or reinstalling the library.
    renko_df = df2.get_ohlc_data() 
    
    # Calculate Bar Number (Consecutive Bricks)
    renko_df["bar_num"] = np.where(renko_df["uptrend"]==True,1,np.where(renko_df["uptrend"]==False,-1,0))
    for i in range(1,len(renko_df)):
        current_index = renko_df.index[i]
        prev_bar_num = renko_df.loc[renko_df.index[i-1], "bar_num"]
        
        if renko_df.loc[current_index, "bar_num"] > 0 and prev_bar_num > 0:
            renko_df.loc[current_index, "bar_num"] += prev_bar_num
        elif renko_df.loc[current_index, "bar_num"] < 0 and prev_bar_num < 0:
            renko_df.loc[current_index, "bar_num"] += prev_bar_num

    # Final cleanup
    renko_df.drop_duplicates(subset="date",keep="last",inplace=True)
    
    # This DataFrame now has 7 columns (date, open, high, low, close, uptrend, bar_num)
    return renko_df

# --- KPI CALCULATION FUNCTIONS (Unchanged) ---

def CAGR(DF):
    "function to calculate the Cumulative Annual Growth Rate of a trading strategy"
    df = DF.copy()
    df["cum_return"] = (1 + df["ret"]).cumprod()
    n = len(df)/(252*78)
    CAGR = (df["cum_return"].iloc[-1])**(1/n) - 1
    return CAGR

def volatility(DF):
    "function to calculate annualized volatility of a trading strategy"
    df = DF.copy()
    vol = df["ret"].std() * np.sqrt(252*78)
    return vol

def sharpe(DF,rf):
    "function to calculate sharpe ratio ; rf is the risk free rate"
    df = DF.copy()
    sr = (CAGR(df) - rf)/volatility(df)
    return sr
    
def max_dd(DF):
    "function to calculate max drawdown"
    df = DF.copy()
    df["cum_return"] = (1 + df["ret"]).cumprod()
    df["cum_roll_max"] = df["cum_return"].cummax()
    df["drawdown"] = df["cum_roll_max"] - df["cum_return"]
    df["drawdown_pct"] = df["drawdown"]/df["cum_roll_max"]
    max_dd = df["drawdown_pct"].max()
    return max_dd

# --- DATA FETCHING (yfinance) ---

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
## Backtesting Logic (Renko + MACD)
# -----------------------------------------------------------------------------

#Merging renko df with original ohlc df
ohlc_renko = {}
df = copy.deepcopy(ohlc_intraday)
tickers_signal = {}
tickers_ret = {}

for ticker in tickers:
    print(f"📊 Merging Renko and MACD for {ticker}")
    
    df_temp = df[ticker].copy()
    renko = renko_df(df_temp) 
    
    # --- Date Standardization and Clean Merge ---
    
    # 1. Prepare OHLC DataFrame dates (ensure TZ-naive)
    df[ticker]["Date"] = df[ticker].index
    df[ticker]['Date'] = pd.to_datetime(df[ticker]['Date']).dt.tz_localize(None)
    
    # 2. Prepare Renko DataFrame dates (rename and ensure TZ-naive)
    if not renko.empty:
        # Renko output column is 'date', rename to 'Date' to match OHLC DF for merge
        renko.rename(columns={'date': 'Date'}, inplace=True)
        # Convert to datetime and make timezone-naive, resolving the ValueError
        renko['Date'] = pd.to_datetime(renko['Date']).dt.tz_localize(None)
    
        # Merge on the 'Date' column, only pulling 'Date' and 'bar_num' from Renko DF
        ohlc_renko[ticker] = df[ticker].merge(renko.loc[:,["Date","bar_num"]], how="outer", on="Date")
    else:
        ohlc_renko[ticker] = df[ticker].copy()
        ohlc_renko[ticker]["bar_num"] = np.nan
    # --------------------------------------------------------
    
    # Fill NaN bar_num values with the last valid Renko bar number
    ohlc_renko[ticker]["bar_num"] = ohlc_renko[ticker]["bar_num"].ffill()
    
    # Recalculate MACD and slopes on the merged DataFrame
    macd_series, signal_series = MACD(ohlc_renko[ticker],12,26,9)
    ohlc_renko[ticker]["macd"]= macd_series
    ohlc_renko[ticker]["macd_sig"]= signal_series
    
    # Fill any new NaNs introduced by MACD/slope calculation
    ohlc_renko[ticker].dropna(subset=["macd", "macd_sig", "bar_num"], inplace=True) 
    
    # Calculate slopes - only do this AFTER dropping NaNs from MACD calc
    if len(ohlc_renko[ticker]) >= 5: # Slope requires n=5 points
        ohlc_renko[ticker]["macd_slope"] = slope(ohlc_renko[ticker]["macd"],5)
        ohlc_renko[ticker]["macd_sig_slope"] = slope(ohlc_renko[ticker]["macd_sig"],5)
    else:
        # Handle case where not enough data remains after dropna
        ohlc_renko[ticker]["macd_slope"] = np.nan
        ohlc_renko[ticker]["macd_sig_slope"] = np.nan
        ohlc_renko[ticker].dropna(inplace=True)

    tickers_signal[ticker] = ""
    tickers_ret[ticker] = []

# Identifying signals and calculating daily return
for ticker in tickers:
    df_ticker = ohlc_renko[ticker].reset_index(drop=True)
    
    # Check if df is empty or too short
    if df_ticker.empty or len(df_ticker) <= 1:
        print(f"Skipping {ticker}: Insufficient data after indicator calculation.")
        continue
        
    for i in range(len(df_ticker)):
        row = df_ticker.iloc[i]
        
        if i == 0:
            tickers_ret[ticker].append(0)
            continue

        prev_close = df_ticker["Adj Close"].iloc[i-1]
        current_close = row["Adj Close"]
        
        # No position
        if tickers_signal[ticker] == "":
            tickers_ret[ticker].append(0)
            if row["bar_num"]>=2 and row["macd"]>row["macd_sig"] and row["macd_slope"]>row["macd_sig_slope"]:
                tickers_signal[ticker] = "Buy"
            elif row["bar_num"]<=-2 and row["macd"]<row["macd_sig"] and row["macd_slope"]<row["macd_sig_slope"]:
                tickers_signal[ticker] = "Sell"
        
        # Long position
        elif tickers_signal[ticker] == "Buy":
            tickers_ret[ticker].append((current_close/prev_close)-1)
            
            if row["bar_num"]<=-2 and row["macd"]<row["macd_sig"] and row["macd_slope"]<row["macd_sig_slope"]:
                tickers_signal[ticker] = "Sell"
            elif row["macd"]<row["macd_sig"] and row["macd_slope"]<row["macd_sig_slope"]:
                tickers_signal[ticker] = ""
                
        # Short position
        elif tickers_signal[ticker] == "Sell":
            tickers_ret[ticker].append((prev_close/current_close)-1)
            
            if row["bar_num"]>=2 and row["macd"]>row["macd_sig"] and row["macd_slope"]>row["macd_sig_slope"]:
                tickers_signal[ticker] = "Buy"
            elif row["macd"]>row["macd_sig"] and row["macd_slope"]>row["macd_sig_slope"]:
                tickers_signal[ticker] = ""

    ohlc_renko[ticker]["ret"] = np.array(tickers_ret[ticker])

#calculating overall strategy's KPIs
strategy_df = pd.DataFrame()
for ticker in tickers:
    if "ret" in ohlc_renko[ticker].columns:
        strategy_df[ticker] = ohlc_renko[ticker]["ret"]

# Proceed only if the strategy_df is not empty
if not strategy_df.empty:
    strategy_df["ret"] = strategy_df.mean(axis=1)

    print("\n--- 🎯 Overall Strategy KPIs (Equal-Weighted Portfolio - Renko + MACD) ---")
    print(f"CAGR: {CAGR(strategy_df) * 100:.2f}%")
    print(f"Sharpe Ratio (RF=2.5%): {sharpe(strategy_df, 0.025):.2f}")
    print(f"Max Drawdown: {max_dd(strategy_df) * 100:.2f}%")
    print("---------------------------------------------------------")

    #calculating individual stock's KPIs
    cagr, sharpe_ratios, max_drawdown = {}, {}, {}
    for ticker in tickers:
        if "ret" in ohlc_renko[ticker].columns and not ohlc_renko[ticker]["ret"].empty:
            cagr[ticker] =  CAGR(ohlc_renko[ticker])
            sharpe_ratios[ticker] =  sharpe(ohlc_renko[ticker],0.025)
            max_drawdown[ticker] =  max_dd(ohlc_renko[ticker])

    KPI_df = pd.DataFrame([cagr,sharpe_ratios,max_drawdown],index=["Return","Sharpe Ratio","Max Drawdown"])      
    
    print("\n--- 📈 Individual Stock KPIs (Renko + MACD) ---")
    print(KPI_df.T)
else:
    print("\n--- ⚠️ Strategy KPIs not calculated: Not enough data for backtesting after filtering. ---")