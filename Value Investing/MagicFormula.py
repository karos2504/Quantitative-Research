import yfinance as yf
import pandas as pd

tickers = ["AAPL", "MSFT", "V", "JNJ", "WMT", "JPM", "PG", "MA", "NVDA", "UNH",
           "HD", "VZ", "MRK", "KO", "PEP", "XOM", "CVX", "ABBV", "PFE", "T",
           "INTC", "CSCO", "BA", "MCD", "NKE"]

def fetch_data(ticker):
    """Fetches financial data for a given ticker."""
    t = yf.Ticker(ticker)
    try:
        info = t.info
    except Exception:
        return None # Skip invalid tickers

    bs = t.balance_sheet
    inc = t.financials
    cf = t.cashflow

    def safe(df, key):
        """Safely retrieve the most recent financial figure."""
        try:
            return df.loc[key].dropna().values[0]
        except:
            return None

    data = {
        "EBIT": safe(inc, "EBIT"),
        "MarketCap": info.get("marketCap"),
        "CashFlowOps": safe(cf, "Total Cash From Operating Activities"),
        "Capex": safe(cf, "Capital Expenditures"),
        "CurrAsset": safe(bs, "Total Current Assets"),
        "CurrLiab": safe(bs, "Total Current Liabilities"),
        "PPE": safe(bs, "Property Plant And Equipment Net"),
        "BookValue": safe(bs, "Total Stockholder Equity"),
        "TotDebt": safe(bs, "Long Term Debt") or 0,
        "PrefStock": safe(bs, "Preferred Stock") or 0,
        "MinInterest": 0,
        "DivYield": info.get("dividendYield")
    }
    return data

# Fetch financials
financials = {}
for ticker in tickers:
    data = fetch_data(ticker)
    if data:
        financials[ticker] = data

# Build DataFrame
df = pd.DataFrame.from_dict(financials, orient='index')
df = df.apply(pd.to_numeric, errors='coerce')

# Compute TEV (Enterprise Value)
df["TEV"] = (
    df["MarketCap"].fillna(0)
    + df["TotDebt"].fillna(0)
    + df["PrefStock"].fillna(0)
    + df["MinInterest"].fillna(0)
    - (df["CurrAsset"].fillna(0) - df["CurrLiab"].fillna(0))
)

# Compute metrics
df["EarningYield"] = df["EBIT"] / df["TEV"]
df["FCFYield"] = (df["CashFlowOps"] - df["Capex"]) / df["MarketCap"]

# Calculate Invested Capital (IC): PPE + (Current Assets - Current Liabilities)
IC = df["PPE"].fillna(0) + df["CurrAsset"].fillna(0) - df["CurrLiab"].fillna(0)

# Set a floor on Invested Capital to avoid division by zero/near-zero
# If IC <= 0, set IC to a small positive floor (e.g., 1000) for the ROC calculation.
IC_positive_floor = IC.apply(lambda x: max(x, 1000))

# Calculate ROC
df["ROC"] = df["EBIT"] / IC_positive_floor

# Filter rows with necessary data
required_fields = ["EarningYield", "ROC", "DivYield"]
df = df.dropna(subset=required_fields)

# Rank for Magic Formula
df["Rank_EY"] = df["EarningYield"].rank(ascending=False)
df["Rank_ROC"] = df["ROC"].rank(ascending=False)
df["CombRank"] = df["Rank_EY"] + df["Rank_ROC"]
df["MagicFormulaRank"] = df["CombRank"].rank()

# Value stocks based on Magic Formula
value_stocks = df.sort_values("MagicFormulaRank")[["EarningYield", "ROC", "MagicFormulaRank"]].head(10)

# Highest Dividend Yield
high_div_stocks = df.sort_values("DivYield", ascending=False)["DivYield"].head(10)

# Combined Magic Formula + Dividend Yield
df["Rank_Div"] = df["DivYield"].rank(ascending=False)
df["CombinedRank"] = (df["Rank_EY"] + df["Rank_ROC"] + df["Rank_Div"]).rank()
combined_value_stocks = df.sort_values("CombinedRank")[["EarningYield", "ROC", "DivYield", "CombinedRank"]].head(10)

# Output
pd.set_option('display.float_format', lambda x: f'{x:,.4f}')
print("\n=== Value Stocks Based on Magic Formula ===")
print(value_stocks)

print("\n=== Highest Dividend Yield Stocks ===")
print(high_div_stocks)

print("\n=== Magic Formula + Dividend Yield Combined ===")
print(combined_value_stocks)