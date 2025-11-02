import yfinance as yf
import pandas as pd
import numpy as np

# List of tickers (DJIA components)
tickers = ["MMM","AXP","AAPL","BA","CAT","CVX","CSCO","KO","DIS","XOM",
           "GE","GS","HD","IBM","INTC","JNJ","JPM","MCD","MRK",
           "MSFT","NKE","PFE","PG","TRV","UNH","VZ","V","WMT"]

# --- Standardized Metric Mapping (Highly Robust with Fallbacks) ---
stats_map = {
    "NetIncome": ["Net Income"],
    "TotAssets": ["Total Assets"],
    "CashFlowOps": ["Cash Flow From Operating Activities", 
                    "Total Cash From Operating Activities", 
                    "Operating Cash Flow"], 
    "LTDebt": ["Long Term Debt", 
               "Long Term Debt And Capital Lease Obligation", 
               "Non Current Debt"], 
    "CurrAssets": ["Current Assets", 
                   "Total Current Assets", 
                   "Total Assets Net Of Current Liabilities"],
    "CurrLiab": ["Current Liabilities", 
                 "Total Current Liabilities",
                 "Current Liabilities And Debt"],
    "CommStock": ["Ordinary Shares Number", 
                  "Share Issued", 
                  "Common Stock"], 
    "TotRevenue": ["Total Revenue"],
    "GrossProfit": ["Gross Profit"]
}

# --- Robust Data Fetching Function ---
def fetch_financial_data(ticker):
    """Fetches and aligns the 9 required Piotroski metrics for the 2 most recent years."""
    stock = yf.Ticker(ticker)
    
    try:
        bs = stock.balance_sheet
        is_ = stock.financials
        cf = stock.cashflow
    except Exception:
        return None

    combined_df = pd.concat([is_, bs, cf], axis=0) 
    required_data = {}
    
    # Iterate through the list of possible keys for each metric
    for label, possible_fields in stats_map.items():
        s = pd.Series([], dtype=float)
        
        for field in possible_fields:
            if field in combined_df.index:
                s = combined_df.loc[field]
                break 

        # Skip if the metric is not found AT ALL after checking all possibilities.
        if s.empty: 
            return None 
        
        # Extract the top 2 values (Year_0, Year_1)
        s_standardized = s.head(2)
        
        # Handle cases where only one year is available by padding with NaN
        if len(s_standardized) < 2:
            s_standardized = s_standardized.reindex(s_standardized.index.union(pd.Index(['Year_1_Placeholder']))).head(2)
        
        required_data[label] = s_standardized.values
    
    # Create the final DataFrame (Metrics as Rows, Standardized Years as Columns)
    df = pd.DataFrame(required_data).T
    df.columns = ['Year_0', 'Year_1']
    
    return df

# -------------------------------------------------------------
# --- Collecting Data ---
# -------------------------------------------------------------
all_data = {}
for ticker in tickers:
    df = fetch_financial_data(ticker)
    if df is not None:
        all_data[ticker] = df

# -------------------------------------------------------------
# --- Piotroski F-Score Calculation Function ---
# -------------------------------------------------------------
def piotroski_f_score(all_data):
    """Calculates the Piotroski F-Score (0-9) for each stock."""
    scores = {}
    
    # Financial Institution exclusion list (ratios are unreliable for F-Score)
    financial_tickers = ["AXP", "GS", "JPM", "TRV", "V"] 

    for ticker, data in all_data.items():
        try:
            if ticker in financial_tickers:
                continue 
            
            cy = data['Year_0']
            py = data['Year_1']
            
            # Helper function to safely extract a scalar value (NaN -> 0.0)
            def get_scalar(series_element):
                return float(series_element if not pd.isna(series_element) else 0.0)

            # Helper function for safe division (Denominator 0 -> 0.0)
            def safe_div(numerator_val, denominator_val):
                N = get_scalar(numerator_val)
                D = get_scalar(denominator_val)
                return N / D if D != 0 else 0.0
            
            # --- CALCULATE REQUIRED RATIOS ---
            ROA_CY = safe_div(cy["NetIncome"], cy["TotAssets"])
            CFO_ROA_NUM_CY = safe_div(cy["CashFlowOps"], cy["TotAssets"])
            CR_CY = safe_div(cy["CurrAssets"], cy["CurrLiab"])
            GM_CY = safe_div(cy["GrossProfit"], cy["TotRevenue"])
            ATO_CY = safe_div(cy["TotRevenue"], cy["TotAssets"])
            
            ROA_PY = safe_div(py["NetIncome"], py["TotAssets"])
            CR_PY = safe_div(py["CurrAssets"], py["CurrLiab"])
            GM_PY = safe_div(py["GrossProfit"], py["TotRevenue"])
            ATO_PY = safe_div(py["TotRevenue"], py["TotAssets"])

            # --- PROFITABILITY (4 points) ---
            ROA_FS = int(ROA_CY > 0)
            CFO_FS = int(get_scalar(cy["CashFlowOps"]) > 0)
            ROA_D_FS = int(ROA_CY > ROA_PY)
            CFO_ROA_FS = int(CFO_ROA_NUM_CY > ROA_CY)

            # --- LEVERAGE, LIQUIDITY, & SOURCE OF FUNDS (3 points) ---
            LTD_FS = int(get_scalar(cy["LTDebt"]) < get_scalar(py["LTDebt"]))
            CR_FS = int(CR_CY > CR_PY)
            DILUTION_FS = int(get_scalar(cy["CommStock"]) <= get_scalar(py["CommStock"]))

            # --- OPERATING EFFICIENCY (2 points) ---
            GM_FS = int(GM_CY > GM_PY)
            ATO_FS = int(ATO_CY > ATO_PY)

            total_score = sum([
                ROA_FS, CFO_FS, ROA_D_FS, CFO_ROA_FS,
                LTD_FS, CR_FS, DILUTION_FS, GM_FS, ATO_FS
            ])

            scores[ticker] = total_score

        except Exception:
            pass 

    return pd.Series(scores, name="F-Score").sort_values(ascending=False)

# -------------------------------------------------------------
# --- Compute and Print F-Scores ---
# -------------------------------------------------------------
f_scores = piotroski_f_score(all_data)

print("\nTop Piotroski F-Score Stocks:")
print(f_scores.head(10))
