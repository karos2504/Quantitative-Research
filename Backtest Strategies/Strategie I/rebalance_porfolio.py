import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt

# ---------------------------- CONFIGURATION ---------------------------- #
TICKERS = [
    "MMM", "AXP", "T", "BA", "CAT", "CSCO", "KO", "XOM", "GE", "GS", "HD",
    "IBM", "INTC", "JNJ", "JPM", "MCD", "MRK", "MSFT", "NKE", "PFE", "PG",
    "TRV", "UNH", "VZ", "V", "WMT", "DIS"
]

START_DATE = dt.datetime.today() - dt.timedelta(days=365 * 10)
END_DATE = dt.datetime.today()
INTERVAL = '1mo'
NUM_OF_TRADING = 12
RISK_FREE_RATE = 0.025

# ---------------------------- DATA FETCHING ---------------------------- #
def fetch_ohlcv_data(tickers, start, end, interval):
    """Fetch OHLCV data for given tickers."""
    return {
        ticker: yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=False).dropna()
        for ticker in tickers
    }

# ---------------------------- KPIs ---------------------------- #
def calculate_cagr_from_returns(returns_series, periods_per_year):
    """
    Calculate CAGR given a series of periodic returns (not prices).
    """
    cumulative_return = (1 + returns_series).prod()
    years = len(returns_series) / periods_per_year
    cagr = cumulative_return ** (1 / years) - 1
    return cagr

def calculate_cagr_from_prices(df, periods_per_year):
    """
    Calculate CAGR from a price DataFrame with 'Adj Close'.
    """
    start = df['Adj Close'].iloc[0]
    end = df['Adj Close'].iloc[-1]
    years = len(df) / periods_per_year
    cagr = (end / start) ** (1 / years) - 1
    return cagr.item()

def calculate_sharpe_ratio(returns_series, risk_free_rate, periods_per_year):
    """
    Annualized Sharpe ratio from a returns series.
    """
    excess_return = returns_series - risk_free_rate / periods_per_year
    mean_excess = excess_return.mean() * periods_per_year
    std_excess = returns_series.std() * np.sqrt(periods_per_year)
    return mean_excess / std_excess if std_excess != 0 else np.nan

def calculate_maximum_drawdown(returns_series):
    """
    Maximum drawdown from a returns series.
    """
    cumulative = (1 + returns_series).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()

# ---------------------------- MONTHLY RETURN CALCULATION ---------------------------- #
def calculate_monthly_returns(data):
    """Calculate monthly returns for each ticker."""
    return_df = pd.DataFrame()
    for ticker in data:
        df = data[ticker].copy()
        df['Monthly Return'] = df['Adj Close'].pct_change()
        return_df[ticker] = df['Monthly Return']
    return return_df.dropna()

# ---------------------------- PORTFOLIO STRATEGY ---------------------------- #
def run_portfolio_strategy(returns_df, portfolio_size, drop_count):
    """
    Implements a monthly rebalancing strategy:
    - Picks top `portfolio_size` performers initially.
    - Drops `drop_count` worst performers monthly.
    """
    portfolio = []
    monthly_returns = [0]  # Initial return

    for i in range(len(returns_df)):
        if portfolio:
            # Mean return of current portfolio
            monthly_returns.append(returns_df[portfolio].iloc[i].mean())

            # Drop worst performers
            worst = returns_df[portfolio].iloc[i].nsmallest(drop_count).index.tolist()
            portfolio = [stock for stock in portfolio if stock not in worst]

        # Fill portfolio with new top performers
        needed = portfolio_size - len(portfolio)
        top_performers = returns_df.iloc[i].nlargest(needed).index.tolist()
        portfolio += top_performers

    return pd.DataFrame(monthly_returns, columns=["Monthly Return"]).iloc[1:]

# ---------------------------- PLOTTING FUNCTION ---------------------------- #
def plot_strategy_vs_benchmark(strategy_returns, benchmark_returns, title="Cumulative Returns: Strategy vs Benchmark"):
    """
    Plot cumulative returns of strategy vs benchmark.
    """
    plt.figure(figsize=(12, 6))
    plt.plot((1 + strategy_returns).cumprod().reset_index(drop=True), label="Strategy Return", linewidth=2)
    plt.plot((1 + benchmark_returns).cumprod().reset_index(drop=True), label="Benchmark Return", linewidth=2, linestyle='--')
    plt.title(title)
    plt.xlabel("Months")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# ---------------------------- MAIN ---------------------------- #
def main():
    # Fetch and process data
    ohlcv_data = fetch_ohlcv_data(TICKERS, START_DATE, END_DATE, INTERVAL)
    returns_df = calculate_monthly_returns(ohlcv_data)

    # Run strategy
    strategy_returns = run_portfolio_strategy(returns_df, portfolio_size=6, drop_count=3)

    # Run benchmark (DJI)
    dji = yf.download("^DJI", start=START_DATE, end=END_DATE, interval=INTERVAL, auto_adjust=False)
    dji['Monthly Return'] = dji['Adj Close'].pct_change().dropna()

    # KPIs for Strategy
    print("\n--- Strategy KPIs ---")
    cagr_strategy = calculate_cagr_from_returns(strategy_returns["Monthly Return"], NUM_OF_TRADING)
    sharpe_strategy = calculate_sharpe_ratio(strategy_returns["Monthly Return"], RISK_FREE_RATE, NUM_OF_TRADING)
    maxdd_strategy = calculate_maximum_drawdown(strategy_returns["Monthly Return"])
    print(f"CAGR: {cagr_strategy:.4f}")
    print(f"Sharpe Ratio: {sharpe_strategy:.4f}")
    print(f"Max Drawdown: {maxdd_strategy:.4f}")

    # KPIs for DJI Benchmark (using price-based CAGR)
    print("\n--- DJI Benchmark KPIs ---")
    cagr_dji = calculate_cagr_from_prices(dji, NUM_OF_TRADING)
    sharpe_dji = calculate_sharpe_ratio(dji["Monthly Return"], RISK_FREE_RATE, NUM_OF_TRADING)
    maxdd_dji = calculate_maximum_drawdown(dji["Monthly Return"])
    print(f"CAGR: {cagr_dji:.4f}")
    print(f"Sharpe Ratio: {sharpe_dji:.4f}")
    print(f"Max Drawdown: {maxdd_dji:.4f}")

    # Visualization
    plot_strategy_vs_benchmark(
        strategy_returns["Monthly Return"],
        dji["Monthly Return"],
        title="Cumulative Returns: Strategy vs DJI"
    )

if __name__ == '__main__':
    main()
