"""
Monthly Portfolio Rebalancing
"""

import sys
import os
import multiprocessing
from pathlib import Path

os.environ['PYTHONWARNINGS'] = 'ignore:resource_tracker:UserWarning'

if os.name == 'posix':
    try:
        if multiprocessing.get_start_method(allow_none=True) is None:
            multiprocessing.set_start_method('spawn', force=True)
    except (RuntimeError, ValueError):
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backtesting_engine.backtesting import VBTBacktester
from config.settings import CASH, COMMISSION
from portfolio_construction import kpi
from scipy.stats import zscore


from dataclasses import dataclass, field
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# ============================================================
#  CONFIG & METADATA
# ============================================================
@dataclass(frozen=True)
class StrategyConfig:
    risk_free_rate: float = 0.00
    candidate_size: int = 3
    min_weight: float = 0.02
    max_weight: float = 0.60
    min_positions: int = 3
    max_positions: int = 15
    mom_6_1_veto: float = 0.00
    rebal_threshold: float = 0.25
    smoothing: float = 0.3
    min_hold_days: int = 60
    order_book_min_delta: float = 0.005
    txn_cost_bps: int = 10
    min_backtest_months: int = 24
    entry_pullback: float = 0.01
    entry_rsi: int = 50
    min_mean_conviction: float = 0.01
    vol_comp_short: int = 20
    vol_comp_long: int = 100
    do_nothing_threshold: float = 0.001
    hard_risk_cap_drawdown: float = 0.25
    hard_risk_cap_recovery: float = 0.05
    strong_mom_threshold: float = 0.05
    apply_corr_penalty: bool = True
    convexity_vix_thresh_warn: float = 0.04
    convexity_vix_thresh_crash: float = 0.07
    convexity_min_invest: float = 0.20
    convexity_default_invest: float = 1.0
    hedge_tickers: list[str] = field(default_factory=lambda: ['TLT', 'GLD'])
    hedge_crash_alloc: float = 0.30
    hedge_normal_alloc: float = 0.0
    min_weight_change: float = 0.05
    max_drawdown_limit: float = 0.25
    cash: float = CASH
    commission: float = COMMISSION


@dataclass(frozen=True)
class UniverseMetadata:
    universe_with_sectors: dict[str, str] = field(default_factory=lambda: {
        "AAPL":"IT","MSFT":"IT","NVDA":"IT","AVGO":"IT","ORCL":"IT",
        "CRM":"IT","AMD":"IT","QCOM":"IT","TXN":"IT","CSCO":"IT",
        "INTU":"IT","IBM":"IT","NOW":"IT","AMAT":"IT","MU":"IT",
        "INTC":"IT","ADBE":"IT","KLAC":"IT","LRCX":"IT","ADI":"IT",
        "UNH":"HC","JNJ":"HC","LLY":"HC","ABBV":"HC","MRK":"HC",
        "TMO":"HC","ABT":"HC","DHR":"HC","BMY":"HC","AMGN":"HC",
        "PFE":"HC","SYK":"HC","ISRG":"HC","MDT":"HC","CI":"HC",
        "ELV":"HC","HCA":"HC","VRTX":"HC",
        "BRK-B":"FIN","JPM":"FIN","BAC":"FIN","WFC":"FIN","GS":"FIN",
        "MS":"FIN","BLK":"FIN","SCHW":"FIN","AXP":"FIN","CB":"FIN",
        "TRV":"FIN","PNC":"FIN","USB":"FIN","MET":"FIN",
        "PRU":"FIN","ICE":"FIN","CME":"FIN",
        "AMZN":"CD","TSLA":"CD","HD":"CD","MCD":"CD","NKE":"CD",
        "LOW":"CD","SBUX":"CD","TJX":"CD","BKNG":"CD","MAR":"CD",
        "F":"CD","GM":"CD","ORLY":"CD","AZO":"CD",
        "PG":"CS","KO":"CS","PEP":"CS","COST":"CS","WMT":"CS",
        "PM":"CS","MO":"CS","CL":"CS","KMB":"CS","GIS":"CS","SYY":"CS",
        "GE":"IND","CAT":"IND","HON":"IND","UNP":"IND","RTX":"IND",
        "LMT":"IND","DE":"IND","BA":"IND","UPS":"IND","FDX":"IND",
        "EMR":"IND","ETN":"IND","ITW":"IND","MMM":"IND","NSC":"IND","WM":"IND",
        "GOOGL":"COM","META":"COM","NFLX":"COM","DIS":"COM","CMCSA":"COM",
        "T":"COM","VZ":"COM","TMUS":"COM","EA":"COM","TTWO":"COM",
        "XOM":"EN","CVX":"EN","COP":"EN","EOG":"EN","SLB":"EN",
        "MPC":"EN","PSX":"EN","VLO":"EN","OXY":"EN","HAL":"EN","DVN":"EN",
        "NEE":"UT","DUK":"UT","SO":"UT","D":"UT","AEP":"UT",
        "EXC":"UT","SRE":"UT","XEL":"UT","ED":"UT","PEG":"UT",
        "PLD":"RE","AMT":"RE","EQIX":"RE","CCI":"RE",
        "PSA":"RE","SPG":"RE","O":"RE","WELL":"RE",
        "LIN":"MAT","APD":"MAT","SHW":"MAT","FCX":"MAT","NEM":"MAT",
        "NUE":"MAT","VMC":"MAT","MLM":"MAT","PPG":"MAT","ECL":"MAT",
        "SPY":"OTHER",
    })
    
    sector_colors: dict[str, str] = field(default_factory=lambda: {
        "IT":"#4C78A8","HC":"#72B7B2","FIN":"#F58518","CD":"#E45756",
        "CS":"#54A24B","IND":"#B279A2","COM":"#FF9DA6","EN":"#9D755D",
        "UT":"#BAB0AC","RE":"#EECA3B","MAT":"#76B7B2","OTHER":"#aaaaaa",
    })

    @property
    def tickers(self) -> list[str]:
        return list(self.universe_with_sectors.keys())


# Default configuration
CONFIG = StrategyConfig()
UNIVERSE = UniverseMetadata()

START_DATE = dt.datetime.today() - dt.timedelta(days=365 * 10)
END_DATE   = dt.datetime.today()

REPORTS_DIR = Path(__file__).parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


# ============================================================
#  HELPERS
# ============================================================
def _to_period_index(s: pd.Series) -> pd.Series:
    if not isinstance(s.index, pd.PeriodIndex):
        s = s.copy()
        s.index = pd.PeriodIndex(s.index, freq='M')
    return s


def _extract_close(raw: pd.DataFrame) -> pd.Series:
    close = raw["Close"] if not isinstance(raw.columns, pd.MultiIndex) \
            else raw["Close"].squeeze()
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close


# ============================================================
#  DATA
# ============================================================
def build_prices(data: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {t: df['Close'] for t, df in data.items()}
    ).dropna(how='all').ffill(limit=2)


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1))


def compute_12_1(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices.shift(1) / prices.shift(13))


def compute_6_1(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices.shift(1) / prices.shift(7))


def compute_3_1(prices: pd.DataFrame) -> pd.DataFrame:
    """3-1 momentum: 3-month (approx) log-return used for short-term acceleration."""
    return np.log(prices.shift(1) / prices.shift(4))


# ------------------------
# Daily / short-term helpers
# ------------------------
def compute_short_term_return(daily_close: pd.Series, window_days: int = 5) -> float:
    """Return over the past `window_days` trading days as a simple pct (not log).
    Expects a pandas Series indexed by Timestamp ordered ascending. Returns np.nan when insufficient data.
    """
    if daily_close is None or len(daily_close.dropna()) < window_days:
        return np.nan
    recent = daily_close.dropna().iloc[-window_days:]
    return float(recent.iloc[-1] / recent.iloc[0] - 1)


def compute_rsi(daily_close: pd.Series, length: int = 14) -> float:
    """Simple RSI implementation (Wilders-like EMA). Returns last RSI value or np.nan."""
    s = daily_close.dropna()
    if len(s) < length + 1:
        return np.nan
    delta = s.diff().dropna()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/length, adjust=False).mean()
    roll_down = down.ewm(alpha=1/length, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


def volume_spike_score(daily_volume: pd.Series, lookback: int = 60) -> float:
    """Score for recent volume spike: ratio of last 5-day avg to lookback avg.
    Returns 1.0 baseline; >1 indicates spike. np.nan on insufficient data."""
    v = daily_volume.dropna()
    if len(v) < max(lookback, 5):
        return np.nan
    recent = v.iloc[-5:].mean()
    hist = v.iloc[-lookback:].mean()
    return float(recent / (hist + 1e-12))


def compute_earnings_revision_proxy(tickers: list) -> pd.Series:
    """Lightweight earnings revision proxy using yfinance info fields.
    This is a fast, best-effort proxy (not a replacement for a real
    analyst-estimate feed). Returns a z-scored Series aligned to
    `tickers` and fills missing with 0.
    """
    out = {}
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            info = tk.info or {}
            eg = info.get('earningsQuarterlyGrowth')
            if eg is None:
                eg = info.get('earningsGrowth') if info.get('earningsGrowth') is not None else 0.0
            out[t] = float(eg) if eg is not None else 0.0
        except Exception:
            out[t] = 0.0

    s = pd.Series(out)
    try:
        zs = zscore(s.fillna(0.0))
    except Exception:
        zs = pd.Series({t: 0.0 for t in tickers})
    return zs.reindex(tickers).fillna(0.0)


def spy_regime(daily_spy_close: pd.Series) -> dict:
    """Return regime dict: trend (SPY > 200d MA) and volatility proxy (21d std).
    Values: {'trend': True/False/np.nan, 'vix_proxy': float}
    """
    s = daily_spy_close.dropna()
    if len(s) == 0:
        return {'trend': np.nan, 'vix_proxy': np.nan}
    ma200 = s.rolling(200).mean()
    try:
        trend = bool(s.iloc[-1] > ma200.iloc[-1]) if not np.isnan(ma200.iloc[-1]) else np.nan
    except Exception:
        trend = np.nan
    vix_proxy = np.nan
    try:
        pct = s.pct_change().dropna()
        if len(pct) >= 21:
            vix_proxy = float(pct.rolling(21).std().iloc[-1])
    except Exception:
        vix_proxy = np.nan
    return {'trend': trend, 'vix_proxy': vix_proxy}


# ============================================================
#  RISK METRICS
# ============================================================
def _calculate_drawdown(returns_list: list[float]) -> float:
    """Calculate current drawdown from a list of returns."""
    if not returns_list:
        return 0.0
    try:
        eq = np.cumprod([1 + r for r in returns_list])
        peak = np.maximum.accumulate(eq)
        return float((peak[-1] - eq[-1]) / (peak[-1] + 1e-12))
    except Exception:
        return 0.0


def _get_drawdown_multiplier(dd: float) -> float:
    """Determine drawdown-based exposure multiplier (soft de-risking)."""
    if dd > 0.30:
        return 0.3
    elif dd > 0.20:
        return 0.5
    elif dd > 0.10:
        return 0.7
    return 1.0


def _get_market_regime_context(
    daily_spy: pd.Series, 
    spy_mom: float, 
    i: int, 
    returns_window: pd.DataFrame
) -> dict:
    """Determine market regime: trend, vix_proxy, and crash_mode."""
    reg = {'trend': np.nan, 'vix_proxy': np.nan}
    try:
        if daily_spy is not None:
            reg = spy_regime(daily_spy)
    except Exception:
        pass
    
    crash_mode = not pd.isna(spy_mom) and spy_mom < 0
    
    # Fallback/enhancement for vix_proxy
    if pd.isna(reg.get('vix_proxy')):
        try:
            reg['vix_proxy'] = float(np.nanmean(returns_window.std())) if not returns_window.empty else 0.02
        except Exception:
            reg['vix_proxy'] = 0.02
            
    return {
        'trend': reg.get('trend'),
        'vix_proxy': reg.get('vix_proxy'),
        'crash_mode': crash_mode
    }


def _compute_ticker_scores(
    tickers: list[str],
    prices: pd.DataFrame,
    mom_12_1_series: pd.Series,
    mom_6_1_series: pd.Series,
    mom_3_1_df: pd.DataFrame,
    daily_ohlcv: dict,
    vol_series: pd.Series,
    rank_now: pd.Series,
    prev_rank: pd.Series,
    i: int,
    config: StrategyConfig = CONFIG
) -> dict:
    """Compute composite alpha scores and metadata for each ticker."""
    eligible = {}
    for t in tickers:
        try:
            if t not in mom_12_1_series.index or t not in mom_6_1_series.index:
                continue
            s12_val = mom_12_1_series[t]
            s6_val = mom_6_1_series[t]
            
            if pd.isna(s12_val) or pd.isna(s6_val) or s12_val <= 0 or s6_val <= config.mom_6_1_veto:
                continue
            
            # Earnings proxy
            earnings_proxy = np.nan
            try:
                p3 = prices[t].pct_change(3).iloc[i]
                p12 = prices[t].pct_change(12).iloc[i]
                if not pd.isna(p3) and not pd.isna(p12):
                    earnings_proxy = float(p3 - p12)
            except Exception:
                pass

            # Daily signals (entry gating only)
            st_ret, rsi = np.nan, np.nan
            try:
                if daily_ohlcv and t in daily_ohlcv:
                    dclose = daily_ohlcv[t]['Close'].dropna()
                    st_ret = compute_short_term_return(dclose, window_days=5)
                    rsi = compute_rsi(dclose, length=14)
            except Exception:
                pass

            # Composite Score components
            alpha1 = float(s6_val)
            alpha2 = 0.0
            if mom_3_1_df is not None and t in mom_3_1_df.columns:
                val31 = mom_3_1_df.iloc[i].get(t, np.nan)
                if not pd.isna(val31):
                    alpha2 = float(val31) - alpha1
            
            rc = 0.0
            if rank_now is not None and prev_rank is not None:
                rc = float(rank_now.get(t, 0.0) - prev_rank.get(t, 0.0))

            comp_score = 0.5 * alpha1 + 0.3 * alpha2 + 0.2 * rc
            
            # Volatility adjustment
            v_adj = vol_series.get(t, np.nan) if isinstance(vol_series, pd.Series) else np.nan
            if not np.isnan(v_adj) and v_adj > 0:
                comp_score = comp_score / (v_adj ** 0.5)

            # Volatility compression expansion
            vol_comp = np.nan
            try:
                if daily_ohlcv and t in daily_ohlcv and 'Close' in daily_ohlcv[t].columns:
                    dclose = daily_ohlcv[t]['Close'].dropna()
                    if len(dclose) >= config.vol_comp_long:
                        dpct = dclose.pct_change().dropna()
                        v_short = dpct.rolling(config.vol_comp_short).std().iloc[-1]
                        v_long = dpct.rolling(config.vol_comp_long).std().iloc[-1]
                        if not np.isnan(v_short) and not np.isnan(v_long) and v_long > 0:
                            vol_comp = float(v_short / (v_long + 1e-12))
                else:
                    m = prices[t].pct_change().dropna()
                    if len(m) >= 12:
                        v_short = m.rolling(3).std().iloc[i] if i < len(m) else m.iloc[-3:].std()
                        v_long = m.rolling(12).std().iloc[i] if i < len(m) else m.iloc[-12:].std()
                        if not pd.isna(v_short) and not pd.isna(v_long) and v_long > 0:
                            vol_comp = float(v_short / (v_long + 1e-12))
            except Exception:
                pass

            if not np.isnan(vol_comp):
                comp_score += (0.3 * float(vol_comp))

            eligible[t] = {
                'score': float(comp_score),
                'mom_12_1': float(s12_val),
                'mom_6_1': float(s6_val),
                'earnings_proxy': earnings_proxy,
                'st_ret': st_ret,
                'rsi': rsi,
            }
        except Exception:
            continue
    return eligible


def _apply_entry_gating(
    candidates: list[str], 
    eligible: dict, 
    regime: dict, 
    current_weights: dict,
    config: StrategyConfig = CONFIG
) -> list[str]:
    """Filter candidates based on entry rules (pullbacks, RSI, momentum)."""
    allowed = []
    for t in candidates:
        ed = eligible[t]
        mom12 = ed.get('mom_12_1', 0.0)
        st_ret = ed.get('st_ret', np.nan)
        rsi = ed.get('rsi', np.nan)

        st_ok = False if np.isnan(st_ret) else (st_ret <= -config.entry_pullback)
        rsi_ok = False if np.isnan(rsi) else (rsi < config.entry_rsi)
        strong_mom = (mom12 > config.strong_mom_threshold)

        if strong_mom:
            allowed.append(t)
            continue

        if t in current_weights and st_ok:
            allowed.append(t)
            continue

        if np.isnan(st_ret) or np.isnan(rsi):
            if strong_mom:
                allowed.append(t)
            continue

        if regime.get('trend') is True:
            if strong_mom or (st_ok and rsi_ok):
                allowed.append(t)
            continue

        if st_ok and rsi_ok:
            allowed.append(t)
            
    return allowed


def _calculate_exposure_scaling(
    regime: dict,
    mean_conv: float,
    avg_eligible_score: float,
    dd_multiplier: float,
    dd: float,
    spy_mom: float,
    config: StrategyConfig = CONFIG
) -> float:
    """Combine signal, volatility, and drawdown into a final exposure multiplier."""
    invest_frac = config.convexity_default_invest
    vproxy = regime.get('vix_proxy', np.nan)
    crash_mode = regime.get('crash_mode', False)
    
    if not pd.isna(vproxy):
        if vproxy >= config.convexity_vix_thresh_crash or crash_mode:
            invest_frac = max(config.convexity_min_invest, 0.3)
        elif vproxy >= config.convexity_vix_thresh_warn:
            invest_frac = 0.6

    overall_signal = max(mean_conv, avg_eligible_score)
    market_vol = vproxy if not pd.isna(vproxy) else 0.02
    
    exposure_factor = float(np.clip((overall_signal / (market_vol * 5 + 1e-12)), 0.2, 1.0))
    
    if regime.get('trend') is True and not pd.isna(spy_mom) and spy_mom > 0:
        exposure_factor = min(1.0, exposure_factor + 0.15)

    combined_factor = 0.5 * dd_multiplier + 0.5 * exposure_factor
    combined_factor = float(np.clip(combined_factor, 0.0, 1.0))
    
    equities_alloc = invest_frac * combined_factor
    
    if regime.get('trend') is False:
        equities_alloc *= 0.5
        
    if dd > config.max_drawdown_limit:
        equities_alloc *= 0.3
        
    return equities_alloc


def _apply_anti_churn(
    new_weights: dict,
    prev_weights: dict,
    last_trade_date: dict,
    date: pd.Timestamp,
    config: StrategyConfig = CONFIG
) -> dict:
    """Enforce min_hold_days and min_weight_change to reduce turnover."""
    if not prev_weights:
        return new_weights
        
    updated_weights = dict(new_weights)
    dt_date = date.to_timestamp() if hasattr(date, 'to_timestamp') else pd.to_datetime(date)
    
    for t in list(prev_weights.keys()):
        last_dt = last_trade_date.get(t)
        if last_dt is None:
            continue
            
        try:
            if (dt_date - last_dt) < pd.Timedelta(days=config.min_hold_days):
                prev_w = prev_weights.get(t, 0.0)
                prop_w = updated_weights.get(t, 0.0)
                if abs(prop_w - prev_w) > config.min_weight_change:
                    updated_weights[t] = prev_w
        except Exception:
            continue
            
    return updated_weights


def compute_full_metrics(strat: pd.Series, bench: pd.Series, config: StrategyConfig = CONFIG) -> dict:
    """Compute comprehensive risk/return metrics for the strategy vs benchmark."""
    strat = strat.dropna()
    bench_aligned = bench.reindex(strat.index).fillna(0)

    cagr = kpi.cagr_from_returns(strat, periods_per_year=12)
    ann_vol = kpi.volatility(strat, periods_per_year=12)
    sharpe = kpi.sharpe_ratio(strat, risk_free_rate=config.risk_free_rate, periods_per_year=12)
    sortino = kpi.sortino_ratio(strat, risk_free_rate=config.risk_free_rate, periods_per_year=12)
    max_dd = kpi.max_drawdown(strat)
    calmar = kpi.calmar_ratio(strat, periods_per_year=12)
    ir = kpi.information_ratio(strat, bench_aligned, periods_per_year=12)
    gain_pain = kpi.gain_pain_ratio(strat)
    max_recovery = kpi.max_recovery_period(strat)

    active = strat - bench_aligned
    te = kpi.volatility(active, periods_per_year=12)
    
    # Add Alpha, Beta, Correlation
    try:
        st_arr, bh_arr = strat.values, bench_aligned.values
        cov_mat = np.cov(st_arr, bh_arr)
        beta = cov_mat[0, 1] / cov_mat[1, 1] if cov_mat[1, 1] > 0 else 0.0
        alpha = (st_arr.mean() - beta * bh_arr.mean()) * 12
        corr = float(np.corrcoef(st_arr, bh_arr)[0, 1])
    except Exception:
        alpha, beta, corr = 0.0, 0.0, 0.0

    return {
        "CAGR (%)": round(cagr * 100, 2),
        "Ann. Vol (%)": round(ann_vol * 100, 2),
        "Sharpe": round(sharpe, 3),
        "Sortino": round(sortino, 3),
        "Calmar": round(calmar, 3),
        "Max Drawdown (%)": round(max_dd * 100, 2),
        "Max Recovery (mo)": max_recovery,
        "Alpha (ann.)": round(alpha, 4),
        "Beta": round(beta, 4),
        "Correlation": round(corr, 4),
        "Info Ratio vs SPY": round(ir, 3),
        "Tracking Error (%)": round(te * 100, 2),
        "Win Rate (%)": round((strat > 0).mean() * 100, 1),
        "Gain/Pain": round(gain_pain, 2),
        "Months": len(strat),
    }


def _handle_pnl_and_weights(
    current_weights: dict,
    returns_row: pd.Series
) -> tuple[float, dict]:
    """Calculate PnL for the period and update weights for mark-to-market."""
    pnl = 0.0
    if not current_weights:
        return 0.0, {}

    # Calculate PnL based on returns of held assets
    pnl = sum(
        returns_row[t] * w
        for t, w in current_weights.items()
        if t in returns_row.index and not pd.isna(returns_row[t])
    )
    
    # Update weights (mark-to-market)
    divisor = (1 + pnl)
    if divisor == 0:
        return pnl, {}
        
    updated_weights = {
        t: w * (1 + returns_row[t]) / divisor
        for t, w in current_weights.items()
        if t in returns_row.index and not pd.isna(returns_row[t])
    }
    return pnl, updated_weights


def _get_portfolio_candidates(
    eligible: dict,
    regime: dict,
    config: StrategyConfig = CONFIG
) -> list[str]:
    """Select the top candidates based on regime and composite score."""
    ranked = sorted(eligible.keys(), key=lambda x: eligible[x]['score'], reverse=True)
    
    local_top_n = config.candidate_size
    if regime.get('trend') is True:
        local_top_n = 3 # Special case from original logic
        
    top_n = max(config.min_positions, min(config.max_positions, min(len(ranked), local_top_n)))
    return ranked[:top_n]


def _calculate_target_allocations(
    allowed: list[str],
    eligible: dict,
    regime: dict,
    returns_window: pd.DataFrame,
    config: StrategyConfig = CONFIG
) -> dict:
    """Determine individual asset allocations based on conviction or risk-parity."""
    scores = pd.Series([eligible[t]['score'] for t in allowed], index=allowed)
    conv_scores = scores.clip(lower=0) ** 1.5
    
    # Correlation penalty
    if config.apply_corr_penalty and len(allowed) > 1:
        try:
            corr_matrix = returns_window[allowed].corr().abs().fillna(0.0)
            for t in conv_scores.index:
                avg_corr = float(corr_matrix[t].mean())
                conv_scores[t] *= (1.0 - min(0.5, avg_corr))
        except Exception:
            pass

    # Basic normalization
    if conv_scores.sum() <= 0:
        return {}
        
    normalized = (conv_scores / conv_scores.sum()).to_dict()
    
    # Risk-parity fallback in bear markets
    if regime.get('trend') is False and len(normalized) > 0:
        vols = returns_window[allowed].std()
        inv = (1.0 / (vols + 1e-12)).reindex(allowed).fillna(0)
        if inv.sum() > 0:
            normalized = (inv / inv.sum()).to_dict()
            
    return normalized


def run_strategy(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    mom_12_1: pd.DataFrame,
    mom_6_1: pd.DataFrame,
    mom_3_1: pd.DataFrame,
    daily_ohlcv: dict = None,
    daily_spy: pd.Series = None,
    config: StrategyConfig = CONFIG,
    universe: UniverseMetadata = UNIVERSE
) -> tuple[pd.Series, pd.DataFrame, dict]:
    """Execute the momentum rebalancing strategy."""
    monthly_returns = []
    current_weights = {}
    prev_weights = {}
    last_raw_optimal = {}
    weights_history = {}
    order_book_rows = []
    last_trade_date = {}

    total_turnover = 0.0
    total_txn_cost = 0.0

    for i in range(len(returns)):
        date = returns.index[i]
        returns_row = returns.iloc[i]

        # 1. Mark-to-market
        pnl, current_weights = _handle_pnl_and_weights(current_weights, returns_row)
        monthly_returns.append(pnl)

        # 2. Risk Context
        dd = _calculate_drawdown(monthly_returns)
        dd_multiplier = _get_drawdown_multiplier(dd)
        
        spy_mom = mom_12_1.iloc[i].get("SPY", np.nan)
        returns_window = returns.iloc[max(0, i - 36):i] # Use 12mo window for vol usually
        regime = _get_market_regime_context(daily_spy, spy_mom, i, returns_window)

        # 3. Ticker Scoring
        rank_now = mom_12_1.iloc[i].rank(pct=True)
        prev_rank = mom_12_1.iloc[i-1].rank(pct=True) if i > 0 else None
        vol = returns_window.std()
        
        eligible = _compute_ticker_scores(
            universe.tickers, prices, mom_12_1.iloc[i], mom_6_1.iloc[i], 
            mom_3_1, daily_ohlcv, vol, rank_now, prev_rank, i, config
        )

        if len(eligible) < config.min_positions:
            weights_history[date] = dict(current_weights)
            continue

        # 4. Portfolio Selection
        candidates = _get_portfolio_candidates(eligible, regime, config)
        allowed = _apply_entry_gating(candidates, eligible, regime, current_weights, config)
        
        if not allowed:
            weights_history[date] = dict(current_weights)
            continue

        # 5. Optimization & Weighing
        target_allocs = _calculate_target_allocations(allowed, eligible, regime, returns_window, config)
        if not target_allocs:
            weights_history[date] = dict(current_weights)
            continue

        # Do-nothing filters
        avg_eligible_score = float(np.nanmean([abs(v['score']) for v in eligible.values()]))
        mean_conv = float(pd.Series(target_allocs).mean() if target_allocs else 0.0)
        
        if avg_eligible_score < config.do_nothing_threshold or mean_conv < config.min_mean_conviction:
            weights_history[date] = dict(current_weights)
            continue

        # 6. Exposure Scaling & Hedging
        equities_alloc = _calculate_exposure_scaling(regime, mean_conv, avg_eligible_score, dd_multiplier, dd, spy_mom, config)
        
        hedge_alloc = 0.0
        vproxy = regime.get('vix_proxy', 0.0)
        if (not pd.isna(vproxy) and vproxy >= config.convexity_vix_thresh_crash) or regime.get('crash_mode'):
            hedge_alloc = config.hedge_crash_alloc
            
        equities_alloc = max(0.0, equities_alloc - hedge_alloc)
        
        # Merge allocations
        new_weights = {t: float(target_allocs.get(t, 0.0) * equities_alloc) for t in target_allocs}
        
        available_hedges = [h for h in config.hedge_tickers if h in prices.columns]
        if hedge_alloc > 0 and available_hedges:
            per_h = float(min(config.max_weight, hedge_alloc / len(available_hedges)))
            for h in available_hedges:
                new_weights[h] = per_h

        # Clip and clean
        new_weights = {t: min(config.max_weight, max(config.min_weight, w)) for t, w in new_weights.items()}

        # 7. Anti-churn & Rebalance Gate
        new_weights = _apply_anti_churn(new_weights, prev_weights, last_trade_date, date, config)
        
        total_invested = sum(new_weights.values())
        if total_invested <= 0:
            weights_history[date] = dict(current_weights)
            continue

        # Drift check
        if last_raw_optimal:
            drift = max(
                abs(target_allocs.get(t, 0) - last_raw_optimal.get(t, 0))
                for t in set(target_allocs) | set(last_raw_optimal)
            )
            if drift < config.rebal_threshold:
                weights_history[date] = dict(current_weights)
                continue

        last_raw_optimal = dict(target_allocs)

        # Smoothing
        if current_weights:
            new_weights = {
                t: config.smoothing * current_weights.get(t, 0) + (1 - config.smoothing) * new_weights.get(t, 0)
                for t in set(current_weights) | set(new_weights)
            }

        # 8. Execution (Order Book & Costs)
        def _get_ob_row(ticker, action, weight):
            return {
                "Date": str(date), "Ticker": ticker, "Sector": universe.universe_with_sectors.get(ticker, "?"),
                "Action": action, "Weight_%": round(weight * 100, 2),
                "Mom_12_1_%": round(eligible.get(ticker, {}).get('mom_12_1', 0) * 100, 2),
                "Price": round(float(prices[ticker].iloc[i]), 2) if ticker in prices.columns else None,
            }

        prev_set, new_set = set(prev_weights), set(new_weights)
        for t in new_set - prev_set:
            order_book_rows.append(_get_ob_row(t, "BUY", new_weights[t]))
        for t in prev_set - new_set:
            order_book_rows.append(_get_ob_row(t, "SELL", 0.0))
        for t in prev_set & new_set:
            delta = new_weights[t] - prev_weights[t]
            if abs(delta) > config.order_book_min_delta:
                order_book_rows.append(_get_ob_row(t, "ADD" if delta > 0 else "TRIM", new_weights[t]))

        turnover = sum(abs(new_weights.get(t, 0) - prev_weights.get(t, 0)) for t in set(prev_weights) | set(new_weights))
        txn_cost = turnover * config.txn_cost_bps / 10_000
        total_turnover += turnover
        total_txn_cost += txn_cost
        monthly_returns[-1] -= txn_cost

        # Update tracking
        dt_date = date.to_timestamp() if hasattr(date, 'to_timestamp') else pd.to_datetime(date)
        for t in set(new_weights.keys()) | set(prev_weights.keys()):
            if abs(new_weights.get(t, 0) - prev_weights.get(t, 0)) > config.order_book_min_delta:
                last_trade_date[t] = dt_date

        current_weights = new_weights
        prev_weights = dict(new_weights)
        weights_history[date] = dict(new_weights)

    logger.info(f"  Total turnover: {total_turnover:.2f}")
    logger.info(f"  Total txn cost: {total_txn_cost*100:.2f}%")

    return pd.Series(monthly_returns, index=returns.index, name="Monthly Return"), pd.DataFrame(order_book_rows), weights_history


# ============================================================
#  ROLLING RISK METRICS
# ============================================================
def rolling_risk_metrics(
    strat_ts: pd.Series,
    spy_ts:   pd.Series,
    window:   int = 24,
    config:   StrategyConfig = CONFIG
) -> pd.DataFrame:
    idx = strat_ts.index
    sharpes, sortinos, calmars, irs, tes = [], [], [], [], []

    for i in range(len(idx)):
        if i < window:
            sharpes.append(np.nan);  sortinos.append(np.nan)
            calmars.append(np.nan);  irs.append(np.nan)
            tes.append(np.nan)
            continue
        s = strat_ts.iloc[i - window: i]
        b = spy_ts.reindex(s.index).fillna(0)
        sharpes.append(kpi.sharpe_ratio(s,  risk_free_rate=config.risk_free_rate, periods_per_year=12))
        sortinos.append(kpi.sortino_ratio(s, risk_free_rate=config.risk_free_rate, periods_per_year=12))
        calmars.append(kpi.calmar_ratio(s,  periods_per_year=12))
        irs.append(kpi.information_ratio(s, b, periods_per_year=12))
        tes.append(kpi.volatility(s - b, periods_per_year=12))

    return pd.DataFrame(
        {"Sharpe": sharpes, "Sortino": sortinos,
         "Calmar": calmars, "IR": irs, "TE": tes},
        index=idx,
    )


# ============================================================
#  DASHBOARD
# ============================================================

def _resample_spy_to_monthly(spy_daily_close: pd.Series) -> pd.Series:
    """
    Convert daily SPY prices to monthly prices on month-START timestamps.
    vs_benchmark() does pct_change().reindex(self._returns.index).
    When self._returns is monthly (155 rows), benchmark_series must also be
    monthly prices so pct_change() yields monthly returns that reindex cleanly.
    Uses month-start timestamps to match Period('M').to_timestamp() output.
    """
    try:
        monthly = spy_daily_close.resample('ME').last().dropna()
    except ValueError:
        monthly = spy_daily_close.resample('M').last().dropna()
    monthly.index = monthly.index.to_period('M').to_timestamp()
    return monthly


def _add_equity_traces(fig: go.Figure, equity: pd.Series, spy_eq: pd.Series, row: int, col: int) -> None:
    """Add strategy and benchmark equity curves to the dashboard."""
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity.values, name="Strategy",
        line=dict(color="#4C78A8", width=2.5),
        hovertemplate="%{x|%b %Y}<br>$%{y:,.0f}<extra>Strategy</extra>",
    ), row=row, col=col)
    
    fig.add_trace(go.Scatter(
        x=spy_eq.index, y=spy_eq.values, name="SPY",
        line=dict(color="#F58518", width=1.8, dash="dot"),
        hovertemplate="%{x|%b %Y}<br>$%{y:,.0f}<extra>SPY</extra>",
    ), row=row, col=col)
    
    for series, color in [(equity, "#4C78A8"), (spy_eq, "#F58518")]:
        fig.add_annotation(
            x=series.index[-1], y=float(series.iloc[-1]),
            text=f"  ${float(series.iloc[-1]):,.0f}",
            showarrow=False, font=dict(color=color, size=11),
            row=row, col=col,
        )


def _add_drawdown_trace(fig: go.Figure, drawdown: pd.Series, row: int, col: int) -> None:
    """Add the underwater drawdown area plot to the dashboard."""
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values,
        fill="tozeroy", fillcolor="rgba(229,115,115,0.25)",
        line=dict(color="#E45756", width=1.5), name="Drawdown",
        hovertemplate="%{x|%b %Y}<br>%{y:.1f}%<extra>Drawdown</extra>",
    ), row=row, col=col)


def _add_heatmap_trace(fig: go.Figure, heat_pivot: pd.DataFrame, row: int, col: int) -> None:
    """Add the monthly returns heatmap to the dashboard."""
    valid = heat_pivot.values[~np.isnan(heat_pivot.values)]
    zmax = max(abs(valid.max()), abs(valid.min())) if len(valid) else 1.0
    
    fig.add_trace(go.Heatmap(
        z=heat_pivot.values,
        x=heat_pivot.columns.tolist(),
        y=[str(y) for y in heat_pivot.index.tolist()],
        colorscale=[[0.0, "#c0392b"], [0.5, "#f7f7f7"], [1.0, "#27ae60"]],
        zmid=0, zmin=-zmax, zmax=zmax,
        text=np.round(heat_pivot.values, 1), texttemplate="%{text}",
        textfont=dict(size=9),
        colorbar=dict(len=0.28, y=0.50, thickness=12, title="%"),
        hovertemplate="<b>%{y} %{x}</b><br>%{z:.2f}%<extra></extra>",
        name="Monthly Ret",
    ), row=row, col=col)


def _add_rolling_metrics_traces(fig: go.Figure, roll: pd.DataFrame, row: int, col: int) -> None:
    """Add rolling risk/return ratios to the dashboard."""
    for metric, color in [("Sharpe", "#4C78A8"), ("Sortino", "#72B7B2"), ("Calmar", "#F58518")]:
        fig.add_trace(go.Scatter(
            x=roll.index, y=roll[metric].values,
            line=dict(color=color, width=1.8), name=metric,
            hovertemplate=f"%{{x|%b %Y}}<br>{metric}: %{{y:.2f}}<extra></extra>",
        ), row=row, col=col)
    fig.add_hline(y=0, line=dict(color="gray", dash="dot", width=1), row=row, col=col)
    fig.add_hline(y=1, line=dict(color="#27ae60", dash="dash", width=1), row=row, col=col)


def _add_portfolio_composition_traces(
    fig: go.Figure, 
    wh_df: pd.DataFrame, 
    row: int, 
    col: int,
    universe: UniverseMetadata = UNIVERSE
) -> None:
    """Add the stacked sector allocation plot to the dashboard."""
    sector_wh = pd.DataFrame(index=wh_df.index)
    for sec in sorted(set(universe.universe_with_sectors.values())):
        tks = [t for t in wh_df.columns if universe.universe_with_sectors.get(t) == sec]
        if tks:
            sector_wh[sec] = wh_df[tks].sum(axis=1)
            
    for sec in sector_wh.columns:
        color = universe.sector_colors.get(sec, "#aaaaaa")
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        fig.add_trace(go.Scatter(
            x=sector_wh.index, y=sector_wh[sec].values,
            stackgroup="one", name=sec,
            line=dict(width=0.5, color=color),
            fillcolor=f"rgba({r},{g},{b},0.7)",
            hovertemplate=f"<b>{sec}</b><br>%{{x|%b %Y}}<br>%{{y:.1f}}%<extra></extra>",
            legendgroup=sec,
        ), row=row, col=col)


def _add_risk_summary_traces(fig: go.Figure, roll: pd.DataFrame, metrics: dict, row: int, col: int) -> None:
    """Add the rolling IR, TE, and a text summary annotation to the dashboard."""
    ir_colors = ["#27ae60" if v >= 0 else "#c0392b" for v in roll["IR"].fillna(0)]
    
    fig.add_trace(go.Bar(
        x=roll.index, y=roll["IR"].values,
        marker_color=ir_colors, name="Rolling IR vs SPY",
        hovertemplate="%{x|%b %Y}<br>IR: %{y:.2f}<extra></extra>",
        opacity=0.75,
    ), row=row, col=col)
    
    fig.add_trace(go.Scatter(
        x=roll.index, y=(roll["TE"] * 100).values,
        line=dict(color="#9D755D", width=1.5, dash="dot"),
        name="Tracking Error (%)",
        hovertemplate="%{x|%b %Y}<br>TE: %{y:.1f}%<extra></extra>",
        yaxis="y6",
    ), row=row, col=col, secondary_y=True)
    
    fig.add_hline(y=0, line=dict(color="gray", dash="dot", width=1), row=row, col=col)

    metric_lines = [
        "<b>Risk-Adjusted Summary</b>",
        f"CAGR:           {metrics['CAGR (%)']:.1f}%",
        f"Ann. Vol:        {metrics['Ann. Vol (%)']:.1f}%",
        f"Sharpe:          {metrics['Sharpe']:.2f}",
        f"Sortino:          {metrics['Sortino']:.2f}",
        f"Calmar:          {metrics['Calmar']:.2f}",
        f"Max DD:         {metrics['Max Drawdown (%)']:.1f}%",
        f"Max Recovery: {metrics['Max Recovery (mo)']}mo",
        f"IR vs SPY:       {metrics['Info Ratio vs SPY']:.2f}",
        f"Tracking Err:   {metrics['Tracking Error (%)']:.1f}%",
        f"Win Rate:        {metrics['Win Rate (%)']:.0f}%",
        f"Gain/Pain:       {metrics['Gain/Pain']:.2f}",
    ]
    
    fig.add_annotation(
        x=0.99, y=0.12, xref="paper", yref="paper",
        text="<br>".join(metric_lines),
        align="left", showarrow=False,
        font=dict(size=10, family="monospace"),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#cccccc", borderwidth=1,
        xanchor="right",
    )


def build_dashboard(
    strategy_returns: pd.Series,
    order_book_df:    pd.DataFrame,
    weights_history:  dict,
    spy_returns:      pd.Series,
    metrics:          dict,
) -> go.Figure:
    """Construct the interactive Plotly dashboard for strategy analysis."""
    # Data Preparation
    strategy_returns = _to_period_index(strategy_returns)
    spy_returns      = _to_period_index(spy_returns)
    spy_aligned      = spy_returns.reindex(strategy_returns.index, method='ffill').fillna(0)

    strat_ts       = strategy_returns.copy()
    strat_ts.index = strategy_returns.index.to_timestamp()
    spy_ts         = spy_aligned.copy()
    spy_ts.index   = spy_aligned.index.to_timestamp()

    equity   = (1 + strat_ts).cumprod() * 100_000
    spy_eq   = (1 + spy_ts).cumprod()   * 100_000
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max * 100

    df_ret          = strat_ts.to_frame("ret")
    df_ret["year"]  = df_ret.index.year
    df_ret["month"] = df_ret.index.month
    heat_pivot      = df_ret.pivot(index="year", columns="month", values="ret") * 100
    heat_pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                          "Jul","Aug","Sep","Oct","Nov","Dec"]

    roll = rolling_risk_metrics(strat_ts, spy_ts, window=24)

    wh_ts = {
        k.to_timestamp() if isinstance(k, pd.Period) else k: v
        for k, v in weights_history.items()
    }
    wh_df = pd.DataFrame(wh_ts).T.fillna(0) * 100

    # Initialize Figure
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "📈 Equity Curve vs SPY",
            "🌊 Underwater Drawdown",
            "📅 Monthly Returns Heatmap (%)",
            "📊 Rolling Sharpe / Sortino / Calmar (24-mo)",
            "🥧 Portfolio Composition Over Time",
            "📋 Risk Summary + Rolling IR vs SPY",
        ),
        row_heights=[0.33, 0.33, 0.34],
        vertical_spacing=0.10, horizontal_spacing=0.08,
        specs=[
            [{"type": "xy"},  {"type": "xy"}],
            [{"type": "xy"},  {"type": "xy"}],
            [{"type": "xy"},  {"type": "xy", "secondary_y": True}],
        ],
    )

    # Populate Subplots
    _add_equity_traces(fig, equity, spy_eq, row=1, col=1)
    _add_drawdown_trace(fig, drawdown, row=1, col=2)
    _add_heatmap_trace(fig, heat_pivot, row=2, col=1)
    _add_rolling_metrics_traces(fig, roll, row=2, col=2)
    _add_portfolio_composition_traces(fig, wh_df, row=3, col=1)
    _add_risk_summary_traces(fig, roll, metrics, row=3, col=2)

    # Layout Customization
    fig.update_layout(
        title=dict(
            text="<b>Markowitz Momentum Portfolio — Risk-Adjusted Dashboard</b>",
            font=dict(size=20), x=0.5,
        ),
        height=1350, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1, font=dict(size=10)),
        margin=dict(l=60, r=60, t=80, b=40),
        hovermode="x unified", barmode="relative",
    )
    
    fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1, tickprefix="$", tickformat=",")
    fig.update_yaxes(title_text="Drawdown (%)",       row=1, col=2)
    fig.update_yaxes(title_text="Return (%)",         row=2, col=1)
    fig.update_yaxes(title_text="Ratio",              row=2, col=2)
    fig.update_yaxes(title_text="Allocation (%)",     row=3, col=1)
    fig.update_yaxes(title_text="Info Ratio",         row=3, col=2)
    fig.update_yaxes(title_text="Tracking Error (%)", row=3, col=2, secondary_y=True)

    return fig


def _print_risk_metrics_comparison(metrics: dict, spy_metrics: dict) -> None:
    """Print a formatted comparison table of risk metrics."""
    print("\n" + "=" * 62)
    print("  RISK-ADJUSTED METRICS COMPARISON")
    print("=" * 62)
    print(f"  {'Metric':<26} {'Strategy':>12} {'SPY':>10}")
    print(f"  {'-'*26} {'-'*12} {'-'*10}")
    for k in metrics:
        print(f"  {k:<26} {str(metrics[k]):>12} {str(spy_metrics.get(k, '—')):>10}")
    print("=" * 62 + "\n")


def _save_outputs(order_book_df: pd.DataFrame, metrics: dict, spy_metrics: dict) -> None:
    """Save strategy outputs to CSV files."""
    ob_path = REPORTS_DIR / "order_book.csv"
    order_book_df.to_csv(ob_path, index=False)
    logger.info(f"  ✅ Order book saved → {ob_path}")

    metrics_path = REPORTS_DIR / "metrics.csv"
    pd.DataFrame([metrics, spy_metrics], index=["Strategy", "SPY"]).to_csv(metrics_path)
    logger.info(f"  ✅ Metrics saved    → {metrics_path}")


def main(config: StrategyConfig = CONFIG, universe: UniverseMetadata = UNIVERSE) -> None:
    """Main execution flow for the rebalancing strategy."""
    logger.info("=" * 62)
    logger.info("  Monthly Rebalancing: Risk-Adjusted Optimisation")
    logger.info("=" * 62)
    logger.info(f"\n  Target: Robust Momentum + Volatility Weighting")
    logger.info(f"  Candidates: top {config.candidate_size}  |  bounds: [{config.min_weight*100:.0f}%, {config.max_weight*100:.0f}%]")
    logger.info(f"  Rebalance threshold: {config.rebal_threshold*100:.0f}% drift")
    logger.info(f"  Weight Smoothing: {config.smoothing}")
    logger.info(f"  Txn cost: {config.txn_cost_bps} bps\n")

    # 1. Benchmark Data
    logger.info("  Fetching benchmarks...")
    spy_start = END_DATE - dt.timedelta(days=365 * 14)
    spy_raw = yf.download("SPY", start=spy_start, end=END_DATE, interval="1mo", auto_adjust=True, progress=False)
    spy_price = _extract_close(spy_raw)
    spy_rets = _to_period_index(np.log(spy_price / spy_price.shift(1)).dropna())
    
    spy_daily_raw = yf.download("SPY", start=START_DATE, end=END_DATE, interval="1d", auto_adjust=True, progress=False)
    spy_daily_close = _extract_close(spy_daily_raw).dropna()
    spy_monthly_prices = _resample_spy_to_monthly(spy_daily_close)

    # 2. Universe Data
    logger.info("  Fetching universe data...")
    from data_ingestion.data_store import load_universe_data, update_universe_data
    update_universe_data(universe.tickers, start=START_DATE, end=END_DATE, interval='1mo')
    ohlcv = load_universe_data(universe.tickers, interval='1mo')
    prices = build_prices(ohlcv)
    returns = compute_returns(prices).dropna(how='all')
    prices = prices.reindex(returns.index)

    # Filtering
    returns.index = pd.PeriodIndex(returns.index, freq='M')
    prices.index = pd.PeriodIndex(prices.index, freq='M')
    good_tickers = prices.notna().mean()[lambda x: x > 0.80].index
    prices, returns = prices[good_tickers], returns[good_tickers]

    if len(returns) < config.min_backtest_months:
        logger.warning(f"⚠️  Only {len(returns)} months — need ≥ {config.min_backtest_months}. Exiting.")
        return

    # 3. Strategy Execution
    logger.info("  Running strategy ...\n")
    update_universe_data(universe.tickers, start=START_DATE, end=END_DATE, interval='1d')
    daily_ohlcv = load_universe_data(universe.tickers, interval='1d')
    daily_spy = _extract_close(daily_ohlcv.get('SPY', spy_daily_raw)).dropna()

    mom_12_1 = compute_12_1(prices)
    mom_6_1 = compute_6_1(prices)
    mom_3_1 = compute_3_1(prices)

    strategy_returns, order_book_df, weights_history = run_strategy(
        prices, returns, mom_12_1, mom_6_1, mom_3_1, daily_ohlcv, daily_spy, config, universe
    )

    # 4. VBT Backtest
    logger.info("\n  Running VBT Backtest engine...")
    weights_df = pd.DataFrame.from_dict(weights_history, orient="index").sort_index()
    weights_df = weights_df.reindex(prices.index, method="ffill").fillna(0.0)
    weights_df.index = pd.to_datetime([d.to_timestamp() if hasattr(d, 'to_timestamp') else d for d in weights_df.index])
    
    vbt_prices = prices.copy()
    vbt_prices.index = pd.to_datetime([d.to_timestamp() if hasattr(d, 'to_timestamp') else d for d in vbt_prices.index])

    bt = VBTBacktester(close=vbt_prices, freq='30D', init_cash=config.cash, commission=config.commission)
    bt.run_from_weights(weights_df)
    
    # Analyze
    bt._returns = bt._portfolio.returns(group_by=True)
    if hasattr(bt._returns, 'columns'): bt._returns = bt._returns.iloc[:, 0]
    
    bt.monte_carlo(n_simulations=1000, print_report=True)
    bt.walk_forward(n_splits=5, print_report=True)
    bt.stress_testing(print_report=True)
    bt.deflated_sharpe(n_trials=1, print_report=True)
    bt.trade_analysis(print_report=True)
    bt.regime_analysis(print_report=True)
    bt.risk_metrics(print_report=True)
    bt.kelly_sizing(print_report=True)

    # 5. Final Reporting
    spy_simple_aligned = (np.exp(spy_rets) - 1).reindex(strategy_returns.index, method='ffill').fillna(0)
    strat_simple_aligned = np.exp(strategy_returns) - 1
    
    metrics = compute_full_metrics(strat_simple_aligned, spy_simple_aligned, config)
    spy_metrics = compute_full_metrics(spy_simple_aligned, spy_simple_aligned, config)
    
    _print_risk_metrics_comparison(metrics, spy_metrics)
    _save_outputs(order_book_df, metrics, spy_metrics)

    # Holdout validation
    try:
        ho = strategy_returns.dropna().iloc[-24:]
        if len(ho) >= 6:
            spy_ho = spy_rets.reindex(ho.index, method='ffill').fillna(0)
            ho_m = compute_full_metrics((np.exp(ho) - 1), (np.exp(spy_ho) - 1), config)
            logger.info("\n  🔒 Frozen Holdout (last 24 months) Metrics")
            for k, v in ho_m.items(): logger.info(f"    {k}: {v}")
    except Exception: pass

    logger.info("\n  Building dashboard...")
    fig = build_dashboard(strategy_returns, order_book_df, weights_history, spy_rets, metrics)
    dash_path = REPORTS_DIR / "portfolio_dashboard.html"
    fig.write_html(str(dash_path), include_plotlyjs="cdn", full_html=True)
    logger.info(f"  ✅ Dashboard saved  → {dash_path}\n")


if __name__ == '__main__':
    try:
        main()
    finally:
        try:
            from joblib.externals.loky import get_reusable_executor
            get_reusable_executor().shutdown(wait=True)
        except (ImportError, AttributeError):
            pass
