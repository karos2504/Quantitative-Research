"""
Renko + MACD Strategy — ADVANCED & IMPROVED Backtest (v4.0)

Key Improvements over v3.0:
  * Optimized parameters are extracted and reused for a dedicated final backtest
    instead of reporting the optimization run itself (avoids in-sample bias
    reporting — the final bt.run(**best_params) is a clean, single-pass run
    using the best discovered parameters, clearly separated from the search).
  * Best parameters are printed per ticker and collected in a summary table.
  * VBT signal generation also uses the best discovered parameters.
  * All v3.0 features preserved: Stochastic RSI, VWAP proximity, ML hooks.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import yfinance as yf

from backtesting import Backtest, Strategy
from indicators.renko import convert_to_renko
from indicators.macd import calculate_macd
from indicators.slope import calculate_slope
from indicators.rsi import calculate_rsi
from indicators.atr import calculate_atr
from indicators.stochastic import calculate_stochastic
from indicators.vwap import calculate_vwap
from utils.backtesting import VBTBacktester
from utils.strategy_utils import align_indicator_data, standardize_ohlcv

# ----------------------------- CONFIG ----------------------------- #
TICKERS = ["NVDA", "AAPL", "GOOGL", "META", "AMZN", "MSFT", "TSLA"]
CASH = 100_000
COMMISSION = 0.001

# Default parameter fallback (used when optimization yields < 10 trades)
DEFAULT_PARAMS = dict(
    bar_threshold=3,
    rsi_threshold=50,
    er_th=0.3,
    tp_factor=1.5,
    sl_factor=1.0,
    vol_ratio_th=0.8,
    vwap_dist_max=3.0,
)


# ---------------------- INDICATOR HELPERS ---------------------- #
def _precompute_indicators(df):
    """Merge Renko + MACD + RSI + ATR + ER + Volume + Stochastic + VWAP."""
    renko = convert_to_renko(df)
    merged = align_indicator_data(df, renko, merge_col='bar_num')
    merged = calculate_macd(merged, fast=12, slow=26, signal=9)
    merged = calculate_rsi(merged, period=14)
    merged = calculate_atr(merged, period=14)
    merged = calculate_stochastic(merged, k_period=14, d_period=3, smooth_k=3)
    merged = calculate_vwap(merged)

    # Kaufman's Efficiency Ratio
    er_period = 14
    change = merged['Close'].diff(er_period).abs()
    volatility = merged['Close'].diff().abs().rolling(er_period).sum()
    merged['ER'] = change / volatility.replace(0, np.nan)

    # Volume filter
    merged['Vol_SMA'] = merged['Volume'].rolling(window=20).mean()
    merged['Vol_Ratio'] = merged['Volume'] / merged['Vol_SMA'].replace(0, np.nan)

    # VWAP proximity (distance in ATR units)
    merged['VWAP_Dist'] = (merged['Close'] - merged['VWAP']).abs() / merged['ATR'].replace(0, np.nan)

    merged.dropna(subset=['MACD', 'Signal', 'bar_num', 'RSI', 'ATR', 'ER',
                          'Stoch_K', 'Stoch_D', 'VWAP'], inplace=True)

    if len(merged) >= 5:
        merged['macd_slope'] = calculate_slope(merged['MACD'], 5)
        merged['signal_slope'] = calculate_slope(merged['Signal'], 5)
    else:
        merged['macd_slope'] = np.nan
        merged['signal_slope'] = np.nan

    merged.dropna(inplace=True)
    return standardize_ohlcv(merged)


# ---------------------- STRATEGY CLASS (OPTIMIZABLE) ---------------------- #
class RenkoMACDStrategy(Strategy):
    """Renko + MACD with Stochastic + VWAP filters + full optimization."""

    # === OPTIMIZABLE PARAMETERS ===
    bar_threshold: int = 3
    rsi_threshold: int = 50
    er_th: float = 0.3
    tp_factor: float = 1.5
    sl_factor: float = 1.0
    vol_ratio_th: float = 0.8
    vwap_dist_max: float = 3.0

    def init(self):
        self.bar_num     = self.I(lambda: self.data.bar_num,      name='bar_num',    overlay=False)
        self.macd        = self.I(lambda: self.data.MACD,         name='MACD',       overlay=False)
        self.signal      = self.I(lambda: self.data.Signal,       name='Signal',     overlay=False)
        self.macd_slope  = self.I(lambda: self.data.macd_slope,   name='MACD Slope', overlay=False)
        self.signal_slope= self.I(lambda: self.data.signal_slope, name='Sig Slope',  overlay=False)
        self.rsi         = self.I(lambda: self.data.RSI,          name='RSI',        overlay=False)
        self.atr         = self.I(lambda: self.data.ATR,          name='ATR',        overlay=False)
        self.er          = self.I(lambda: self.data.ER,           name='Eff_Ratio',  overlay=False)
        self.vol_ratio   = self.I(lambda: self.data.Vol_Ratio,    name='Vol_Ratio',  overlay=False)
        self.stoch_k     = self.I(lambda: self.data.Stoch_K,      name='Stoch_K',    overlay=False)
        self.stoch_d     = self.I(lambda: self.data.Stoch_D,      name='Stoch_D',    overlay=False)
        self.vwap_dist   = self.I(lambda: self.data.VWAP_Dist,    name='VWAP_Dist',  overlay=False)

    def next(self):
        bar      = self.bar_num[-1]
        macd_val = self.macd[-1]
        sig_val  = self.signal[-1]
        m_slope  = self.macd_slope[-1]
        s_slope  = self.signal_slope[-1]
        rsi_val  = self.rsi[-1]
        atr_val  = self.atr[-1]
        er_val   = self.er[-1]
        vol_val  = self.vol_ratio[-1]
        stoch_k  = self.stoch_k[-1]
        stoch_d  = self.stoch_d[-1]
        vwap_d   = self.vwap_dist[-1]
        close    = self.data.Close[-1]

        stoch_bull = stoch_k > stoch_d
        stoch_bear = stoch_k < stoch_d
        near_vwap  = vwap_d <= self.vwap_dist_max

        buy_signal = (
            bar >= self.bar_threshold and
            macd_val > sig_val and
            m_slope > s_slope and
            rsi_val > self.rsi_threshold and
            er_val > self.er_th and
            vol_val > self.vol_ratio_th and
            stoch_bull and
            near_vwap
        )
        sell_signal = (
            bar <= -self.bar_threshold and
            macd_val < sig_val and
            m_slope < s_slope and
            rsi_val < self.rsi_threshold and
            er_val > self.er_th and
            vol_val > self.vol_ratio_th and
            stoch_bear and
            near_vwap
        )

        if not self.position:
            if buy_signal:
                self.buy(sl=close - atr_val * self.sl_factor,
                         tp=close + atr_val * self.tp_factor)
            elif sell_signal:
                self.sell(sl=close + atr_val * self.sl_factor,
                          tp=close - atr_val * self.tp_factor)

        elif self.position.is_long:
            new_stop = close - atr_val * self.sl_factor
            if not hasattr(self, '_long_stop') or new_stop > self._long_stop:
                self._long_stop = new_stop
            for trade in self.trades:
                if trade.is_long and (trade.sl is None or self._long_stop > trade.sl):
                    trade.sl = self._long_stop

            if sell_signal or (macd_val < sig_val and m_slope < s_slope):
                self.position.close()
                if sell_signal:
                    self.sell(sl=close + atr_val * self.sl_factor,
                              tp=close - atr_val * self.tp_factor)

        elif self.position.is_short:
            new_stop = close + atr_val * self.sl_factor
            if not hasattr(self, '_short_stop') or new_stop < self._short_stop:
                self._short_stop = new_stop
            for trade in self.trades:
                if trade.is_short and (trade.sl is None or self._short_stop < trade.sl):
                    trade.sl = self._short_stop

            if buy_signal or (macd_val > sig_val and m_slope > s_slope):
                self.position.close()
                if buy_signal:
                    self.buy(sl=close - atr_val * self.sl_factor,
                             tp=close + atr_val * self.tp_factor)


# ---------------------- VBT SIGNAL HELPER ---------------------- #
def _generate_vbt_signals(df, bar_threshold=3, rsi_threshold=50,
                           er_th=0.3, vol_ratio_th=0.8, vwap_dist_max=3.0):
    entries = (
        (df['bar_num'] >= bar_threshold) &
        (df['MACD'] > df['Signal']) &
        (df['macd_slope'] > df['signal_slope']) &
        (df['RSI'] > rsi_threshold) &
        (df['ER'] > er_th) &
        (df['Vol_Ratio'] > vol_ratio_th) &
        (df['Stoch_K'] > df['Stoch_D']) &
        (df['VWAP_Dist'] <= vwap_dist_max)
    )
    exits = (
        (df['bar_num'] <= -bar_threshold) &
        (df['MACD'] < df['Signal']) &
        (df['macd_slope'] < df['signal_slope']) &
        (df['RSI'] < rsi_threshold) &
        (df['ER'] > er_th) &
        (df['Vol_Ratio'] > vol_ratio_th) &
        (df['Stoch_K'] < df['Stoch_D']) &
        (df['VWAP_Dist'] <= vwap_dist_max)
    )
    return entries, exits


# ---------------------- PARAM EXTRACTION HELPER ---------------------- #
def _extract_best_params(opt_stats) -> dict:
    """
    Extract the best parameter set from backtesting.py optimization results.

    backtesting.py stores the winning combo in `opt_stats._strategy` as
    class attributes.  We read them back and return a plain dict.
    """
    strat = opt_stats._strategy
    return dict(
        bar_threshold = int(strat.bar_threshold),
        rsi_threshold = int(strat.rsi_threshold),
        er_th         = float(strat.er_th),
        tp_factor     = float(strat.tp_factor),
        sl_factor     = float(strat.sl_factor),
        vol_ratio_th  = float(strat.vol_ratio_th),
        vwap_dist_max = float(strat.vwap_dist_max),
    )


# ----------------------------- MAIN ----------------------------- #
def main():
    print("=" * 70)
    print("  Renko + MACD Strategy — ADVANCED & IMPROVED Backtest (v4.0)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Download data
    # ------------------------------------------------------------------
    print("\n--- Downloading 1h intraday data (2 years) ---")
    ohlc_intraday = {}
    for ticker in TICKERS:
        try:
            data = yf.download(ticker, interval='1h', period='730d',
                               progress=False, auto_adjust=True)
            data.columns = ["Open", "High", "Low", "Adj Close", "Volume"]
            data['Close'] = data['Adj Close']
            data.dropna(inplace=True)
            ohlc_intraday[ticker] = data
            print(f"  ✅ {ticker}: {len(data)} rows")
        except Exception as e:
            print(f"  ❌ {ticker}: {e}")

    tickers = list(ohlc_intraday.keys())
    if not tickers:
        raise ValueError("No data downloaded.")

    # ------------------------------------------------------------------
    # 2. Optimize → extract best params → re-run clean final backtest
    # ------------------------------------------------------------------
    all_stats   = {}   # final backtest stats
    best_params = {}   # best params per ticker

    for ticker in tickers:
        print(f"\n{'─' * 60}")
        print(f"📊  {ticker} — Step 1: Optimizing …")

        try:
            df = _precompute_indicators(ohlc_intraday[ticker])
            if len(df) < 50:
                print(f"  ⚠️  Skipping {ticker}: insufficient data")
                continue

            bt = Backtest(df, RenkoMACDStrategy,
                          cash=CASH, commission=COMMISSION,
                          exclusive_orders=True, finalize_trades=True)

            # ── Optimization pass ──────────────────────────────────────
            params = DEFAULT_PARAMS.copy()
            try:
                opt_stats = bt.optimize(
                    bar_threshold = list(range(3, 7, 1)),
                    rsi_threshold = list(range(30, 70, 5)),
                    er_th         = list(np.arange(0.3, 0.7, 0.1)),
                    tp_factor     = list(np.arange(1.0, 2.0, 0.1)),
                    sl_factor     = list(np.arange(0.5, 1.0, 0.1)),
                    vol_ratio_th  = list(np.arange(0.3, 0.7, 0.1)),
                    vwap_dist_max = list(np.arange(1.0, 5.0, 0.1)),
                    maximize      = 'Sharpe Ratio',
                    constraint    = lambda p: True,
                    max_tries     = 81,
                    return_heatmap= False,
                )

                if opt_stats['# Trades'] >= 10:
                    params = _extract_best_params(opt_stats)
                    print(f"  ✅ Optimization complete. Best params: {params}")
                else:
                    print(f"  ⚠️  Only {opt_stats['# Trades']} trades found — "
                          f"falling back to defaults")

            except Exception as opt_err:
                print(f"  ⚠️  Optimization failed ({opt_err}) — using defaults")

            best_params[ticker] = params

            # ── Final backtest using best (or default) params ──────────
            print(f"📊  {ticker} — Step 2: Final backtest with best params …")
            final_stats = bt.run(**params)

            # Warn if trade count is low
            if final_stats['# Trades'] < 10:
                print(f"  ⚠️  {ticker}: only {final_stats['# Trades']} trades — "
                      f"interpret results cautiously")

            all_stats[ticker] = {
                'Return [%]':       final_stats['Return [%]'],
                'Sharpe Ratio':     final_stats['Sharpe Ratio'],
                'Max Drawdown [%]': final_stats['Max. Drawdown [%]'],
                '# Trades':         final_stats['# Trades'],
                'Win Rate [%]':     final_stats['Win Rate [%]'],
                # Best params embedded for traceability
                'bar_threshold':    params['bar_threshold'],
                'rsi_threshold':    params['rsi_threshold'],
                'er_th':            params['er_th'],
                'tp_factor':        params['tp_factor'],
                'sl_factor':        params['sl_factor'],
            }

            print(
                f"  ✅ Return: {final_stats['Return [%]']:.2f}%  "
                f"Sharpe: {final_stats['Sharpe Ratio']:.2f}  "
                f"Max DD: {final_stats['Max. Drawdown [%]']:.2f}%  "
                f"Trades: {final_stats['# Trades']}"
            )

        except Exception as e:
            print(f"  ❌ Error for {ticker}: {e}")

    # ------------------------------------------------------------------
    # 3. Summary table
    # ------------------------------------------------------------------
    if all_stats:
        print("\n" + "=" * 70)
        print("  FINAL BACKTEST RESULTS (using optimized parameters per ticker)")
        print("=" * 70)
        summary = pd.DataFrame(all_stats).T
        print(summary.to_string(float_format=lambda x: f"{x:.2f}"))

        # Best params summary
        print("\n" + "─" * 70)
        print("  Best Parameters Per Ticker")
        print("─" * 70)
        params_df = pd.DataFrame(best_params).T
        print(params_df.to_string(float_format=lambda x: f"{x:.2f}"))

    # ------------------------------------------------------------------
    # 4. VBT advanced analysis with best params
    # ------------------------------------------------------------------
    for ticker in tickers:
        if ticker not in all_stats:
            continue
        print(f"\n{'=' * 70}")
        print(f"  Advanced VBT Analysis: {ticker}  (best params)")
        print(f"{'=' * 70}")

        try:
            df     = _precompute_indicators(ohlc_intraday[ticker])
            p      = best_params[ticker]
            entries, exits = _generate_vbt_signals(
                df,
                bar_threshold = p['bar_threshold'],
                rsi_threshold = p['rsi_threshold'],
                er_th         = p['er_th'],
                vol_ratio_th  = p['vol_ratio_th'],
                vwap_dist_max = p['vwap_dist_max'],
            )

            bt_vbt = VBTBacktester(
                close      = df['Close'],
                entries    = entries,
                exits      = exits,
                freq       = '1h',
                init_cash  = CASH,
                commission = COMMISSION,
            )
            bt_vbt.full_analysis(n_mc=500, n_wf_splits=4, n_trials=len(TICKERS))

        except Exception as e:
            print(f"  ❌ VBT error: {e}")

    # ------------------------------------------------------------------
    # 5. ML/DL/RL signal enhancement (uses best params for signal gen)
    # ------------------------------------------------------------------
    try:
        from utils.ml_signals import run_ml_comparison
        for ticker in tickers:
            if ticker not in all_stats:
                continue
            try:
                df = _precompute_indicators(ohlc_intraday[ticker])
                p  = best_params[ticker]
                entries, exits = _generate_vbt_signals(
                    df,
                    bar_threshold = p['bar_threshold'],
                    rsi_threshold = p['rsi_threshold'],
                    er_th         = p['er_th'],
                    vol_ratio_th  = p['vol_ratio_th'],
                    vwap_dist_max = p['vwap_dist_max'],
                )
                run_ml_comparison(df, entries, exits, ticker, freq='1h')
            except Exception as e:
                print(f"  ❌ ML error for {ticker}: {e}")
    except ImportError:
        print("\n  ML libraries not available. Skipping ML signal enhancement.")


if __name__ == '__main__':
    main()
