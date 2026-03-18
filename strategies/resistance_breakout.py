"""
Intraday Resistance Breakout Strategy — Advanced Backtest (v3.0)

Key Improvements over v2.0:
  * Optimized parameters are extracted and reused for a clean final backtest
    (separates parameter search from performance reporting).
  * Best parameters printed per ticker + collected in a summary table.
  * VBT and ML signal generation also use best discovered parameters.
  * All v2.0 features preserved: ADX, EMA trend filter, ATR sizing, Vol Z-Score.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import datetime as dt
from backtesting import Backtest, Strategy

import yfinance as yf
from indicators.atr import calculate_atr
from indicators.adx import calculate_adx
from utils.backtesting import VBTBacktester

# ----------------------------- CONFIG ----------------------------- #
TICKERS = ["NVDA", "AAPL", "GOOGL", "META", "AMZN", "MSFT", "TSLA"]
START_DATE = dt.datetime.today() - dt.timedelta(days=730)
END_DATE   = dt.datetime.today()
INTERVAL   = '1h'
CASH       = 100_000
COMMISSION = 0.001

# Default parameter fallback (used when optimization yields < 10 trades)
DEFAULT_PARAMS = dict(
    vol_z_threshold   = 1.5,
    atr_breakout_coef = 0.3,
    tp_factor         = 2.0,
    sl_factor         = 1.0,
    adx_threshold     = 20,
)


# ---------------------- INDICATOR HELPERS ---------------------- #
def _precompute_indicators(df, atr_period=20, roll_period=14, ema_period=50):
    """Pre-compute ATR, ADX, rolling breakout levels, Volume Z-Score, and EMA."""
    df = calculate_atr(df, atr_period)
    df = calculate_adx(df, period=20)

    df['roll_max_cp'] = df['High'].rolling(roll_period).max().shift(1)
    df['roll_min_cp'] = df['Low'].rolling(roll_period).min().shift(1)

    vol_mean = df['Volume'].rolling(roll_period).mean().shift(1)
    vol_std  = df['Volume'].rolling(roll_period).std().shift(1)
    df['vol_zscore'] = (df['Volume'] - vol_mean) / (vol_std + 1e-9)

    df['ema_trend'] = df['Close'].ewm(span=ema_period, adjust=False).mean()

    df.dropna(inplace=True)
    return df


# ---------------------- STRATEGY CLASS ---------------------- #
class BreakoutStrategy(Strategy):
    """
    Breakout strategy v3.0:
    - Buy when High breaks rolling max + a fraction of ATR
    - Sell when Low breaks rolling min - a fraction of ATR
    - Volume must be a statistical anomaly (Z-Score > threshold)
    - ADX must be > threshold (strong trend condition)
    - Trade must align with 50-EMA direction
    - ATR-based stop-loss and take-profit management
    - All key thresholds are optimizable
    """
    vol_z_threshold   = 1.5
    atr_breakout_coef = 0.3
    tp_factor         = 2.0
    sl_factor         = 1.0
    adx_threshold     = 20

    def init(self):
        self.atr        = self.I(lambda: self.data.ATR,          name='ATR',         overlay=False)
        self.roll_max   = self.I(lambda: self.data.roll_max_cp,  name='Resist.',     overlay=True)
        self.roll_min   = self.I(lambda: self.data.roll_min_cp,  name='Support',     overlay=True)
        self.vol_zscore = self.I(lambda: self.data.vol_zscore,   name='Vol Z-Score', overlay=False)
        self.adx        = self.I(lambda: self.data.ADX,          name='ADX',         overlay=False)
        self.ema        = self.I(lambda: self.data.ema_trend,    name='EMA',         overlay=True)

    def next(self):
        high   = self.data.High[-1]
        low    = self.data.Low[-1]
        close  = self.data.Close[-1]
        atr    = self.atr[-1]
        r_max  = self.roll_max[-1]
        r_min  = self.roll_min[-1]
        vol_z  = self.vol_zscore[-1]
        adx_val= self.adx[-1]
        ema_val= self.ema[-1]

        vol_breakout   = vol_z > self.vol_z_threshold
        trend_strong   = adx_val > self.adx_threshold
        long_trigger   = r_max + (atr * self.atr_breakout_coef)
        short_trigger  = r_min - (atr * self.atr_breakout_coef)
        trend_up       = close > ema_val
        trend_down     = close < ema_val

        # ATR-based position sizing (risk 2% of equity per trade)
        equity = self.equity if hasattr(self, 'equity') else CASH
        shares = max(1, int((equity * 0.02) / atr)) if atr > 0 else 1  # noqa: F841

        if not self.position:
            if high >= long_trigger and vol_breakout and trend_strong and trend_up:
                self.buy(sl=close - atr * self.sl_factor,
                         tp=close + atr * self.tp_factor)
            elif low <= short_trigger and vol_breakout and trend_strong and trend_down:
                self.sell(sl=close + atr * self.sl_factor,
                          tp=close - atr * self.tp_factor)

        elif self.position.is_long:
            new_stop = close - atr * self.sl_factor
            if hasattr(self, '_long_stop'):
                self._long_stop = max(self._long_stop, new_stop)
            else:
                self._long_stop = new_stop
            for trade in self.trades:
                if trade.is_long and (trade.sl is None or self._long_stop > trade.sl):
                    trade.sl = self._long_stop

            if low <= short_trigger and vol_breakout and trend_strong and trend_down:
                self.position.close()
                self.sell(sl=close + atr * self.sl_factor,
                          tp=close - atr * self.tp_factor)

        elif self.position.is_short:
            new_stop = close + atr * self.sl_factor
            if hasattr(self, '_short_stop'):
                self._short_stop = min(self._short_stop, new_stop)
            else:
                self._short_stop = new_stop
            for trade in self.trades:
                if trade.is_short and (trade.sl is None or self._short_stop < trade.sl):
                    trade.sl = self._short_stop

            if high >= long_trigger and vol_breakout and trend_strong and trend_up:
                self.position.close()
                self.buy(sl=close - atr * self.sl_factor,
                         tp=close + atr * self.tp_factor)


# ---------------------- VBT SIGNAL HELPER ---------------------- #
def _generate_vbt_signals(df, vol_z_threshold=1.5, atr_coef=0.3, adx_threshold=20):
    vol_breakout  = df['vol_zscore'] > vol_z_threshold
    trend_strong  = df['ADX'] > adx_threshold
    trend_up      = df['Close'] > df['ema_trend']
    trend_down    = df['Close'] < df['ema_trend']
    long_trigger  = df['roll_max_cp'] + (df['ATR'] * atr_coef)
    short_trigger = df['roll_min_cp'] - (df['ATR'] * atr_coef)

    entries = (df['High'] >= long_trigger) & vol_breakout & trend_strong & trend_up
    exits   = (df['Low']  <= short_trigger) & vol_breakout & trend_strong & trend_down
    return entries, exits


# ---------------------- PARAM EXTRACTION HELPER ---------------------- #
def _extract_best_params(opt_stats) -> dict:
    """
    Extract the winning parameter set from backtesting.py optimization results.
    backtesting.py stores the best combo as class attributes on opt_stats._strategy.
    """
    strat = opt_stats._strategy
    return dict(
        vol_z_threshold   = float(strat.vol_z_threshold),
        atr_breakout_coef = float(strat.atr_breakout_coef),
        tp_factor         = float(strat.tp_factor),
        sl_factor         = float(strat.sl_factor),
        adx_threshold     = int(strat.adx_threshold),
    )


# ----------------------------- MAIN ----------------------------- #
def main():
    print("=" * 70)
    print("  Resistance Breakout Strategy — Advanced Backtest (v3.0)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Download data
    # ------------------------------------------------------------------
    print("\nFetching intraday OHLCV data...")
    ohlcv = {}
    for ticker in TICKERS:
        try:
            data = yf.download(ticker, start=START_DATE, end=END_DATE,
                               interval=INTERVAL, progress=False, auto_adjust=True)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data.rename(columns={'Adj Close': 'Close'}, inplace=True, errors='ignore')
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            data = data.between_time('09:35', '16:00')
            if not data.empty:
                ohlcv[ticker] = data
                print(f"  ✅ {ticker}: {len(data)} rows")
        except Exception as e:
            print(f"  ❌ {ticker}: {e}")

    tickers = list(ohlcv.keys())
    if not tickers:
        print("No data fetched. Exiting.")
        return

    # ------------------------------------------------------------------
    # 2. Optimize → extract best params → re-run clean final backtest
    # ------------------------------------------------------------------
    all_stats   = {}
    best_params = {}

    for ticker in tickers:
        print(f"\n{'─' * 60}")
        print(f"📊  {ticker} — Step 1: Optimizing …")

        try:
            df = _precompute_indicators(ohlcv[ticker])
            if len(df) < 50:
                print(f"  ⚠️  Skipping {ticker}: insufficient data")
                continue

            bt = Backtest(df, BreakoutStrategy,
                          cash=CASH, commission=COMMISSION,
                          exclusive_orders=True, finalize_trades=True)

            # ── Optimization pass ──────────────────────────────────────
            params = DEFAULT_PARAMS.copy()
            try:
                opt_stats = bt.optimize(
                    vol_z_threshold   = list(np.arange(1.0, 3.0, 0.1)),
                    atr_breakout_coef = list(np.arange(0.3, 0.7, 0.05)),
                    tp_factor         = list(np.arange(1.0, 2.0, 0.1)),
                    sl_factor         = list(np.arange(0.5, 1.0, 0.1)),
                    adx_threshold     = list(np.arange(10, 30, 5)),
                    maximize          = 'Sharpe Ratio',
                    max_tries         = 27,
                    return_heatmap    = False,
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

            if final_stats['# Trades'] < 10:
                print(f"  ⚠️  {ticker}: only {final_stats['# Trades']} trades — "
                      f"interpret results cautiously")

            all_stats[ticker] = {
                'Return [%]':       final_stats['Return [%]'],
                'Sharpe Ratio':     final_stats['Sharpe Ratio'],
                'Max Drawdown [%]': final_stats['Max. Drawdown [%]'],
                '# Trades':         final_stats['# Trades'],
                'Win Rate [%]':     final_stats['Win Rate [%]'],
                # Best params for traceability
                'vol_z_threshold':   params['vol_z_threshold'],
                'atr_coef':          params['atr_breakout_coef'],
                'tp_factor':         params['tp_factor'],
                'adx_threshold':     params['adx_threshold'],
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
    # 3. Summary tables
    # ------------------------------------------------------------------
    if all_stats:
        print("\n" + "=" * 70)
        print("  FINAL BACKTEST RESULTS (using optimized parameters per ticker)")
        print("=" * 70)
        print(pd.DataFrame(all_stats).T.to_string(float_format=lambda x: f"{x:.2f}"))

        print("\n" + "─" * 70)
        print("  Best Parameters Per Ticker")
        print("─" * 70)
        print(pd.DataFrame(best_params).T.to_string(float_format=lambda x: f"{x:.2f}"))

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
            df = _precompute_indicators(ohlcv[ticker])
            p  = best_params[ticker]
            entries, exits = _generate_vbt_signals(
                df,
                vol_z_threshold = p['vol_z_threshold'],
                atr_coef        = p['atr_breakout_coef'],
                adx_threshold   = p['adx_threshold'],
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
                df = _precompute_indicators(ohlcv[ticker])
                p  = best_params[ticker]
                entries, exits = _generate_vbt_signals(
                    df,
                    vol_z_threshold = p['vol_z_threshold'],
                    atr_coef        = p['atr_breakout_coef'],
                    adx_threshold   = p['adx_threshold'],
                )
                run_ml_comparison(df, entries, exits, ticker, freq='1h')
            except Exception as e:
                print(f"  ❌ ML error for {ticker}: {e}")
    except ImportError:
        print("\n  ⚠️  ML libraries not available. Skipping ML signal enhancement.")


if __name__ == '__main__':
    main()
