"""Standalone research-only 5-minute trading signal backtest. Requires pandas and numpy."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def features(df):
    x = df.copy()
    previous = x.close.shift(1)
    tr = pd.concat([x.high-x.low, (x.high-previous).abs(), (x.low-previous).abs()], axis=1).max(axis=1)
    x["atr14"] = tr.rolling(14).mean()
    x["mom12"] = x.close.pct_change(12)
    hi = x.high.shift(1).rolling(20).max()
    lo = x.low.shift(1).rolling(20).min()
    x["sweep_low"] = (x.low < lo) & (x.close > x.open)
    x["sweep_high"] = (x.high > hi) & (x.close < x.open)
    x["signal"] = np.select([(x.close > hi) & (x.mom12 > 0),
                             (x.close < lo) & (x.mom12 < 0)], [1, -1], default=0)
    return x

def load(path):
    x = pd.read_csv(path)
    x.columns = x.columns.str.lower().str.strip()
    t = next((c for c in ("timestamp", "datetime", "date", "time") if c in x), None)
    if t is None:
        raise ValueError("CSV needs a timestamp/datetime/date/time column")
    x[t] = pd.to_datetime(x[t], utc=True)
    x = x.set_index(t).sort_index()
    if x.index.has_duplicates:
        raise ValueError("Duplicate timestamps")
    for c in ("open", "high", "low", "close"):
        if c not in x:
            raise ValueError("Missing column: " + c)
        x[c] = pd.to_numeric(x[c], errors="raise")
    if not np.isfinite(x[["open","high","low","close"]].to_numpy()).all():
        raise ValueError("OHLC contains non-finite values")
    if (x[["open","high","low","close"]] <= 0).any().any():
        raise ValueError("OHLC must be positive")
    if ((x.high < x[["open","close","low"]].max(axis=1)) |
        (x.low > x[["open","close","high"]].min(axis=1))).any():
        raise ValueError("Inconsistent OHLC")
    return x

def backtest(df, spread_bps=2, slippage_bps=1, hold_bars=3):
    if min(spread_bps, slippage_bps) < 0 or hold_bars < 1:
        raise ValueError("Invalid settings")
    x = features(df)
    trades = []
    i = 20
    while i + 1 + hold_bars < len(x):
        direction = int(x.signal.iloc[i])
        if direction and np.isfinite(x.atr14.iloc[i]):
            entry_i, exit_i = i + 1, i + 1 + hold_bars
            entry, exit_price = float(x.open.iloc[entry_i]), float(x.open.iloc[exit_i])
            gross = direction * (exit_price / entry - 1)
            net = gross - (spread_bps + 2 * slippage_bps) / 10000
            trades.append({"signal_time":str(x.index[i]),"entry_time":str(x.index[entry_i]),
                           "exit_time":str(x.index[exit_i]),"direction":direction,
                           "gross_return":gross,"net_return":net})
            i = exit_i
        else:
            i += 1
    return pd.DataFrame(trades, columns=["signal_time","entry_time","exit_time","direction","gross_return","net_return"])

def self_test():
    idx = pd.date_range("2026-01-01", periods=120, freq="5min", tz="UTC")
    price = np.r_[np.full(35, 100.0), np.linspace(100, 115, 85)]
    df = pd.DataFrame({"open":price,"high":price+0.1,"low":price-0.1,"close":price},index=idx)
    a = features(df)
    assert a.signal.iloc[:20].eq(0).all()
    assert a.signal.iloc[40:].eq(1).any()
    trades = backtest(df, spread_bps=0, slippage_bps=0)
    costly = backtest(df, spread_bps=4, slippage_bps=2)
    assert len(trades) > 0 and len(trades) == len(costly)
    assert np.allclose(trades.net_return - costly.net_return, 0.0008)
    altered = df.copy()
    altered.iloc[-1, altered.columns.get_loc("close")] = 120
    assert features(altered).signal.iloc[:-1].equals(a.signal.iloc[:-1])
    print("PASS: causal signals, execution costs, and next-bar entry smoke tests")

def main():
    p = argparse.ArgumentParser(description="Research-only signal baseline; no ML or live trading")
    p.add_argument("--csv", help="5-minute OHLC CSV with timestamp column")
    p.add_argument("--output", default="ai_signal_trades.csv")
    p.add_argument("--spread-bps", type=float, default=2)
    p.add_argument("--slippage-bps", type=float, default=1)
    p.add_argument("--hold-bars", type=int, default=3)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.csv:
        p.error("Provide --csv DATA.csv or --self-test")
    trades = backtest(load(args.csv), args.spread_bps, args.slippage_bps, args.hold_bars)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(args.output, index=False)
    if len(trades):
        print(f"Trades: {len(trades)} | Mean net: {trades.net_return.mean():.4%} | Win rate: {(trades.net_return>0).mean():.2%}")
    else:
        print("No qualifying trades")
    print("Exploratory only; no ML, no validated profitability, no session/financing model.")

if __name__ == "__main__":
    main()
