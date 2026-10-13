"""Research-only causal signals and next-bar-open execution backtest."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def load(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.strip()
    for candidate in ('timestamp', 'datetime', 'date', 'time'):
        if candidate in df:
            df[candidate] = pd.to_datetime(df[candidate], utc=True)
            df = df.set_index(candidate)
            break
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError('CSV needs timestamp/datetime/date/time column')
    df = df.sort_index()
    if df.index.has_duplicates:
        raise ValueError('Duplicate timestamps')
    for c in ('open','high','low','close'):
        if c not in df: raise ValueError(f'Missing {c}')
        df[c] = pd.to_numeric(df[c], errors='raise')
    if (df[['open','high','low','close']] <= 0).any().any():
        raise ValueError('OHLC must be positive')
    return df


def features(df):
    x = df.copy()
    prev = x.close.shift(1)
    tr = pd.concat([x.high-x.low, (x.high-prev).abs(), (x.low-prev).abs()],axis=1).max(axis=1)
    x['atr14'] = tr.rolling(14).mean()
    x['mom12'] = x.close.pct_change(12)
    hi = x.high.shift(1).rolling(20).max()
    lo = x.low.shift(1).rolling(20).min()
    x['breakout_long'] = (x.close > hi) & (x.mom12 > 0)
    x['breakout_short'] = (x.close < lo) & (x.mom12 < 0)
    x['sweep_low'] = (x.low < lo) & (x.close > x.open)
    x['sweep_high'] = (x.high > hi) & (x.close < x.open)
    x['signal'] = np.select([x.breakout_long, x.breakout_short], [1,-1],default=0)
    return x


def backtest(df, spread_bps=2.0, slippage_bps=1.0, hold_bars=3):
    """Non-overlapping trades: enter next open, exit open after hold_bars.
    Costs round trip in bps; no overnight, financing or position sizing modeled.
    """
    if hold_bars < 1 or spread_bps < 0 or slippage_bps < 0:
        raise ValueError('Invalid execution settings')
    x = features(df)
    trades = []
    i = 20
    while i + hold_bars < len(x):
        direction = int(x.signal.iloc[i])
        if direction and np.isfinite(x.atr14.iloc[i]):
            entry_idx, exit_idx = i+1, i+1+hold_bars
            if exit_idx >= len(x): break
            entry, exit_price = float(x.open.iloc[entry_idx]), float(x.open.iloc[exit_idx])
            gross = direction * (exit_price/entry - 1)
            cost = 2 * (spread_bps/2 + slippage_bps) / 10000
            trades.append(dict(signal_time=str(x.index[i]), entry_time=str(x.index[entry_idx]),
                               exit_time=str(x.index[exit_idx]), direction=direction,
                               gross_return=gross, net_return=gross-cost))
            i = exit_idx
        else:
            i += 1
    return pd.DataFrame(trades, columns=['signal_time','entry_time','exit_time','direction','gross_return','net_return'])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--csv',required=True)
    p.add_argument('--output',default='ai_signal_trades.csv')
    p.add_argument('--spread-bps',type=float,default=2)
    p.add_argument('--slippage-bps',type=float,default=1)
    p.add_argument('--hold-bars',type=int,default=3)
    args=p.parse_args()
    trades=backtest(load(args.csv),args.spread_bps,args.slippage_bps,args.hold_bars)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    trades.to_csv(args.output,index=False)
    if len(trades):
        print(f'Trades: {len(trades)} | Mean net return: {trades.net_return.mean():.5%} | Win rate: {(trades.net_return>0).mean():.2%}')
    else: print('No qualifying trades')
    print('Exploratory only: no ML model fitted, no validated profitability.')

if __name__=='__main__': main()
