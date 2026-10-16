"""Single-file exploratory strategy comparison. Requires numpy and pandas.
Strategies are research-inspired prototypes, NOT exact replications of SSRN papers.
Usage: python ai_research_v2.py --csv market.csv --market US --out results
       python ai_research_v2.py --self-test
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

STRATEGIES = ('baseline', 'momentum', 'orb', 'retest')

def load_csv(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.strip()
    tc = next((c for c in ('timestamp','datetime','date','time') if c in df.columns), None)
    if tc is None: raise ValueError('Missing timestamp column')
    df[tc] = pd.to_datetime(df[tc], utc=True)
    df = df.set_index(tc).sort_index()
    if df.index.has_duplicates: raise ValueError('Duplicate timestamps')
    for c in ('open','high','low','close'):
        if c not in df: raise ValueError('Missing '+c)
        df[c] = pd.to_numeric(df[c], errors='raise')
    a = df[['open','high','low','close']]
    if not np.isfinite(a.to_numpy()).all() or (a<=0).any().any(): raise ValueError('Invalid OHLC')
    if (df.high < a[['open','close','low']].max(axis=1)).any() or (df.low > a[['open','close','high']].min(axis=1)).any():
        raise ValueError('Inconsistent OHLC')
    return df

def make_signals(df, market='US'):
    x = df.copy()
    prev = x.close.shift(1)
    tr = pd.concat([x.high-x.low, (x.high-prev).abs(), (x.low-prev).abs()],axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    hi = x.high.shift(1).rolling(20).max()
    lo = x.low.shift(1).rolling(20).min()
    mom = x.close.pct_change(12)
    x['baseline'] = np.select([(x.close>hi)&(mom>0),(x.close<lo)&(mom<0)],[1,-1],0)
    # Intraday momentum: 12-bar return, 48-bar trend, volatility gate. Fixed ex ante.
    trend = x.close.rolling(48).mean()
    x['momentum'] = np.select([(mom>0.001)&(x.close>trend)&(tr<3*atr),
                               (mom< -0.001)&(x.close<trend)&(tr<3*atr)],[1,-1],0)
    # ORB: first 30 minutes of cash session; entries on subsequent bars only.
    # Europe/Berlin and America/New_York timezone conversions account for DST.
    tz = 'America/New_York' if market=='US' else 'Europe/Berlin'
    local = x.index.tz_convert(tz)
    hh = local.hour*60 + local.minute
    start = 9*60+30 if market=='US' else 9*60
    session = (hh>=start)&(hh<start+390 if market=='US' else hh<start+510)
    opening = (hh>=start)&(hh<start+30)
    dates = pd.Series(local.date,index=x.index)
    # Each day needs 6 opening candles, and opening bars must be contiguous.
    opening_high = x.high.where(opening).groupby(dates).transform('max')
    opening_low = x.low.where(opening).groupby(dates).transform('min')
    opening_count = pd.Series(opening.astype(int),index=x.index).groupby(dates).transform('sum')
    post = session & (hh>=start+30) & (opening_count==6)
    x['orb'] = np.select([post&(x.close>opening_high)&(x.close.shift(1)<=opening_high)&(mom>0),
                          post&(x.close<opening_low)&(x.close.shift(1)>=opening_low)&(mom<0)],[1,-1],0)
    # Breakout-retest: previous bar pierced prior 20-bar boundary; current bar
    # returns to that boundary and closes back in breakout direction.
    upbreak = (x.close.shift(1)>hi.shift(1))
    downbreak = (x.close.shift(1)<lo.shift(1))
    x['retest'] = np.select([upbreak&(x.low<=hi.shift(1))&(x.close>hi.shift(1))&(x.close>x.open),
                            downbreak&(x.high>=lo.shift(1))&(x.close<lo.shift(1))&(x.close<x.open)],[1,-1],0)
    x.loc[atr.isna(),list(STRATEGIES)] = 0
    return x

def backtest(x, strategy, spread_bps=2, slippage_bps=1, hold_bars=3, max_gap_minutes=7):
    if strategy not in STRATEGIES: raise ValueError('Unknown strategy')
    if spread_bps<0 or slippage_bps<0 or hold_bars<1: raise ValueError('Invalid execution parameters')
    out=[]; i=50
    while i+1+hold_bars<len(x):
        direction=int(x[strategy].iloc[i])
        end=i+1+hold_bars
        # Require continuous 5-minute bars across signal, entry and exit.
        window=x.index[i:end+1]
        valid=(np.diff(window.asi8)/60e9 <= max_gap_minutes).all()
        if direction and valid:
            entry=float(x.open.iloc[i+1]); exit_=float(x.open.iloc[end])
            gross=direction*(exit_/entry-1)
            net=gross-(spread_bps+2*slippage_bps)/10000
            out.append((strategy,x.index[i],x.index[i+1],x.index[end],direction,gross,net))
            i=end
        else: i+=1
    return pd.DataFrame(out,columns=['strategy','signal_time','entry_time','exit_time','direction','gross_return','net_return'])

def summarize(trades, market, cutoff):
    rows=[]
    for strategy in STRATEGIES:
        t=trades[trades.strategy==strategy]
        for segment, sub in [('train',t[t.signal_time<cutoff]),('holdout',t[t.signal_time>=cutoff])]:
            n=len(sub)
            # Net returns are fractions of entry not portfolio returns; cumulative figure
            # assumes one unit of notional per non-overlapping trade, without leverage.
            rows.append(dict(market=market,strategy=strategy,segment=segment,trades=n,
                win_rate=float((sub.net_return>0).mean()) if n else np.nan,
                mean_net_bps=float(sub.net_return.mean()*10000) if n else np.nan,
                sum_net_bps=float(sub.net_return.sum()*10000) if n else np.nan))
    return pd.DataFrame(rows)

def run(csv, market, out, spread, slippage, hold):
    df=load_csv(csv)
    if len(df)<150: raise ValueError('Need at least 150 candles')
    x=make_signals(df,market)
    cutoff=x.index[int(len(x)*0.7)]
    trades=pd.concat([backtest(x,s,spread,slippage,hold) for s in STRATEGIES],ignore_index=True)
    report=summarize(trades,market,cutoff)
    Path(out).mkdir(parents=True,exist_ok=True)
    trades.to_csv(Path(out)/f'{market}_trades.csv',index=False)
    report.to_csv(Path(out)/f'{market}_comparison.csv',index=False)
    print(f'{market}: {len(df)} candles, {x.index.min()} to {x.index.max()}; holdout begins {cutoff}')
    print(report.to_string(index=False,float_format=lambda n:f'{n:.3f}'))
    print('WARNING: Exploratory prototypes, not paper replications. No financing, portfolio sizing or instrument-specific spread model.')
    return report

def self_test():
    idx=pd.date_range('2026-01-05 14:30',periods=600,freq='5min',tz='UTC')
    p=100+np.arange(600)*.025+np.sin(np.arange(600)/4)*0.45
    df=pd.DataFrame({'open':p,'high':p+.08,'low':p-.08,'close':p},index=idx)
    x=make_signals(df)
    assert set(STRATEGIES).issubset(x.columns)
    assert (x[list(STRATEGIES)].iloc[:14]==0).all().all()
    a=backtest(x,'baseline',0,0); b=backtest(x,'baseline',4,2)
    assert len(a)==len(b) and len(a)>0
    assert np.allclose(a.net_return-b.net_return,.0008)
    changed=df.copy(); changed.iloc[-1,3]=p[-1]+.01
    assert make_signals(changed)[list(STRATEGIES)].iloc[:-1].equals(x[list(STRATEGIES)].iloc[:-1])
    print('PASS: signals, causality smoke test, cost accounting')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--csv'); p.add_argument('--market',choices=['US','EU'],default='US')
    p.add_argument('--out',default='research_results'); p.add_argument('--spread-bps',type=float,default=2)
    p.add_argument('--slippage-bps',type=float,default=1); p.add_argument('--hold-bars',type=int,default=3)
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test: self_test(); return
    if not a.csv: p.error('--csv required')
    run(a.csv,a.market,a.out,a.spread_bps,a.slippage_bps,a.hold_bars)
if __name__=='__main__': main()
