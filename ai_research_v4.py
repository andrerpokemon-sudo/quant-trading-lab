"""Single-file exploratory V4: causal mean reversion, regime gating and pair-data audit.
Requires pandas, numpy. Not investment advice or a paper replication.
Examples:
 python ai_research_v4.py --csv SPX500_5m.csv --out v4_spx
 python ai_research_v4.py --csv NAS100_5min_Extended_Dataset.csv --out v4_nas
 python ai_research_v4.py --csv NAS100.csv --pair-csv SPX500.csv --out pair_audit
 python ai_research_v4.py --self-test
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

STRATEGIES=('mean_revert','mean_revert_regime','trend_regime')

def load(path):
    x=pd.read_csv(path);x.columns=x.columns.str.lower().str.strip()
    t=next((c for c in ('timestamp','datetime','date','time') if c in x),None)
    if t is None:raise ValueError('Timestamp column required')
    x[t]=pd.to_datetime(x[t],utc=True);x=x.set_index(t).sort_index()
    if x.index.has_duplicates:raise ValueError('Duplicate timestamps')
    for c in ('open','high','low','close'):
        if c not in x:raise ValueError('Missing '+c)
        x[c]=pd.to_numeric(x[c],errors='raise')
    o=x[['open','high','low','close']]
    if not np.isfinite(o.to_numpy()).all() or (o<=0).any().any():raise ValueError('Invalid OHLC')
    if ((x.high<o[['open','low','close']].max(axis=1))|(x.low>o[['open','high','close']].min(axis=1))).any():raise ValueError('Inconsistent OHLC')
    return x

def signals(x):
    y=x.copy(); c=y.close
    mean=c.rolling(24).mean(); sd=c.rolling(24).std()
    z=(c-mean)/sd.replace(0,np.nan)
    prev=c.shift(1)
    tr=pd.concat([y.high-y.low,(y.high-prev).abs(),(y.low-prev).abs()],axis=1).max(axis=1)
    atr=tr.rolling(14).mean(); baseline=atr.rolling(96).median()
    # Gate based solely on currently completed bar; no future price access.
    normal=(atr/baseline).between(.6,1.5) & ((c-c.shift(12)).abs()/atr < 4)
    trending=((c-c.shift(12)).abs()/atr>2) & (atr/baseline).between(.6,2)
    mr=np.select([z < -1.75,z > 1.75],[1,-1],default=0)
    y['mean_revert']=mr
    y['mean_revert_regime']=np.where(normal,mr,0)
    y['trend_regime']=np.select([trending&(c>c.shift(12)),trending&(c<c.shift(12))],[1,-1],default=0)
    y.loc[baseline.isna(),list(STRATEGIES)]=0
    return y

def trades(y,strategy,hold=3,spread=2,slippage=1):
    if hold<1 or min(spread,slippage)<0:raise ValueError('Invalid costs/hold')
    rows=[];i=120
    while i+1+hold<len(y):
        d=int(y[strategy].iloc[i]); j=i+1+hold
        times=y.index[i:j+1]
        # No entry across missing bars or session gaps.
        if d and (np.diff(times.asi8)/60e9<=7).all():
            entry=float(y.open.iloc[i+1]); exit_=float(y.open.iloc[j]);gross=d*(exit_/entry-1)
            rows.append((strategy,y.index[i],y.index[i+1],y.index[j],d,gross,gross-(spread+2*slippage)/10000))
            i=j
        else:i+=1
    return pd.DataFrame(rows,columns=['strategy','signal_time','entry_time','exit_time','direction','gross_return','net_return'])

def audit_pair(a,b):
    common=a.index.intersection(b.index)
    # Require 20 distinct dates and 2000 synchronous 5-minute observations.
    days=len(pd.Index(common.date).unique())
    return {'shared_bars':len(common),'shared_days':days,'eligible':len(common)>=2000 and days>=20}

def run(csv,out,hold=3,spread=2,slippage=1,pair_csv=None):
    x=load(csv);Path(out).mkdir(parents=True,exist_ok=True)
    if pair_csv:
        status=audit_pair(x,load(pair_csv));pd.DataFrame([status]).to_csv(Path(out)/'pair_data_audit.csv',index=False)
        print('PAIR AUDIT:',status)
        print('Pairs backtest intentionally not implemented: must first establish stationarity, hedging, aligned quotes and non-leaking train/test estimation.')
    if len(x)<500:raise ValueError('At least 500 bars needed')
    y=signals(x);c1=y.index[int(.6*len(y))];c2=y.index[int(.8*len(y))]
    report=[];alltr=[]
    for s in STRATEGIES:
        t=trades(y,s,hold,spread,slippage);alltr.append(t)
        for name,start,end in [('development',y.index[0],c1),('validation',c1,c2),('holdout',c2,y.index[-1]+pd.Timedelta(nanoseconds=1))]:
            q=t[(t.signal_time>=start)&(t.exit_time<end)]
            report.append(dict(strategy=s,segment=name,trades=len(q),mean_net_bps=q.net_return.mean()*1e4 if len(q) else np.nan,win_rate=(q.net_return>0).mean() if len(q) else np.nan))
    report=pd.DataFrame(report);report.to_csv(Path(out)/'v4_comparison.csv',index=False)
    pd.concat(alltr,ignore_index=True).to_csv(Path(out)/'v4_trades.csv',index=False)
    print('Candles:',len(x),'from',x.index[0],'to',x.index[-1]);print('Split boundaries:',c1,c2)
    print(report.to_string(index=False,float_format=lambda z:f'{z:.3f}'))
    print('Exploratory only. Fixed rules; no training/optimisation performed. Constant assumed costs; no funding, sizing, realistic IG spread schedule or verified data provenance.')
    return report

def self_test():
    i=pd.date_range('2026-01-05',periods=1500,freq='5min',tz='UTC');n=np.arange(len(i))
    p=100+np.sin(n/9)*.9+n*.0002
    x=pd.DataFrame({'open':p,'high':p+.1,'low':p-.1,'close':p},index=i)
    y=signals(x)
    assert set(STRATEGIES).issubset(y.columns)
    z=x.copy();z.iloc[-1,z.columns.get_loc('close')]+=0.01
    assert signals(z)[list(STRATEGIES)].iloc[:-1].equals(y[list(STRATEGIES)].iloc[:-1])
    a=trades(y,'mean_revert',spread=0,slippage=0);b=trades(y,'mean_revert',spread=4,slippage=2)
    assert len(a)>0 and len(a)==len(b) and np.allclose(a.net_return-b.net_return,.0008)
    assert not audit_pair(x,x.iloc[:100])['eligible']
    print('PASS: signal causality, cost accounting, pair-data guard')

def main():
    p=argparse.ArgumentParser();p.add_argument('--csv');p.add_argument('--pair-csv');p.add_argument('--out',default='v4_results');p.add_argument('--hold',type=int,default=3);p.add_argument('--spread-bps',type=float,default=2);p.add_argument('--slippage-bps',type=float,default=1);p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:self_test();return
    if not a.csv:p.error('--csv is required')
    run(a.csv,a.out,a.hold,a.spread_bps,a.slippage_bps,a.pair_csv)
if __name__=='__main__':main()
