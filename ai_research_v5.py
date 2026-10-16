"""V5: auditable market-data gate before further strategy fitting.
Usage: python ai_research_v5.py --csv NAS100.csv --pair-csv SPX500.csv --out v5_results
       python ai_research_v5.py --self-test
Requires pandas and numpy. No data downloads or live trading.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def load(path):
    x=pd.read_csv(path); x.columns=x.columns.str.lower().str.strip()
    t=next((c for c in ('timestamp','datetime','date','time') if c in x.columns),None)
    if t is None: raise ValueError('Missing timestamp column')
    x[t]=pd.to_datetime(x[t],utc=True,errors='raise')
    if x[t].isna().any(): raise ValueError('Invalid timestamp')
    x=x.set_index(t).sort_index()
    for c in ('open','high','low','close'):
        if c not in x: raise ValueError('Missing '+c)
        x[c]=pd.to_numeric(x[c],errors='coerce')
    return x

def audit(x,name):
    o=x[['open','high','low','close']]
    valid=np.isfinite(o.to_numpy()).all(axis=1)&(o.to_numpy()>0).all(axis=1)
    consistent=(x.high>=o[['open','close','low']].max(axis=1))&(x.low<=o[['open','close','high']].min(axis=1))
    deltas=x.index.to_series().diff().dt.total_seconds().div(60)
    dates=pd.Index(x.index.date)
    daily=pd.Series(1,index=x.index).resample('D').sum()
    active=daily[daily>0]
    return dict(dataset=name,bars=len(x),first_utc=str(x.index.min()),last_utc=str(x.index.max()),calendar_days=(x.index.max()-x.index.min()).days+1,
                active_dates=dates.nunique(),duplicate_timestamps=int(x.index.duplicated().sum()),invalid_ohlc=int((~valid).sum()),inconsistent_ohlc=int((~consistent).sum()),
                non_5min_steps=int((deltas.notna()&(deltas!=5)).sum()),gaps_over_15min=int((deltas>15).sum()),max_gap_hours=round(float(deltas.max()/60),2),
                median_bars_per_active_day=float(active.median()),volume_column=('volume' in x.columns))

def pair_gate(a,b):
    common=a.index.intersection(b.index)
    dates=pd.Index(common.date)
    # Eligibility is necessary, not sufficient: synchronisation, provenance,
    # rolling hedge estimation and cointegration still need verification.
    return dict(shared_bars=len(common),shared_days=dates.nunique(),first_shared=str(common.min()) if len(common) else '',last_shared=str(common.max()) if len(common) else '',
                enough_overlap=bool(len(common)>=5000 and dates.nunique()>=30),research_approved=False,
                reason='Further provenance, rolling cointegration and timestamp audits required' if len(common)>=5000 and dates.nunique()>=30 else 'Insufficient common history (require >=5000 bars across >=30 days)')

def run(csv,pair_csv,out):
    a=load(csv); rows=[audit(a,Path(csv).name)]
    p=Path(out);p.mkdir(parents=True,exist_ok=True)
    if pair_csv:
        b=load(pair_csv);rows.append(audit(b,Path(pair_csv).name))
        pair=pd.DataFrame([pair_gate(a,b)]);pair.to_csv(p/'v5_pair_gate.csv',index=False)
        print('PAIR GATE\n'+pair.to_string(index=False))
    report=pd.DataFrame(rows);report.to_csv(p/'v5_data_audit.csv',index=False)
    print('DATA AUDIT\n'+report.to_string(index=False))
    print('No strategy profitability asserted. Data source and timezone provenance must be independently verified.')

def self_test():
    i=pd.date_range('2026-01-01',periods=100,freq='5min',tz='UTC')
    x=pd.DataFrame({'open':100.,'high':101.,'low':99.,'close':100.},index=i)
    assert audit(x,'synthetic')['non_5min_steps']==0
    assert pair_gate(x,x)['enough_overlap'] is False
    y=x.drop(x.index[30]);assert audit(y,'synthetic')['non_5min_steps']==1
    print('PASS: interval detection and pair eligibility guard')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--csv');parser.add_argument('--pair-csv');parser.add_argument('--out',default='v5_results');parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    if args.self_test:self_test()
    elif args.csv:run(args.csv,args.pair_csv,args.out)
    else:parser.error('Provide --csv or --self-test')
