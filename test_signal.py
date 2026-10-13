import numpy as np
import pandas as pd
from ai_signal import features, backtest

def sample():
    rng=np.random.default_rng(42)
    close=100+np.cumsum(rng.normal(0,0.2,300))
    index=pd.date_range('2025-01-01',periods=300,freq='5min',tz='UTC')
    return pd.DataFrame({'open':close,'high':close+0.1,'low':close-0.1,'close':close},index=index)

def test_no_future_leakage():
    x=sample()
    a=features(x.iloc[:200])
    b=features(x)
    pd.testing.assert_series_equal(a.signal,b.signal.iloc[:200])

def test_costs_reduce_return():
    x=sample()
    a=backtest(x,0,0)
    b=backtest(x,4,2)
    assert len(a)==len(b)
    if len(a): assert np.allclose(a.net_return-b.net_return,0.0008)

def test_entry_after_signal():
    trades=backtest(sample())
    if len(trades): assert (pd.to_datetime(trades.entry_time)>pd.to_datetime(trades.signal_time)).all()
