import numpy as np
import pandas as pd
import pytest
from pairs_lab import backtest_pair, discover

def sample():
    rng = np.random.default_rng(42)
    idx = pd.bdate_range('2020-01-01', periods=520)
    a = np.cumsum(rng.normal(0, .01, len(idx)))
    b = a + rng.normal(0, .015, len(idx))
    c = np.cumsum(rng.normal(0, .01, len(idx)))
    return pd.DataFrame({'AAA': 100*np.exp(a), 'BBB': 80*np.exp(b), 'CCC': 100*np.exp(c)}, index=idx)

def test_selection_is_training_only():
    p = sample()
    cutoff = p.index[350]
    altered = p.copy()
    altered.iloc[350:] *= 5
    assert discover(p, cutoff) == discover(altered, cutoff)

def test_costs_reduce_performance():
    p = sample()
    cutoff = p.index[300]
    free = backtest_pair(p, 'AAA', 'BBB', cutoff, cost_bps=0)
    paid = backtest_pair(p, 'AAA', 'BBB', cutoff, cost_bps=20)
    assert paid['total_return'] <= free['total_return'] + 1e-10

def test_short_history_rejected():
    with pytest.raises(ValueError):
        backtest_pair(sample().iloc[:100], 'AAA', 'BBB', '2020-04-01')
