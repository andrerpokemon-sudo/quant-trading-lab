# AI Signal Engine V1 — iPhone upload

This ZIP contains **three repository-relative files**: `ai_signal.py`, `tests/test_signal.py`, `.github/workflows/ai-signal.yml`. Extract the ZIP first; **GitHub does not automatically unpack ZIPs**.

1. In iOS Files, tap ZIP to extract it.
2. On GitHub in Safari, open `andrerpokemon-sudo/quant-trading-lab` and use **Add file → Upload files** (request Desktop Website if needed).
3. Upload `ai_signal.py`, preserving the `tests/` and `.github/workflows/` folder paths. If the phone uploader cannot preserve folders, use GitHub Codespaces web editor to create the paths and paste file contents.
4. Commit to `main`, open **Actions → AI signal research tests → Run workflow**.

This first version is a deterministic market-structure/momentum baseline, **not yet an AI/ML model**. The tests check causality and costs; they do not establish a profitable strategy. No broker connections or real-money orders. Run a research backtest with `python ai_signal.py --csv YOUR_5MIN_OHLC.csv --output results.csv`. Input needs timestamp, open, high, low, close. CSVs are not bundled.

Important: assumes continuous bars and flat bps execution costs; excludes financing, spread variation, session boundaries, FX, borrow, gap risk, taxes, contract specifications and realistic stop/limit fills. Do not use for live trading. Next stage: add purged walk-forward ML with truly unseen holdout and realistic IG market costs.
