# Quant Trading Lab

Research-only stock pairs scanner and chronological holdout backtester.

## iPhone setup

1. Extract the ZIP in iOS Files.
2. Open your repository on GitHub in Safari; choose **Add file → Upload files** (request desktop website if necessary).
3. Select the extracted files and folders, preserving `.github/workflows/`.
4. Commit. Under **Actions**, run **Manual pairs scan → Run workflow**.
5. Download `pairs-scan-results` under that run's Artifacts section.

If GitHub's mobile upload refuses folders, use a desktop browser or GitHub Codespaces to upload the extracted directory. GitHub does not automatically unpack ZIP archives.

## Local usage

```bash
pip install -r requirements.txt
python -m pytest -q
python pairs_lab.py --output results.csv
```

## Important limitations

This is not an executable trading strategy. The default universe is illustrative and subject to survivorship bias. Cointegration selection is based on training data, but multiple testing can produce false discoveries. Transaction costs are approximate and exclude borrow availability, shorting restrictions, financing, slippage, tax, FX and corporate-action data issues. The backtester uses one fixed training split and a simplified daily rebalanced spread return; a true tradable self-financing portfolio and rolling walk-forward evaluation are future work. No broker access or live orders are included. Treat all outputs as exploratory.
