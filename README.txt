====================================================================
 Dukascopy Historical Data Downloader
 Free tool by Ayman Fouda | aymanfouda.com | contact@aymanfouda.com
====================================================================

Download free, legal historical price data (OHLCV candles) from
Dukascopy Bank's public data feed, saved as clean CSV files ready
for backtesting, Excel, TradingView import, or any analysis tool.

WHAT YOU CAN DOWNLOAD
---------------------
- Forex pairs:      EURUSD, GBPUSD, USDJPY, GBPJPY, ...
- Metals:           XAUUSD (gold), XAGUSD (silver)
- Index CFDs:       US500, US30, NAS100, GER40, UK100, JP225, DXY
- US stock CFDs:    AAPL, TSLA, MSFT, ...
- Timeframes:       1min, 5min, 15min, 30min, 1hour, 4hour, daily

SETUP (one time, ~2 minutes)
----------------------------
1. Install Python 3.9 or newer from https://python.org
   (on Windows, tick "Add Python to PATH" during install)
2. Open a terminal (Command Prompt / Terminal app) and run:

     pip install dukascopy-python pandas

HOW TO USE - EASY MODE
----------------------
Run the script with no options and just answer the questions:

     python dukascopy_downloader.py

HOW TO USE - COMMAND LINE
-------------------------
     python dukascopy_downloader.py --symbols EURUSD XAUUSD --timeframe h1 --start 2024-01-01 --end 2025-01-01

     --symbols     one or more short symbols (EURUSD XAUUSD US500 AAPL)
     --timeframe   m1 m5 m15 m30 h1 h4 d1        (default: h1)
     --start       YYYY-MM-DD                     (default: 1 year ago)
     --end         YYYY-MM-DD, exclusive          (default: today)
     --list GOLD   browse available instruments (optional filter)

OUTPUT
------
CSV files are written to the "data" folder next to the script:

     EURUSD_h1_20240101_20250101.csv
     Date, Open, High, Low, Close, Volume

Timestamps are UTC. Volume is tick volume (Dukascopy does not
publish real traded volume).

GOOD TO KNOW
------------
- Downloads are split per month and cached in the "cache" folder.
  If a download is interrupted, run the same command again and it
  resumes from where it stopped.
- Large M1 ranges take a while (about a minute per year per symbol
  on a normal connection). H1 and D1 are much faster.
- The data comes from Dukascopy Bank's freely published feed and is
  free to use for personal research and backtesting.

NEED MORE?
----------
I build custom trading bots, indicators, backtests and dashboards:

     Website:  https://aymanfouda.com
     Fiverr:   https://fiverr.com/ayman_fouda
     Email:    contact@aymanfouda.com
