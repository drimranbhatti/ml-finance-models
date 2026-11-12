"""Download the datasets used by the notebooks into data/.

    python scripts/fetch_data.py
"""
from pathlib import Path

import yfinance as yf

DATA = Path(__file__).resolve().parents[1] / "data"
SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
LARGE_CAPS = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "BRK-B", "JPM", "JNJ", "V", "PG", "XOM", "UNH",
              "HD", "MA", "CVX", "LLY", "PFE", "KO", "PEP", "MRK", "ABBV", "BAC", "WMT", "COST", "DIS", "CSCO",
              "ORCL", "INTC", "ADBE", "CRM", "NFLX", "T", "VZ", "MCD", "NKE", "IBM", "CAT", "GE", "HON"]
MACRO = ["SPY", "^VIX", "^TNX", "^IRX", "HYG", "IEF", "GLD", "UUP"]


def dl(tickers, start):
    px = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    return px.dropna(how="all").round(4)


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    dl(SECTORS + ["SPY"], "2005-01-01").to_csv(DATA / "sector_etfs.csv")
    dl(LARGE_CAPS + ["SPY"], "2015-01-01").to_csv(DATA / "large_caps.csv")
    dl(MACRO, "2008-01-01").to_csv(DATA / "macro.csv")
    print("written to", DATA)
