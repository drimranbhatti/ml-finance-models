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


def french_monthly(name):
    """Monthly table from a Ken French data-library zip (values in %, converted to decimals)."""
    import io, urllib.request, zipfile
    import pandas as pd
    url = f"https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{name}_CSV.zip"
    raw = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url).read())).read(f"{name}.csv").decode("latin-1").splitlines()
    start = next(i for i, l in enumerate(raw) if l.strip().startswith(",")) 
    rows = []
    for l in raw[start + 1:]:
        parts = [p.strip() for p in l.split(",")]
        if len(parts[0]) != 6 or not parts[0].isdigit():
            break
        rows.append(parts)
    cols = [c.strip() for c in raw[start].split(",")][1:]
    df = pd.DataFrame([r[1:] for r in rows], index=pd.to_datetime([r[0] for r in rows], format="%Y%m") + pd.offsets.MonthEnd(0), columns=cols).astype(float) / 100
    return df


def french_daily_realised_var(name):
    """Monthly realised variance (sum of squared daily returns) from a Ken French daily file."""
    import io, urllib.request, zipfile
    import pandas as pd
    url = f"https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{name}_CSV.zip"
    raw = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url).read())).read(f"{name}.csv").decode("latin-1").splitlines()
    start = next(i for i, l in enumerate(raw) if l.strip().startswith(","))
    cols = [c.strip() for c in raw[start].split(",")][1:]
    rows = [[p.strip() for p in l.split(",")] for l in raw[start + 1:] if l[:8].strip().isdigit() and len(l.split(",")[0].strip()) == 8]
    df = pd.DataFrame([r[1:] for r in rows], index=pd.to_datetime([r[0] for r in rows], format="%Y%m%d"), columns=cols).astype(float) / 100
    return (df ** 2).resample("ME").sum()


def factor_data():
    import pandas as pd
    ff = french_monthly("F-F_Research_Data_5_Factors_2x3").join(french_monthly("F-F_Momentum_Factor"), how="inner")
    ff.columns = [c.strip() for c in ff.columns]
    fred = {}
    for s in ["GS10", "TB3MS", "BAA", "AAA"]:
        f = pd.read_csv(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}", index_col=0, parse_dates=True).iloc[:, 0]
        fred[s] = f.resample("ME").last()
    macro = pd.DataFrame(fred)
    ff["term_spread"] = (macro["GS10"] - macro["TB3MS"]).reindex(ff.index) / 100
    ff["default_spread"] = (macro["BAA"] - macro["AAA"]).reindex(ff.index) / 100
    rv = french_daily_realised_var("F-F_Research_Data_5_Factors_2x3_daily").join(french_daily_realised_var("F-F_Momentum_Factor_daily"), how="inner")
    rv.columns = ["rv_" + c.strip() for c in rv.columns]
    ff = ff.join(rv.drop(columns=["rv_RF"]), how="left")
    return ff.round(8)


def dl(tickers, start):
    px = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    return px.dropna(how="all").round(4)


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    dl(SECTORS + ["SPY"], "2005-01-01").to_csv(DATA / "sector_etfs.csv")
    dl(LARGE_CAPS + ["SPY"], "2015-01-01").to_csv(DATA / "large_caps.csv")
    dl(MACRO, "2008-01-01").to_csv(DATA / "macro.csv")
    factor_data().to_csv(DATA / "ff_factors_monthly.csv")
    print("written to", DATA)
