import pandas as pd
import yfinance as yf
import requests
from io import StringIO


def fetch_sp100_tickers():
    url = "https://en.wikipedia.org/wiki/S%26P_100"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    html = requests.get(url, headers=headers, timeout=30).text
    tables = pd.read_html(StringIO(html))

    components = None
    for t in tables:
        if "Symbol" in t.columns:
            components = t
            break

    if components is None:
        raise ValueError("Could not find S&P 100 table.")

    tickers_raw = components["Symbol"].astype(str).tolist()
    tickers = [x.replace(".", "-").strip() for x in tickers_raw]
    return tickers


def download_daily_data(tickers, start_date, end_date):
    dfs = []
    failed = []

    for t in tickers:
        try:
            df = yf.download(
                t,
                start=start_date,
                end=end_date,
                auto_adjust=False,
                progress=False,
                threads=False
            )

            if df is None or df.empty:
                failed.append(t)
                continue

            df = df.reset_index()
            df["Ticker"] = t
            dfs.append(df)

        except Exception:
            failed.append(t)

    if not dfs:
        raise RuntimeError("No data downloaded.")

    daily = pd.concat(dfs, ignore_index=True)
    daily["Date"] = pd.to_datetime(daily["Date"])

    return daily, failed


def reshape_to_long_format(daily):
    daily2 = daily.copy()
    mi_cols = [c for c in daily2.columns if isinstance(c, tuple)]

    wide = daily2.loc[:, mi_cols].copy()
    wide.columns = pd.MultiIndex.from_tuples(wide.columns, names=["Price", "Ticker"])

    keep_prices = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
    wide = wide.loc[:, wide.columns.get_level_values("Price").isin(keep_prices)]

    wide.index = pd.to_datetime(daily2["Date"])
    wide.index.name = "Date"

    daily_long = wide.stack(level="Ticker", future_stack=True).reset_index()
    daily_long = daily_long.dropna(
        subset=["Close", "Adj Close", "Open", "High", "Low", "Volume"],
        how="all"
    )

    if "Adj Close" in daily_long.columns and "Close" in daily_long.columns:
        daily_long["Adj Close"] = daily_long["Adj Close"].fillna(daily_long["Close"])

    daily_long = daily_long.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return daily_long