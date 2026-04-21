import numpy as np
import pandas as pd
import yfinance as yf


def build_monthly_market_cap_from_daily(
    daily_df: pd.DataFrame,
    price_col_priority=("Adj Close", "Close"),
    date_col="Date",
    ticker_col="Ticker",
) -> pd.DataFrame:

    df = daily_df.copy()

    if ticker_col not in df.columns:
        raise ValueError(f"daily_df must contain '{ticker_col}'.")
    if date_col not in df.columns:
        raise ValueError(f"daily_df must contain '{date_col}'.")

    price_col = None
    for c in price_col_priority:
        if c in df.columns:
            price_col = c
            break

    if price_col is None:
        raise ValueError(f"Missing price columns: {price_col_priority}")

    df[date_col] = pd.to_datetime(df[date_col])
    df["YearMonth"] = df[date_col].dt.to_period("M").astype(str)

    df = df.sort_values([ticker_col, date_col])
    month_end = (
        df.groupby([ticker_col, "YearMonth"], as_index=False)
        .tail(1)[[ticker_col, "YearMonth", price_col]]
        .rename(columns={price_col: "MonthEndPrice"})
    )

    tickers = month_end[ticker_col].dropna().unique().tolist()
    share_map = {}

    for t in tickers:
        try:
            info = yf.Ticker(t).info
            shares = info.get("sharesOutstanding", np.nan)
            share_map[t] = float(shares) if shares is not None else np.nan
        except Exception:
            share_map[t] = np.nan

    month_end["ShareOutstanding"] = month_end[ticker_col].map(share_map)
    month_end["MonthEnd_Market_Cap"] = (
        month_end["MonthEndPrice"] * month_end["ShareOutstanding"]
    )

    return month_end


def add_market_cap_to_monthly(monthly_df: pd.DataFrame, mcap_df: pd.DataFrame) -> pd.DataFrame:
    out = monthly_df.copy()

    for col in ["Ticker", "YearMonth"]:
        if col not in out.columns:
            raise ValueError(f"monthly_df must contain '{col}'.")
        if col not in mcap_df.columns:
            raise ValueError(f"mcap_df must contain '{col}'.")

    out = out.merge(
        mcap_df[["Ticker", "YearMonth", "MonthEnd_Market_Cap"]],
        on=["Ticker", "YearMonth"],
        how="left",
        validate="one_to_one"
    )
    return out


def clean_and_prepare(monthly_feature: pd.DataFrame) -> pd.DataFrame:
    df = monthly_feature.copy()

    numeric_cols = [
        "Monthly_Return",
        "Monthly_Volatility",
        "Log_Volume",
        "MonthEnd_Market_Cap"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["Monthly_Volatility"] >= 0]
    df.loc[df["MonthEnd_Market_Cap"] <= 0, "MonthEnd_Market_Cap"] = np.nan
    df["Log_Market_Cap"] = np.log(df["MonthEnd_Market_Cap"])

    df = df.dropna(subset=[
        "Monthly_Return",
        "Monthly_Volatility",
        "Log_Volume",
        "MonthEnd_Market_Cap"
    ])

    return df