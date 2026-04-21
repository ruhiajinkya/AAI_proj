import numpy as np
import pandas as pd


def build_monthly_features_from_daily(daily_long: pd.DataFrame) -> pd.DataFrame:
    df = daily_long.copy()

    if "Adj Close" in df.columns:
        price_col = "Adj Close"
    elif "Close" in df.columns:
        price_col = "Close"
    else:
        raise ValueError("Neither 'Adj Close' nor 'Close' exists.")

    df = df.sort_values(["Ticker", "Date"])
    df["Daily_Return"] = df.groupby("Ticker")[price_col].pct_change()
    df["YearMonth"] = pd.to_datetime(df["Date"]).dt.to_period("M").astype(str)

    monthly = df.groupby(["Ticker", "YearMonth"]).agg(
        Monthly_Return=(price_col, lambda x: x.iloc[-1] / x.iloc[0] - 1),
        Monthly_Volatility=("Daily_Return", "std"),
        Avg_Monthly_Volume=("Volume", "mean")
    ).reset_index()

    monthly["Log_Volume"] = np.log(monthly["Avg_Monthly_Volume"].replace(0, np.nan))
    return monthly