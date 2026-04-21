from pathlib import Path

import pandas as pd
import yfinance as yf


def _extract_price_series(raw_data: pd.DataFrame, symbol: str) -> pd.Series:
    if raw_data.empty:
        raise ValueError(f"No price data returned for benchmark {symbol}.")

    if isinstance(raw_data.columns, pd.MultiIndex):
        for field_name in ("Adj Close", "Close"):
            matches = [col for col in raw_data.columns if col[0] == field_name]
            symbol_match = next((col for col in matches if len(col) > 1 and col[1] == symbol), None)
            if symbol_match is not None:
                return raw_data[symbol_match].rename("price")
            if matches:
                return raw_data[matches[0]].rename("price")
    else:
        for field_name in ("Adj Close", "Close"):
            if field_name in raw_data.columns:
                return raw_data[field_name].rename("price")

    raise ValueError(f"Could not find an adjusted or close price series for benchmark {symbol}.")


def fetch_benchmark_monthly_returns(
    symbol: str = "SPY",
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start_date) if start_date is not None else None
    end_ts = pd.Timestamp(end_date) if end_date is not None else None

    download_end = end_ts + pd.Timedelta(days=1) if end_ts is not None else None
    raw_data = yf.download(
        symbol,
        start=start_ts.strftime("%Y-%m-%d") if start_ts is not None else None,
        end=download_end.strftime("%Y-%m-%d") if download_end is not None else None,
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    price_series = _extract_price_series(raw_data, symbol).dropna()
    price_series.index = pd.to_datetime(price_series.index)
    price_series.index.name = "YearMonth"
    monthly = (
        price_series.to_frame()
        .sort_index()
        .resample("M")
        .last()
        .dropna()
        .reset_index()
    )
    monthly["YearMonth"] = pd.to_datetime(monthly["YearMonth"])

    monthly["benchmark_monthly_return"] = monthly["price"].pct_change()
    monthly["benchmark_cumulative_return"] = (1 + monthly["benchmark_monthly_return"].fillna(0)).cumprod() - 1
    monthly["benchmark_symbol"] = symbol
    monthly = monthly.drop(columns=["price"])

    if start_ts is not None:
        monthly = monthly[monthly["YearMonth"] >= start_ts].copy()
    if end_ts is not None:
        monthly = monthly[monthly["YearMonth"] <= end_ts].copy()

    if cache_path is not None:
        cache_path = Path(cache_path)
        monthly.to_csv(cache_path, index=False)

    return monthly.reset_index(drop=True)


def build_portfolio_benchmark_comparison(
    portfolio_df: pd.DataFrame,
    benchmark_symbol: str = "SPY",
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    if portfolio_df.empty:
        return pd.DataFrame()

    data = portfolio_df.copy()
    data["YearMonth"] = pd.to_datetime(data["YearMonth"])

    benchmark_monthly = fetch_benchmark_monthly_returns(
        symbol=benchmark_symbol,
        start_date=data["YearMonth"].min(),
        end_date=data["YearMonth"].max(),
        cache_path=cache_path,
    )

    comparison_frames = []
    for (model_name, evaluation_name), group in data.groupby(["model_name", "evaluation_name"], dropna=False):
        aligned = (
            group.sort_values("YearMonth")
            .merge(
                benchmark_monthly[["YearMonth", "benchmark_monthly_return"]],
                on="YearMonth",
                how="inner",
            )
        )
        if aligned.empty:
            continue

        aligned["benchmark_cumulative_return"] = (1 + aligned["benchmark_monthly_return"]).cumprod() - 1
        aligned["benchmark_symbol"] = benchmark_symbol
        aligned["excess_monthly_return"] = aligned["monthly_return"] - aligned["benchmark_monthly_return"]
        aligned["excess_cumulative_return"] = aligned["cumulative_return"] - aligned["benchmark_cumulative_return"]
        aligned["model_name"] = model_name
        aligned["evaluation_name"] = evaluation_name
        comparison_frames.append(aligned)

    if not comparison_frames:
        return pd.DataFrame()

    return pd.concat(comparison_frames, ignore_index=True)


def summarize_portfolio_benchmark_comparison(comparison_df: pd.DataFrame) -> pd.DataFrame:
    if comparison_df.empty:
        return pd.DataFrame()

    summary_rows = []
    for (model_name, evaluation_name), group in comparison_df.groupby(["model_name", "evaluation_name"], dropna=False):
        ordered = group.sort_values("YearMonth").reset_index(drop=True)
        last_row = ordered.iloc[-1]
        summary_rows.append(
            {
                "model_name": model_name,
                "evaluation_name": evaluation_name,
                "benchmark_symbol": last_row["benchmark_symbol"],
                "tracking_months": int(ordered.shape[0]),
                "portfolio_cumulative_return": last_row["cumulative_return"],
                "benchmark_cumulative_return": last_row["benchmark_cumulative_return"],
                "excess_cumulative_return": last_row["excess_cumulative_return"],
                "avg_monthly_portfolio_return": ordered["monthly_return"].mean(),
                "avg_monthly_benchmark_return": ordered["benchmark_monthly_return"].mean(),
                "avg_monthly_excess_return": ordered["excess_monthly_return"].mean(),
            }
        )

    return pd.DataFrame(summary_rows)
