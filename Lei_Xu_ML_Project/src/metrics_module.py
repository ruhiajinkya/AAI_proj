import pandas as pd


def build_ranked_portfolio(prediction_df: pd.DataFrame, rank_threshold: float = 0.8) -> pd.DataFrame:
    data = prediction_df.copy()
    data["YearMonth"] = pd.to_datetime(data["YearMonth"])
    return data[data["pred_rank"] >= rank_threshold].copy()


def build_portfolio_time_series(
    prediction_df: pd.DataFrame,
    rank_threshold: float = 0.8,
    model_name: str | None = None,
    evaluation_name: str | None = None,
) -> pd.DataFrame:
    portfolio_df = build_ranked_portfolio(prediction_df, rank_threshold=rank_threshold)
    monthly = (
        portfolio_df.groupby("YearMonth")
        .agg(
            monthly_return=("future_return_1m", "mean"),
            picks=("future_return_1m", "size"),
        )
        .reset_index()
        .sort_values("YearMonth")
    )

    if monthly.empty:
        monthly = pd.DataFrame(columns=["YearMonth", "monthly_return", "picks"])

    monthly["cumulative_return"] = (1 + monthly["monthly_return"]).cumprod() - 1 if not monthly.empty else pd.Series(dtype=float)
    monthly["model_name"] = model_name
    monthly["evaluation_name"] = evaluation_name
    return monthly


def compute_portfolio_metrics(prediction_df: pd.DataFrame, rank_threshold: float = 0.8):
    monthly = build_portfolio_time_series(prediction_df, rank_threshold=rank_threshold)

    if monthly.empty:
        return {
            "portfolio_months": 0,
            "avg_monthly_return": None,
            "monthly_return_std": None,
            "monthly_win_rate": None,
            "cumulative_return": None,
            "annualized_sharpe": None,
            "best_month_return": None,
            "worst_month_return": None,
            "avg_monthly_picks": 0.0,
        }

    return_std = monthly["monthly_return"].std()
    annualized_sharpe = None
    if pd.notna(return_std) and return_std > 0:
        annualized_sharpe = (monthly["monthly_return"].mean() / return_std) * (12 ** 0.5)

    return {
        "portfolio_months": int(monthly.shape[0]),
        "avg_monthly_return": monthly["monthly_return"].mean(),
        "monthly_return_std": return_std,
        "monthly_win_rate": (monthly["monthly_return"] > 0).mean(),
        "cumulative_return": monthly["cumulative_return"].iloc[-1],
        "annualized_sharpe": annualized_sharpe,
        "best_month_return": monthly["monthly_return"].max(),
        "worst_month_return": monthly["monthly_return"].min(),
        "avg_monthly_picks": monthly["picks"].mean(),
    }


def metrics_list_to_frame(metrics_list) -> pd.DataFrame:
    return pd.DataFrame(metrics_list)
