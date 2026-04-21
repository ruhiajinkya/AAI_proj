from pathlib import Path

import pandas as pd

try:
    from config import METRICS_DIR
except ImportError:
    METRICS_DIR = Path(__file__).resolve().parent / "artifacts" / "metrics"


TOP_RANK_THRESHOLD = 0.8


def get_metrics_dir() -> Path:
    return Path(METRICS_DIR)


def required_artifact_paths() -> dict[str, Path]:
    metrics_dir = get_metrics_dir()
    return {
        "selected_stock": metrics_dir / "selected_stock.csv",
        "portfolio_time_series": metrics_dir / "portfolio_time_series.csv",
        "walk_forward_metrics": metrics_dir / "walk_forward_metrics_summary.csv",
        "portfolio_vs_spy": metrics_dir / "portfolio_vs_spy.csv",
        "portfolio_vs_spy_summary": metrics_dir / "portfolio_vs_spy_summary.csv",
    }


def validate_demo_artifacts() -> list[str]:
    missing = []
    for name, path in required_artifact_paths().items():
        if name.startswith("portfolio_vs_spy"):
            continue
        if not path.exists():
            missing.append(str(path))
    return missing


def load_selected_stocks() -> pd.DataFrame:
    data = pd.read_csv(required_artifact_paths()["selected_stock"])
    data["YearMonth"] = pd.to_datetime(data["YearMonth"])
    data = data[data["evaluation_name"] == "walk_forward"].copy()
    data = data.sort_values(["model_name", "YearMonth", "rank"], ascending=[True, True, False])
    return data


def load_walk_forward_metrics() -> pd.DataFrame:
    data = pd.read_csv(required_artifact_paths()["walk_forward_metrics"])
    numeric_cols = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "top20_avg_return",
        "overall_avg_return",
        "actual_positive_rate",
        "predicted_positive_rate",
        "portfolio_months",
        "avg_monthly_return",
        "monthly_return_std",
        "monthly_win_rate",
        "cumulative_return",
        "annualized_sharpe",
        "best_month_return",
        "worst_month_return",
        "avg_monthly_picks",
    ]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    return data[data["evaluation_name"] == "walk_forward"].copy()


def load_portfolio_time_series() -> pd.DataFrame:
    data = pd.read_csv(required_artifact_paths()["portfolio_time_series"])
    data["YearMonth"] = pd.to_datetime(data["YearMonth"])
    data = data[data["evaluation_name"] == "walk_forward"].copy()
    return data.sort_values(["model_name", "YearMonth"]).reset_index(drop=True)


def load_portfolio_vs_spy() -> pd.DataFrame:
    path = required_artifact_paths()["portfolio_vs_spy"]
    if not path.exists():
        return pd.DataFrame()

    data = pd.read_csv(path)
    data["YearMonth"] = pd.to_datetime(data["YearMonth"])
    data = data[data["evaluation_name"] == "walk_forward"].copy()
    return data.sort_values(["model_name", "YearMonth"]).reset_index(drop=True)


def load_portfolio_vs_spy_summary() -> pd.DataFrame:
    path = required_artifact_paths()["portfolio_vs_spy_summary"]
    if not path.exists():
        return pd.DataFrame()

    data = pd.read_csv(path)
    return data[data["evaluation_name"] == "walk_forward"].copy()


def available_models(selected_df: pd.DataFrame) -> list[str]:
    return sorted(selected_df["model_name"].dropna().unique().tolist())


def available_months_for_model(selected_df: pd.DataFrame, model_name: str) -> list[pd.Timestamp]:
    model_df = selected_df[selected_df["model_name"] == model_name]
    return sorted(model_df["YearMonth"].dropna().unique().tolist())


def recommended_stocks_for_month(
    selected_df: pd.DataFrame,
    model_name: str,
    target_month,
    rank_threshold: float = TOP_RANK_THRESHOLD,
) -> pd.DataFrame:
    target_ts = pd.Timestamp(target_month)
    recommendations = selected_df[
        (selected_df["model_name"] == model_name)
        & (selected_df["YearMonth"] == target_ts)
        & (selected_df["rank"] >= rank_threshold)
    ].copy()
    return recommendations.sort_values("rank", ascending=False).reset_index(drop=True)


def model_metrics_row(metrics_df: pd.DataFrame, model_name: str) -> pd.Series:
    model_df = metrics_df[metrics_df["model_name"] == model_name]
    if model_df.empty:
        raise ValueError(f"No metrics found for model: {model_name}")
    return model_df.iloc[0]


def portfolio_history_for_model(portfolio_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    return portfolio_df[portfolio_df["model_name"] == model_name].copy()


def benchmark_history_for_model(benchmark_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    if benchmark_df.empty:
        return pd.DataFrame()
    return benchmark_df[benchmark_df["model_name"] == model_name].copy()


def benchmark_summary_row(summary_df: pd.DataFrame, model_name: str) -> pd.Series | None:
    if summary_df.empty:
        return None
    model_df = summary_df[summary_df["model_name"] == model_name]
    if model_df.empty:
        return None
    return model_df.iloc[0]
