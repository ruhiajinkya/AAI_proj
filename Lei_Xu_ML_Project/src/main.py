import argparse
import pandas as pd

from config import (
    START_DATE, END_DATE,
    DAILY_FILE, MONTHLY_FILE, MONTHLY_FEATURE_FILE, ML_FILE, METRICS_DIR, PLOTS_DIR, RAW_DIR
)

try:
    from src.benchmark_module import (
        build_portfolio_benchmark_comparison,
        summarize_portfolio_benchmark_comparison,
    )
    from src.data_loader import fetch_sp100_tickers, download_daily_data, reshape_to_long_format
    from src.preprocess import build_monthly_features_from_daily
    from src.market_cap import build_monthly_market_cap_from_daily, add_market_cap_to_monthly, clean_and_prepare
    from src.feature_engineering import build_ml_dataset
    from src.model_module import (
        ALL_MODEL_FEATURE_COLUMNS,
        run_standard_models,
        logistic_regression_factory,
        random_forest_factory,
        xgboost_factory,
    )
    from src.metrics_module import metrics_list_to_frame, build_portfolio_time_series
    from src.plot_module import plot_cumulative_returns, plot_model_comparison, plot_portfolio_vs_benchmark
    from src.walk_forward_model import (
        LOGISTIC_FEATURE_COLUMNS as WALK_FORWARD_LOGISTIC_FEATURE_COLUMNS,
        RANDOM_FOREST_FEATURE_COLUMNS as WALK_FORWARD_RANDOM_FOREST_FEATURE_COLUMNS,
        XGBOOST_FEATURE_COLUMNS as WALK_FORWARD_XGBOOST_FEATURE_COLUMNS,
        print_walk_forward_summary,
        run_walk_forward_validation,
    )
except ImportError:
    from benchmark_module import (
        build_portfolio_benchmark_comparison,
        summarize_portfolio_benchmark_comparison,
    )
    from data_loader import fetch_sp100_tickers, download_daily_data, reshape_to_long_format
    from preprocess import build_monthly_features_from_daily
    from market_cap import build_monthly_market_cap_from_daily, add_market_cap_to_monthly, clean_and_prepare
    from feature_engineering import build_ml_dataset
    from model_module import (
        ALL_MODEL_FEATURE_COLUMNS,
        run_standard_models,
        logistic_regression_factory,
        random_forest_factory,
        xgboost_factory,
    )
    from metrics_module import metrics_list_to_frame, build_portfolio_time_series
    from plot_module import plot_cumulative_returns, plot_model_comparison, plot_portfolio_vs_benchmark
    from walk_forward_model import (
        LOGISTIC_FEATURE_COLUMNS as WALK_FORWARD_LOGISTIC_FEATURE_COLUMNS,
        RANDOM_FOREST_FEATURE_COLUMNS as WALK_FORWARD_RANDOM_FOREST_FEATURE_COLUMNS,
        XGBOOST_FEATURE_COLUMNS as WALK_FORWARD_XGBOOST_FEATURE_COLUMNS,
        print_walk_forward_summary,
        run_walk_forward_validation,
    )


def build_or_rebuild_ml_dataset():
    print("Step 1: Fetch tickers...")
    tickers = fetch_sp100_tickers()
    print(f"Fetched {len(tickers)} tickers.")

    print("Step 2: Download daily data...")
    daily_raw, failed = download_daily_data(tickers, START_DATE, END_DATE)
    print(f"Failed tickers: {len(failed)}")

    print("Step 3: Reshape to long format...")
    daily_long = reshape_to_long_format(daily_raw)
    daily_long.to_csv(DAILY_FILE, index=False)
    print(f"Saved daily data to: {DAILY_FILE}")

    print("Step 4: Build monthly features...")
    monthly = build_monthly_features_from_daily(daily_long)
    monthly.to_csv(MONTHLY_FILE, index=False)
    print(f"Saved monthly data to: {MONTHLY_FILE}")

    print("Step 5: Build market cap...")
    mcap_df = build_monthly_market_cap_from_daily(daily_long)

    print("Step 6: Merge and clean...")
    monthly_feature = add_market_cap_to_monthly(monthly, mcap_df)
    monthly_feature = clean_and_prepare(monthly_feature)
    monthly_feature.to_csv(MONTHLY_FEATURE_FILE, index=False)
    print(f"Saved monthly feature data to: {MONTHLY_FEATURE_FILE}")

    print("Step 7: Build ML dataset...")
    ml_df = build_ml_dataset(monthly_feature)
    print("ML dataset shape:", ml_df.shape)
    try:
        ml_df.to_csv(ML_FILE, index=False)
        print(f"Saved ML dataset to: {ML_FILE}")
    except PermissionError:
        print(f"Could not write ML dataset to {ML_FILE}. Continuing with rebuilt dataset in memory.")
    return ml_df


def rebuild_ml_dataset_from_local_features():
    print(f"Rebuilding ML dataset from existing monthly feature file: {MONTHLY_FEATURE_FILE}")
    monthly_feature = pd.read_csv(MONTHLY_FEATURE_FILE)
    ml_df = build_ml_dataset(monthly_feature)
    print("ML dataset shape:", ml_df.shape)
    try:
        ml_df.to_csv(ML_FILE, index=False)
        print(f"Saved refreshed ML dataset to: {ML_FILE}")
    except PermissionError:
        print(f"Could not write refreshed ML dataset to {ML_FILE}. Continuing with rebuilt dataset in memory.")
    return ml_df


def backfill_missing_ml_features(ml_df: pd.DataFrame) -> pd.DataFrame:
    repaired_df = ml_df.copy()

    if (
        "momentum_volume_interaction" not in repaired_df.columns
        and {"ret_3m_rank", "vol_3m_rank"}.issubset(repaired_df.columns)
    ):
        repaired_df["momentum_volume_interaction"] = repaired_df["ret_3m_rank"] * repaired_df["vol_3m_rank"]

    return repaired_df


def run_walk_forward_models(ml_df):
    print("Step 9: Run walk-forward models for imbalanced labels...")
    results = []

    logistic_predictions, logistic_metrics = run_walk_forward_validation(
        df=ml_df,
        feature_columns=WALK_FORWARD_LOGISTIC_FEATURE_COLUMNS,
        model_factory=logistic_regression_factory,
        model_name="Logistic Regression",
    )
    print_walk_forward_summary(logistic_metrics)
    results.append({
        "model_name": "Logistic Regression",
        "evaluation_name": "walk_forward",
        "metrics": {"model_name": "Logistic Regression", "evaluation_name": "walk_forward", **logistic_metrics},
        "predictions": logistic_predictions.assign(model_name="Logistic Regression"),
    })

    random_forest_predictions, random_forest_metrics = run_walk_forward_validation(
        df=ml_df,
        feature_columns=WALK_FORWARD_RANDOM_FOREST_FEATURE_COLUMNS,
        model_factory=random_forest_factory,
        model_name="Random Forest",
    )
    print_walk_forward_summary(random_forest_metrics)
    results.append({
        "model_name": "Random Forest",
        "evaluation_name": "walk_forward",
        "metrics": {"model_name": "Random Forest", "evaluation_name": "walk_forward", **random_forest_metrics},
        "predictions": random_forest_predictions.assign(model_name="Random Forest"),
    })

    try:
        xgboost_predictions, xgboost_metrics = run_walk_forward_validation(
            df=ml_df,
            feature_columns=WALK_FORWARD_XGBOOST_FEATURE_COLUMNS,
            model_factory=xgboost_factory,
            model_name="XGBoost",
        )
        print_walk_forward_summary(xgboost_metrics)
        results.append({
            "model_name": "XGBoost",
            "evaluation_name": "walk_forward",
            "metrics": {"model_name": "XGBoost", "evaluation_name": "walk_forward", **xgboost_metrics},
            "predictions": xgboost_predictions.assign(model_name="XGBoost"),
        })
    except ImportError as exc:
        print(f"Skipping XGBoost walk-forward: {exc}")

    return results


def export_metrics_and_plots(standard_results, walk_forward_results):
    all_results = standard_results + walk_forward_results
    metrics_df = metrics_list_to_frame([item["metrics"] for item in all_results])
    metrics_df.to_csv(METRICS_DIR / "model_metrics_summary.csv", index=False)

    walk_forward_metrics_df = metrics_list_to_frame([item["metrics"] for item in walk_forward_results])
    walk_forward_metrics_df.to_csv(METRICS_DIR / "walk_forward_metrics_summary.csv", index=False)

    portfolio_series = []
    selected_stock_frames = []
    for item in all_results:
        prediction_df = item["predictions"].copy()
        prediction_df["evaluation_name"] = item["evaluation_name"]
        selected_stock_frames.append(
            prediction_df[["YearMonth", "Ticker", "pred_prob", "pred_rank"]]
            .rename(columns={"pred_prob": "score", "pred_rank": "rank"})
            .assign(model_name=item["model_name"], evaluation_name=item["evaluation_name"])
        )

    selected_stock_df = pd.concat(selected_stock_frames, ignore_index=True)
    selected_stock_df["YearMonth"] = pd.to_datetime(selected_stock_df["YearMonth"]).dt.strftime("%Y-%m-%d")
    selected_stock_df = selected_stock_df[
        ["YearMonth", "evaluation_name", "model_name", "Ticker", "score", "rank"]
    ].sort_values(["YearMonth", "evaluation_name", "model_name", "rank"], ascending=[True, True, True, False])
    selected_stock_df.to_csv(METRICS_DIR / "selected_stock.csv", index=False)

    for item in all_results:
        prediction_df = item["predictions"].copy()
        prediction_df["evaluation_name"] = item["evaluation_name"]
        portfolio_series.append(
            build_portfolio_time_series(
                prediction_df,
                model_name=item["model_name"],
                evaluation_name=item["evaluation_name"],
            )
        )

    portfolio_df = pd.concat(portfolio_series, ignore_index=True)
    portfolio_df.to_csv(METRICS_DIR / "portfolio_time_series.csv", index=False)

    walk_forward_portfolio_df = portfolio_df[portfolio_df["evaluation_name"] == "walk_forward"].copy()

    benchmark_comparison_df = pd.DataFrame()
    try:
        benchmark_comparison_df = build_portfolio_benchmark_comparison(
            walk_forward_portfolio_df,
            benchmark_symbol="SPY",
            cache_path=RAW_DIR / "spy_monthly_benchmark.csv",
        )
        benchmark_comparison_df.to_csv(METRICS_DIR / "portfolio_vs_spy.csv", index=False)
        benchmark_summary_df = summarize_portfolio_benchmark_comparison(benchmark_comparison_df)
        benchmark_summary_df.to_csv(METRICS_DIR / "portfolio_vs_spy_summary.csv", index=False)
    except Exception as exc:
        print(f"Skipping SPY benchmark comparison: {exc}")

    plot_cumulative_returns(
        walk_forward_portfolio_df,
        PLOTS_DIR / "cumulative_returns.png",
        "Walk-Forward Model Portfolio Cumulative Returns",
    )
    plot_portfolio_vs_benchmark(
        benchmark_comparison_df,
        PLOTS_DIR / "portfolio_vs_spy.png",
        "Walk-Forward Portfolio Cumulative Return vs SPY",
    )
    plot_model_comparison(
        walk_forward_metrics_df,
        PLOTS_DIR / "model_comparison.png",
        "Walk-Forward Model Comparison",
        value_columns=["roc_auc", "top20_avg_return", "cumulative_return", "annualized_sharpe"],
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Build the stock ML dataset and run model evaluations.")
    parser.add_argument(
        "--rebuild-data",
        action="store_true",
        help="Force a fresh download and rebuild of the dataset instead of reusing the saved ML CSV.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if ML_FILE.exists() and not args.rebuild_data:
        print(f"Step 1: Load existing ML dataset from: {ML_FILE}")
        ml_df = pd.read_csv(ML_FILE)
        ml_df = backfill_missing_ml_features(ml_df)
        print("ML dataset shape:", ml_df.shape)
        missing_feature_cols = [col for col in ALL_MODEL_FEATURE_COLUMNS if col not in ml_df.columns]
        if missing_feature_cols:
            print(f"Existing ML dataset is missing required model features: {missing_feature_cols}")
            if MONTHLY_FEATURE_FILE.exists():
                ml_df = rebuild_ml_dataset_from_local_features()
            else:
                print(f"Monthly feature file not found at {MONTHLY_FEATURE_FILE}. Building everything now.")
                ml_df = build_or_rebuild_ml_dataset()
    else:
        if args.rebuild_data:
            print("Rebuilding dataset because `--rebuild-data` was provided.")
        else:
            print(f"ML dataset not found at {ML_FILE}. Building it now.")
        ml_df = build_or_rebuild_ml_dataset()

    standard_results = run_standard_models(ml_df)
    walk_forward_results = run_walk_forward_models(ml_df)
    export_metrics_and_plots(standard_results, walk_forward_results)

    print("Done.")


if __name__ == "__main__":
    main()
