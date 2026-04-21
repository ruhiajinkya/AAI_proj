import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from src.model_module import (
        BASE_FEATURE_COLUMNS as LOGISTIC_FEATURE_COLUMNS,
        RANDOM_FOREST_FEATURE_COLUMNS,
        XGBOOST_FEATURE_COLUMNS,
        logistic_regression_factory,
        random_forest_factory,
        xgboost_factory,
    )
except ImportError:
    from model_module import (
        BASE_FEATURE_COLUMNS as LOGISTIC_FEATURE_COLUMNS,
        RANDOM_FOREST_FEATURE_COLUMNS,
        XGBOOST_FEATURE_COLUMNS,
        logistic_regression_factory,
        random_forest_factory,
        xgboost_factory,
    )
try:
    from src.metrics_module import compute_portfolio_metrics
except ImportError:
    from metrics_module import compute_portfolio_metrics


TARGET_COLUMN = "label_top20"
DEFAULT_THRESHOLD_STRATEGIES = ("target_rate", "balanced_accuracy", "f1")

def prepare_walk_forward_data(
    df: pd.DataFrame,
    feature_columns,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    required_columns = list(feature_columns) + [target_column, "YearMonth", "future_return_1m"]
    data = df.dropna(subset=required_columns).copy()
    data["YearMonth"] = pd.to_datetime(data["YearMonth"])
    data = data.sort_values(["YearMonth", "Ticker"]).reset_index(drop=True)
    return data


def build_walk_forward_slices(
    df: pd.DataFrame,
    min_train_months: int = 24,
    test_months: int = 1,
):
    unique_months = sorted(df["YearMonth"].drop_duplicates())
    slices = []

    for train_end_idx in range(min_train_months, len(unique_months) - test_months + 1):
        train_months = unique_months[:train_end_idx]
        test_window = unique_months[train_end_idx:train_end_idx + test_months]

        train_df = df[df["YearMonth"].isin(train_months)].copy()
        test_df = df[df["YearMonth"].isin(test_window)].copy()

        if train_df.empty or test_df.empty:
            continue
        if train_df[TARGET_COLUMN].nunique() < 2:
            continue

        slices.append((train_df, test_df))

    return slices


def choose_threshold_from_training_predictions(
    y_true: pd.Series,
    y_prob,
    strategy: str = "target_rate",
    target_positive_rate: float | None = None,
):
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_prob = pd.Series(y_prob).reset_index(drop=True)

    if strategy == "fixed":
        return 0.5

    if strategy == "target_rate":
        positive_rate = target_positive_rate if target_positive_rate is not None else float(y_true.mean())
        positive_rate = min(max(positive_rate, 0.01), 0.99)
        return float(y_prob.quantile(1 - positive_rate))

    candidate_thresholds = sorted(set([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] + y_prob.quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).tolist()))
    best_threshold = 0.5
    best_score = float("-inf")

    for threshold in candidate_thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        if strategy == "balanced_accuracy":
            score = balanced_accuracy_score(y_true, y_pred)
        elif strategy == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unsupported threshold strategy: {strategy}")

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold


def run_walk_forward_validation(
    df: pd.DataFrame,
    feature_columns,
    model_factory,
    model_name: str,
    min_train_months: int = 24,
    test_months: int = 1,
    target_column: str = TARGET_COLUMN,
    threshold_strategy: str = "target_rate",
    target_positive_rate: float | None = 0.2,
):
    data = prepare_walk_forward_data(df, feature_columns, target_column=target_column)
    folds = build_walk_forward_slices(data, min_train_months=min_train_months, test_months=test_months)
    predictions = []
    fold_thresholds = []

    for fold_number, (train_df, test_df) in enumerate(folds, start=1):
        X_train = train_df[list(feature_columns)]
        y_train = train_df[target_column]
        X_test = test_df[list(feature_columns)]

        model = model_factory()
        model.fit(X_train, y_train)

        train_prob = model.predict_proba(X_train)[:, 1]
        decision_threshold = choose_threshold_from_training_predictions(
            y_true=y_train,
            y_prob=train_prob,
            strategy=threshold_strategy,
            target_positive_rate=target_positive_rate,
        )

        fold_predictions = test_df.copy()
        fold_predictions["pred_prob"] = model.predict_proba(X_test)[:, 1]
        fold_predictions["pred_label"] = (fold_predictions["pred_prob"] >= decision_threshold).astype(int)
        fold_predictions["decision_threshold"] = decision_threshold
        fold_predictions["fold"] = fold_number
        predictions.append(fold_predictions)
        fold_thresholds.append(decision_threshold)

    if not predictions:
        raise ValueError("No valid walk-forward folds were created. Check the date range and class distribution.")

    prediction_df = pd.concat(predictions, ignore_index=True)
    prediction_df["pred_rank"] = prediction_df.groupby("YearMonth")["pred_prob"].rank(pct=True)

    y_true = prediction_df[target_column]
    y_pred = prediction_df["pred_label"]
    y_prob = prediction_df["pred_prob"]
    portfolio_metrics = compute_portfolio_metrics(prediction_df)

    metrics = {
        "model_name": model_name,
        "num_folds": int(prediction_df["fold"].nunique()),
        "test_months": int(prediction_df["YearMonth"].nunique()),
        "threshold_strategy": threshold_strategy,
        "avg_decision_threshold": float(pd.Series(fold_thresholds).mean()),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
        "top20_avg_return": prediction_df.loc[prediction_df["pred_rank"] >= 0.8, "future_return_1m"].mean(),
        "overall_avg_return": prediction_df["future_return_1m"].mean(),
        "actual_positive_rate": y_true.mean(),
        "predicted_positive_rate": y_pred.mean(),
        **portfolio_metrics,
    }

    return prediction_df, metrics


def print_walk_forward_summary(metrics):
    print(f"=== {metrics['model_name']} Walk-Forward ===")
    print(f"Folds: {metrics['num_folds']}")
    print(f"Test months: {metrics['test_months']}")
    print(f"Threshold strategy: {metrics['threshold_strategy']}")
    print(f"Average decision threshold: {metrics['avg_decision_threshold']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"Actual positive rate: {metrics['actual_positive_rate']:.4f}")
    print(f"Predicted positive rate: {metrics['predicted_positive_rate']:.4f}")
    print(metrics["classification_report"])
    print(f"Top20 avg return: {metrics['top20_avg_return']:.6f}")
    print(f"Overall avg return: {metrics['overall_avg_return']:.6f}")
    print(f"Portfolio months: {metrics['portfolio_months']}")
    print(f"Average monthly portfolio return: {metrics['avg_monthly_return']:.6f}")
    print(f"Monthly return std: {metrics['monthly_return_std']:.6f}")
    print(f"Monthly win rate: {metrics['monthly_win_rate']:.4f}")
    print(f"Cumulative portfolio return: {metrics['cumulative_return']:.6f}")
    if metrics["annualized_sharpe"] is not None:
        print(f"Annualized Sharpe: {metrics['annualized_sharpe']:.4f}")
    print(f"Best month return: {metrics['best_month_return']:.6f}")
    print(f"Worst month return: {metrics['worst_month_return']:.6f}")
    print(f"Average monthly picks: {metrics['avg_monthly_picks']:.2f}")


def compare_threshold_strategies(
    df: pd.DataFrame,
    feature_columns,
    model_factory,
    model_name: str,
    threshold_strategies=DEFAULT_THRESHOLD_STRATEGIES,
):
    results = []

    for strategy in threshold_strategies:
        _, metrics = run_walk_forward_validation(
            df=df,
            feature_columns=feature_columns,
            model_factory=model_factory,
            model_name=model_name,
            threshold_strategy=strategy,
        )
        results.append(metrics)

    return results


def print_strategy_comparison(results):
    summary_df = pd.DataFrame([
        {
            "strategy": item["threshold_strategy"],
            "bal_acc": item["balanced_accuracy"],
            "f1": item["f1"],
            "precision": item["precision"],
            "recall": item["recall"],
            "roc_auc": item["roc_auc"],
            "pred_rate": item["predicted_positive_rate"],
            "avg_monthly_ret": item["avg_monthly_return"],
            "cum_ret": item["cumulative_return"],
            "sharpe": item["annualized_sharpe"],
        }
        for item in results
    ])

    print("Strategy Comparison")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main():
    df = pd.read_csv("sp100_monthly_ml_dataset.csv")

    logistic_results = compare_threshold_strategies(
        df=df,
        feature_columns=LOGISTIC_FEATURE_COLUMNS,
        model_factory=logistic_regression_factory,
        model_name="Logistic Regression",
    )
    for metrics in logistic_results:
        print_walk_forward_summary(metrics)
    print_strategy_comparison(logistic_results)

    random_forest_results = compare_threshold_strategies(
        df=df,
        feature_columns=RANDOM_FOREST_FEATURE_COLUMNS,
        model_factory=random_forest_factory,
        model_name="Random Forest",
    )
    for metrics in random_forest_results:
        print_walk_forward_summary(metrics)
    print_strategy_comparison(random_forest_results)

    try:
        xgboost_results = compare_threshold_strategies(
            df=df,
            feature_columns=XGBOOST_FEATURE_COLUMNS,
            model_factory=xgboost_factory,
            model_name="XGBoost",
        )
        for metrics in xgboost_results:
            print_walk_forward_summary(metrics)
        print_strategy_comparison(xgboost_results)
    except ImportError as exc:
        print(f"Skipping XGBoost walk-forward: {exc}")


if __name__ == "__main__":
    main()
