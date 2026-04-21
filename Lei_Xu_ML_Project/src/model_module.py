import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from src.metrics_module import build_portfolio_time_series, compute_portfolio_metrics
except ImportError:
    from metrics_module import build_portfolio_time_series, compute_portfolio_metrics


BASE_FEATURE_COLUMNS = [
    "ret_1m_rank",
    "ret_3m_rank",
    "ret_6m_rank",
    "ret_12m_rank",
    "vol_1m",
    "vol_3m",
    "volume_ratio_1m_3m_rank",
    "log_market_cap_1m_rank",
    "market_cap_ratio_1m_3m_rank",
    "downside_vol_3m",
    "momentum_volume_interaction",
]

RANDOM_FOREST_FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + [
    "log_volume_change_1m",
    "log_market_cap_change_1m",
]

XGBOOST_FEATURE_COLUMNS = RANDOM_FOREST_FEATURE_COLUMNS
ALL_MODEL_FEATURE_COLUMNS = sorted(set(BASE_FEATURE_COLUMNS + RANDOM_FOREST_FEATURE_COLUMNS + XGBOOST_FEATURE_COLUMNS))

TARGET_COLUMN = "label_top20"


def load_ml_data(csv_path: str = "sp100_monthly_ml_dataset.csv") -> pd.DataFrame:
    return pd.read_csv(csv_path)


def prepare_features(df: pd.DataFrame, feature_columns, target_column: str = TARGET_COLUMN):
    data = df.dropna(subset=list(feature_columns) + [target_column]).copy()
    X = data[list(feature_columns)]
    y = data[target_column]
    return data, X, y


def time_split(df: pd.DataFrame, cutoff: str = "2024-01-01"):
    split_df = df.copy()
    split_df["YearMonth"] = split_df["YearMonth"].astype(str)
    train = split_df[split_df["YearMonth"] < cutoff].copy()
    test = split_df[split_df["YearMonth"] >= cutoff].copy()
    return train, test


def evaluate_classifier_predictions(y_test, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob) if y_prob is not None else None,
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }


def add_prediction_ranks(test_df: pd.DataFrame, y_prob):
    result = test_df.copy()
    result["pred_prob"] = y_prob
    result["pred_rank"] = result.groupby("YearMonth")["pred_prob"].rank(pct=True)
    return result


def compute_top20_avg_return(predictions_df: pd.DataFrame) -> float:
    top20 = predictions_df[predictions_df["pred_rank"] >= 0.8]
    return top20["future_return_1m"].mean()


def compute_overall_avg_return(predictions_df: pd.DataFrame) -> float:
    return predictions_df["future_return_1m"].mean()


def evaluate_top20_monthly_return(df: pd.DataFrame, X_test: pd.DataFrame, y_pred_proba):
    required_columns = {"YearMonth", "future_return_1m"}
    if y_pred_proba is None or not required_columns.issubset(df.columns):
        return None

    test = df.loc[X_test.index].copy()
    test["pred_prob"] = y_pred_proba
    test["pred_rank"] = test.groupby("YearMonth")["pred_prob"].rank(pct=True)
    top20 = test[test["pred_rank"] >= 0.8]
    return top20["future_return_1m"].mean()


def build_test_prediction_frame(df: pd.DataFrame, X_test: pd.DataFrame, y_pred_proba, model_name: str):
    test = df.loc[X_test.index].copy()
    test["pred_prob"] = y_pred_proba
    test["pred_rank"] = test.groupby("YearMonth")["pred_prob"].rank(pct=True)
    test["model_name"] = model_name
    return test


def logistic_regression_factory():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced")),
    ])


def random_forest_factory(random_state: int = 42):
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=5,
        random_state=random_state,
        class_weight="balanced",
    )


def xgboost_factory(random_state: int = 42):
    if XGBClassifier is None:
        raise ImportError("xgboost is not installed. Install it with `pip install xgboost` to run the XGBoost model.")

    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        scale_pos_weight=4.0,
    )


def train_test_model(
    df: pd.DataFrame,
    feature_columns,
    model_factory,
    target_column: str = TARGET_COLUMN,
    test_size: float = 0.2,
    random_state: int = 42,
):
    prepared_df, X, y = prepare_features(df, feature_columns, target_column=target_column)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if len(y.unique()) > 1 else None,
    )

    model = model_factory()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    metrics = evaluate_classifier_predictions(y_test, y_pred, y_prob)

    return prepared_df, model, metrics, X_test, y_test, y_prob


def train_time_split_model(
    df: pd.DataFrame,
    feature_columns,
    model_factory,
    target_column: str = TARGET_COLUMN,
    cutoff: str = "2024-01-01",
):
    prepared_df = df.dropna(subset=list(feature_columns) + [target_column]).copy()
    train_df, test_df = time_split(prepared_df, cutoff=cutoff)
    _, X_train, y_train = prepare_features(train_df, feature_columns, target_column=target_column)
    test_df, X_test, y_test = prepare_features(test_df, feature_columns, target_column=target_column)

    model = model_factory()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    metrics = evaluate_classifier_predictions(y_test, y_pred, y_prob)
    predictions = add_prediction_ranks(test_df, y_prob)

    return model, metrics, predictions, X_test, y_test


def print_standard_model_summary(model_name: str, metrics, top20_avg_return: float | None, overall_avg_return: float | None):
    print(f"=== {model_name} ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    if metrics["roc_auc"] is not None:
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(metrics["classification_report"])
    if top20_avg_return is not None:
        print(f"Top20 avg return: {top20_avg_return:.6f}")
    if overall_avg_return is not None:
        print(f"Overall avg return: {overall_avg_return:.6f}")
    if metrics.get("cumulative_return") is not None:
        print(f"Cumulative portfolio return: {metrics['cumulative_return']:.6f}")
    if metrics.get("annualized_sharpe") is not None:
        print(f"Annualized Sharpe: {metrics['annualized_sharpe']:.4f}")


def run_standard_models(ml_df: pd.DataFrame):
    print("Step 8: Train standard models...")
    results = []

    logistic_df, _, logistic_metrics, X_test, _, y_pred_proba = train_test_model(
        df=ml_df,
        feature_columns=BASE_FEATURE_COLUMNS,
        model_factory=logistic_regression_factory,
    )
    logistic_predictions = build_test_prediction_frame(logistic_df, X_test, y_pred_proba, "Logistic Regression")
    logistic_top20 = evaluate_top20_monthly_return(logistic_df, X_test, y_pred_proba)
    logistic_overall = logistic_df.loc[X_test.index, "future_return_1m"].mean()
    logistic_metrics.update(compute_portfolio_metrics(logistic_predictions))
    print_standard_model_summary(
        model_name="Logistic Regression",
        metrics=logistic_metrics,
        top20_avg_return=logistic_top20,
        overall_avg_return=logistic_overall,
    )
    results.append({
        "model_name": "Logistic Regression",
        "evaluation_name": "standard",
        "metrics": {"model_name": "Logistic Regression", "evaluation_name": "standard", "top20_avg_return": logistic_top20, "overall_avg_return": logistic_overall, **logistic_metrics},
        "predictions": logistic_predictions,
        "portfolio_series": build_portfolio_time_series(logistic_predictions, model_name="Logistic Regression", evaluation_name="standard"),
    })

    _, random_forest_metrics, random_forest_predictions, _, _ = train_time_split_model(
        df=ml_df,
        feature_columns=RANDOM_FOREST_FEATURE_COLUMNS,
        model_factory=random_forest_factory,
    )
    random_forest_metrics.update(compute_portfolio_metrics(random_forest_predictions))
    print_standard_model_summary(
        model_name="Random Forest",
        metrics=random_forest_metrics,
        top20_avg_return=compute_top20_avg_return(random_forest_predictions),
        overall_avg_return=compute_overall_avg_return(random_forest_predictions),
    )
    results.append({
        "model_name": "Random Forest",
        "evaluation_name": "standard",
        "metrics": {
            "model_name": "Random Forest",
            "evaluation_name": "standard",
            "top20_avg_return": compute_top20_avg_return(random_forest_predictions),
            "overall_avg_return": compute_overall_avg_return(random_forest_predictions),
            **random_forest_metrics,
        },
        "predictions": random_forest_predictions.assign(model_name="Random Forest"),
        "portfolio_series": build_portfolio_time_series(random_forest_predictions, model_name="Random Forest", evaluation_name="standard"),
    })

    try:
        _, xgboost_metrics, xgboost_predictions, _, _ = train_time_split_model(
            df=ml_df,
            feature_columns=XGBOOST_FEATURE_COLUMNS,
            model_factory=xgboost_factory,
        )
        xgboost_metrics.update(compute_portfolio_metrics(xgboost_predictions))
        print_standard_model_summary(
            model_name="XGBoost",
            metrics=xgboost_metrics,
            top20_avg_return=compute_top20_avg_return(xgboost_predictions),
            overall_avg_return=compute_overall_avg_return(xgboost_predictions),
        )
        results.append({
            "model_name": "XGBoost",
            "evaluation_name": "standard",
            "metrics": {
                "model_name": "XGBoost",
                "evaluation_name": "standard",
                "top20_avg_return": compute_top20_avg_return(xgboost_predictions),
                "overall_avg_return": compute_overall_avg_return(xgboost_predictions),
                **xgboost_metrics,
            },
            "predictions": xgboost_predictions.assign(model_name="XGBoost"),
            "portfolio_series": build_portfolio_time_series(xgboost_predictions, model_name="XGBoost", evaluation_name="standard"),
        })
    except ImportError as exc:
        print(f"Skipping XGBoost: {exc}")

    return results
