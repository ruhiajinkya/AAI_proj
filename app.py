import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

DEFAULT_METRICS = pd.DataFrame(
    [
        {
            "model": "Random Forest",
            "roc_auc": 0.617869,
            "accuracy": 0.791252,
            "f1": 0.012539,
            "pr_auc": 0.294212,
            "avg_precision_at_k": 0.320000,
            "avg_return": 0.032196,
        },
        {
            "model": "XGBoost",
            "roc_auc": 0.584429,
            "accuracy": 0.788602,
            "f1": 0.006231,
            "pr_auc": 0.269858,
            "avg_precision_at_k": 0.286667,
            "avg_return": 0.027348,
        },
        {
            "model": "Logistic Regression",
            "roc_auc": 0.578571,
            "accuracy": 0.791252,
            "f1": 0.018692,
            "pr_auc": 0.262897,
            "avg_precision_at_k": 0.263333,
            "avg_return": 0.026854,
        },
        {
            "model": "LightGBM",
            "roc_auc": 0.555395,
            "accuracy": 0.789264,
            "f1": 0.042169,
            "pr_auc": 0.251440,
            "avg_precision_at_k": 0.243333,
            "avg_return": 0.020809,
        },
    ]
)

EXPECTED_CORE = ["Ticker", "YearMonth", "label_top20", "future_return_1m"]

st.set_page_config(page_title="Stock Ranking ML App", layout="wide")


def load_notebook_summary(notebook_path: str = "MLproject.ipynb"):
    p = Path(notebook_path)
    if not p.exists():
        return None
    try:
        nb = json.loads(p.read_text(encoding="utf-8"))
        outputs = []
        for cell in nb.get("cells", []):
            for out in cell.get("outputs", []):
                if "text" in out:
                    outputs.append("".join(out["text"]))
        return "\n".join(outputs)[:4000]
    except Exception:
        return None


@st.cache_data

def load_csv(uploaded_file):
    return pd.read_csv(uploaded_file)


@st.cache_data

def prepare_dataframe(df: pd.DataFrame):
    df = df.copy()
    missing_core = [c for c in EXPECTED_CORE if c not in df.columns]
    if missing_core:
        raise ValueError(f"Missing required columns: {', '.join(missing_core)}")

    df["YearMonth"] = pd.to_datetime(df["YearMonth"])
    df = df.sort_values("YearMonth").reset_index(drop=True)

    drop_cols = ["Ticker", "YearMonth", "label_top20", "future_return_1m"]
    features = [col for col in df.columns if col not in drop_cols]
    if not features:
        raise ValueError("No feature columns found after dropping identifiers and targets.")

    feature_frame = df[features].copy()
    if feature_frame.isnull().any().any():
        feature_frame = feature_frame.fillna(feature_frame.median(numeric_only=True))
        feature_frame = feature_frame.fillna(0)

    if not all(np.issubdtype(dtype, np.number) for dtype in feature_frame.dtypes):
        bad_cols = [c for c in feature_frame.columns if not np.issubdtype(feature_frame[c].dtype, np.number)]
        raise ValueError(
            "All model feature columns must be numeric. Non-numeric columns found: " + ", ".join(bad_cols)
        )

    return df, feature_frame, features



def evaluate_ranking(df_subset: pd.DataFrame, probs, top_k: int = 20):
    df_eval = df_subset.copy()
    df_eval["prob"] = probs

    results = []
    for date, group in df_eval.groupby("YearMonth"):
        group = group.sort_values("prob", ascending=False)
        top_k_df = group.head(top_k)
        results.append(
            {
                "date": date,
                "precision_at_k": float(top_k_df["label_top20"].mean()),
                "avg_return": float(top_k_df["future_return_1m"].mean()),
            }
        )

    results_df = pd.DataFrame(results)
    return {
        "avg_precision_at_k": float(results_df["precision_at_k"].mean()),
        "avg_return": float(results_df["avg_return"].mean()),
        "monthly": results_df,
    }



def get_models():
    models = {
        "Logistic Regression": Pipeline(
            [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))]
        ),
        "Random Forest": GridSearchCV(
            RandomForestClassifier(random_state=42),
            {"n_estimators": [100, 200], "max_depth": [5, 10]},
            cv=TimeSeriesSplit(n_splits=3),
            scoring="roc_auc",
            n_jobs=-1,
        ),
    }

    if XGBClassifier is not None:
        models["XGBoost"] = GridSearchCV(
            XGBClassifier(eval_metric="logloss"),
            {"n_estimators": [100, 200], "max_depth": [3, 6], "learning_rate": [0.05, 0.1]},
            cv=TimeSeriesSplit(n_splits=3),
            scoring="roc_auc",
            n_jobs=-1,
        )

    if LGBMClassifier is not None:
        models["LightGBM"] = GridSearchCV(
            LGBMClassifier(verbose=-1),
            {"n_estimators": [100, 200], "num_leaves": [31, 64], "learning_rate": [0.05, 0.1]},
            cv=TimeSeriesSplit(n_splits=3),
            scoring="roc_auc",
            n_jobs=-1,
        )

    return models


@st.cache_resource(show_spinner=False)
def train_models(df: pd.DataFrame, X: pd.DataFrame, split_fraction: float, top_k: int):
    split_date = df["YearMonth"].quantile(split_fraction)
    train_mask = df["YearMonth"] < split_date
    test_mask = df["YearMonth"] >= split_date

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError("The selected split creates an empty train or test set. Adjust the split fraction.")

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = df.loc[train_mask, "label_top20"], df.loc[test_mask, "label_top20"]
    df_test = df.loc[test_mask].copy()

    results = []
    fitted_models = {}
    scored_test_frames = {}

    for model_name, model in get_models().items():
        fitted = model.fit(X_train, y_train)
        estimator = fitted.best_estimator_ if hasattr(fitted, "best_estimator_") else fitted
        probs = estimator.predict_proba(X_test)[:, 1]
        preds = estimator.predict(X_test)
        ranking = evaluate_ranking(df_test, probs, top_k=top_k)

        results.append(
            {
                "model": model_name,
                "roc_auc": float(roc_auc_score(y_test, probs)),
                "accuracy": float(accuracy_score(y_test, preds)),
                "f1": float(f1_score(y_test, preds, zero_division=0)),
                "pr_auc": float(average_precision_score(y_test, probs)),
                "avg_precision_at_k": ranking["avg_precision_at_k"],
                "avg_return": ranking["avg_return"],
            }
        )

        scored = df_test[["Ticker", "YearMonth", "future_return_1m", "label_top20"]].copy()
        scored["probability"] = probs
        scored = scored.sort_values(["YearMonth", "probability"], ascending=[True, False])

        fitted_models[model_name] = estimator
        scored_test_frames[model_name] = {
            "test_scores": scored,
            "monthly_ranking": ranking["monthly"],
        }

    results_df = pd.DataFrame(results).sort_values("avg_return", ascending=False).reset_index(drop=True)
    return {
        "results": results_df,
        "fitted_models": fitted_models,
        "scored_test_frames": scored_test_frames,
        "split_date": split_date,
        "train_size": int(train_mask.sum()),
        "test_size": int(test_mask.sum()),
    }


st.title("S&P 100 Monthly Stock Ranking App")
st.caption("Turns your notebook into an interactive app for model comparison and stock ranking.")

with st.sidebar:
    st.header("Data")
    uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])
    split_fraction = st.slider("Train/Test split percentile", 0.60, 0.95, 0.80, 0.05)
    top_k = st.slider("Top-K stocks per month", 5, 50, 20, 5)

    st.header("Required columns")
    st.write(", ".join(EXPECTED_CORE))

    if st.button("Use notebook baseline only"):
        st.session_state["baseline_only"] = True

baseline_only = st.session_state.get("baseline_only", False) and uploaded_file is None

if uploaded_file is None:
    st.info("Upload the CSV used by your notebook to enable training and ranking. Until then, the app can show the notebook's saved baseline results.")

    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.subheader("Baseline model results from the notebook")
        st.dataframe(DEFAULT_METRICS, use_container_width=True)
    with c2:
        st.subheader("What this app adds")
        st.markdown(
            """
            - upload a dataset and retrain all models
            - compare ROC-AUC, PR-AUC, F1, and ranking return
            - inspect monthly top-ranked stocks
            - present the project as a usable ML application, not just a notebook
            """
        )

    notebook_summary = load_notebook_summary("/mnt/data/MLproject.ipynb")
    if notebook_summary:
        with st.expander("Notebook output preview"):
            st.text(notebook_summary)
    st.stop()

try:
    raw_df = load_csv(uploaded_file)
    df, X, feature_names = prepare_dataframe(raw_df)
except Exception as e:
    st.error(str(e))
    st.stop()

st.success(f"Loaded {len(df):,} rows, {len(feature_names)} features, and {df['Ticker'].nunique()} tickers.")

with st.expander("Preview uploaded data"):
    st.dataframe(df.head(20), use_container_width=True)

with st.spinner("Training models and scoring the test period..."):
    try:
        artifacts = train_models(df, X, split_fraction, top_k)
    except Exception as e:
        st.error(str(e))
        st.stop()

results_df = artifacts["results"]
scored_frames = artifacts["scored_test_frames"]

m1, m2, m3 = st.columns(3)
m1.metric("Train rows", f"{artifacts['train_size']:,}")
m2.metric("Test rows", f"{artifacts['test_size']:,}")
m3.metric("Split date", str(pd.to_datetime(artifacts['split_date']).date()))

st.subheader("Model comparison")
st.dataframe(results_df, use_container_width=True)
st.bar_chart(results_df.set_index("model")[["avg_return", "avg_precision_at_k", "roc_auc"]])

best_model_name = results_df.iloc[0]["model"]
selected_model = st.selectbox("Inspect a model", results_df["model"].tolist(), index=0)
model_payload = scored_frames[selected_model]
scored_test = model_payload["test_scores"]
monthly_ranking = model_payload["monthly_ranking"]

left, right = st.columns([1.2, 1])
with left:
    st.subheader(f"Monthly Top-{top_k} picks: {selected_model}")
    available_dates = sorted(scored_test["YearMonth"].dropna().unique())
    selected_date = st.selectbox(
        "Choose a month",
        available_dates,
        format_func=lambda x: pd.to_datetime(x).strftime("%Y-%m"),
    )
    month_df = scored_test[scored_test["YearMonth"] == selected_date].head(top_k).copy()
    month_df["YearMonth"] = pd.to_datetime(month_df["YearMonth"]).dt.strftime("%Y-%m")
    st.dataframe(month_df, use_container_width=True)

with right:
    st.subheader("Monthly ranking performance")
    monthly_chart = monthly_ranking.copy()
    monthly_chart = monthly_chart.set_index("date")[["precision_at_k", "avg_return"]]
    st.line_chart(monthly_chart)

st.subheader("Best model takeaway")
row = results_df.iloc[0]
st.write(
    f"The strongest model on average test-period return is **{best_model_name}** with "
    f"an average top-{top_k} monthly return of **{row['avg_return']:.2%}** and "
    f"average precision@{top_k} of **{row['avg_precision_at_k']:.2%}**."
)

st.subheader("How to present this in your project demo")
st.markdown(
    """
    1. Explain the problem: ranking stocks likely to land in the next month's top 20 performers.
    2. Show the uploaded dataset and engineered features.
    3. Compare the four ML models on both classification and ranking metrics.
    4. Use the monthly top-picks table to show how the model translates into an actionable screen.
    5. Conclude with why the best model is the most practical choice.
    """
)
