from pathlib import Path

import pandas as pd
import streamlit as st

from demo_data_module import (
    available_models,
    available_months_for_model,
    benchmark_history_for_model,
    benchmark_summary_row,
    load_portfolio_time_series,
    load_portfolio_vs_spy,
    load_portfolio_vs_spy_summary,
    load_selected_stocks,
    load_walk_forward_metrics,
    model_metrics_row,
    portfolio_history_for_model,
    recommended_stocks_for_month,
    validate_demo_artifacts,
)


st.set_page_config(
    page_title="Stock Ranking Demo",
    page_icon=":material/finance:",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_demo_data():
    selected_df = load_selected_stocks()
    metrics_df = load_walk_forward_metrics()
    portfolio_df = load_portfolio_time_series()
    benchmark_df = load_portfolio_vs_spy()
    benchmark_summary_df = load_portfolio_vs_spy_summary()
    return selected_df, metrics_df, portfolio_df, benchmark_df, benchmark_summary_df


def format_pct(value):
    if pd.isna(value):
        return "N/A"
    return f"{value:.2%}"


def format_float(value, digits: int = 3):
    if pd.isna(value):
        return "N/A"
    return f"{value:.{digits}f}"


def build_performance_chart_data(portfolio_history: pd.DataFrame, benchmark_history: pd.DataFrame) -> pd.DataFrame:
    chart_df = portfolio_history[["YearMonth", "cumulative_return"]].rename(
        columns={"cumulative_return": "Portfolio"}
    )
    chart_df = chart_df.set_index("YearMonth")

    if not benchmark_history.empty and "benchmark_cumulative_return" in benchmark_history.columns:
        benchmark_label = benchmark_history["benchmark_symbol"].dropna().iloc[0] if benchmark_history["benchmark_symbol"].notna().any() else "Benchmark"
        chart_df[benchmark_label] = benchmark_history.set_index("YearMonth")["benchmark_cumulative_return"]

    return chart_df.sort_index()


def main():
    missing_artifacts = validate_demo_artifacts()
    if missing_artifacts:
        st.error("Required demo artifacts are missing. Run `python main.py` first to generate them.")
        st.code("\n".join(missing_artifacts))
        st.stop()

    selected_df, metrics_df, portfolio_df, benchmark_df, benchmark_summary_df = load_demo_data()
    models = available_models(selected_df)
    if not models:
        st.error("No walk-forward recommendation data found in `selected_stock.csv`.")
        st.stop()

    st.markdown(
        """
        <style>
        .app-shell {
            background: linear-gradient(135deg, #f4efe4 0%, #f9fbff 45%, #e8f2ea 100%);
            padding: 1.2rem 1.4rem;
            border-radius: 18px;
            border: 1px solid rgba(32, 62, 54, 0.10);
            margin-bottom: 1.2rem;
        }
        .eyebrow {
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #49685b;
            font-size: 0.80rem;
            margin-bottom: 0.35rem;
        }
        .hero-title {
            color: #153128;
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        .hero-copy {
            color: #304a41;
            font-size: 1rem;
            max-width: 58rem;
        }
        </style>
        <div class="app-shell">
            <div class="eyebrow">Walk-Forward Stock Selection Demo</div>
            <div class="hero-title">Model-driven monthly stock recommendations</div>
            <div class="hero-copy">
                Select a walk-forward model and target month to inspect the top-ranked stock picks,
                historical portfolio behavior, and evaluation metrics produced by the offline pipeline.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sidebar = st.sidebar
    sidebar.header("Controls")
    model_name = sidebar.selectbox("Model", models)
    months = available_months_for_model(selected_df, model_name)
    default_month_index = len(months) - 1 if months else 0
    target_month = sidebar.selectbox(
        "Target Month",
        months,
        index=default_month_index,
        format_func=lambda value: pd.Timestamp(value).strftime("%Y-%m"),
    )

    recommendations = recommended_stocks_for_month(selected_df, model_name, target_month)
    metrics_row = model_metrics_row(metrics_df, model_name)
    portfolio_history = portfolio_history_for_model(portfolio_df, model_name)
    benchmark_history = benchmark_history_for_model(benchmark_df, model_name)
    benchmark_summary = benchmark_summary_row(benchmark_summary_df, model_name)

    left_col, right_col = st.columns([1.5, 1], gap="large")

    with left_col:
        st.subheader("Recommended Top-Rank Stocks")
        st.caption(f"Model: {model_name} | Target month: {target_month.strftime('%Y-%m')}")
        if recommendations.empty:
            st.warning("No top-rank recommendations found for the selected month.")
        else:
            display_df = recommendations[["Ticker", "score", "rank"]].copy()
            display_df["score"] = display_df["score"].map(lambda x: f"{x:.4f}")
            display_df["rank"] = display_df["rank"].map(lambda x: f"{x:.2%}")
            st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.subheader("Historical Portfolio Performance")
        chart_df = build_performance_chart_data(portfolio_history, benchmark_history)
        if chart_df.empty:
            st.info("No portfolio history is available for the selected model.")
        else:
            st.line_chart(chart_df, use_container_width=True, height=360)

        if not portfolio_history.empty:
            monthly_view = portfolio_history[["YearMonth", "monthly_return", "picks"]].copy()
            monthly_view["YearMonth"] = monthly_view["YearMonth"].dt.strftime("%Y-%m")
            monthly_view["monthly_return"] = monthly_view["monthly_return"].map(format_pct)
            st.subheader("Monthly Portfolio History")
            st.dataframe(monthly_view, use_container_width=True, hide_index=True)

    with right_col:
        st.subheader("Evaluation Metrics")
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("ROC AUC", format_float(metrics_row.get("roc_auc"), 3))
        metric_col2.metric("F1 Score", format_float(metrics_row.get("f1"), 3))

        metric_col3, metric_col4 = st.columns(2)
        metric_col3.metric("Cumulative Return", format_pct(metrics_row.get("cumulative_return")))
        metric_col4.metric("Sharpe Ratio", format_float(metrics_row.get("annualized_sharpe"), 2))

        metric_col5, metric_col6 = st.columns(2)
        metric_col5.metric("Avg Monthly Return", format_pct(metrics_row.get("avg_monthly_return")))
        metric_col6.metric("Win Rate", format_pct(metrics_row.get("monthly_win_rate")))

        summary_table = pd.DataFrame(
            [
                ("Accuracy", format_pct(metrics_row.get("accuracy"))),
                ("Balanced Accuracy", format_pct(metrics_row.get("balanced_accuracy"))),
                ("Precision", format_pct(metrics_row.get("precision"))),
                ("Recall", format_pct(metrics_row.get("recall"))),
                ("Top20 Avg Return", format_pct(metrics_row.get("top20_avg_return"))),
                ("Best Month", format_pct(metrics_row.get("best_month_return"))),
                ("Worst Month", format_pct(metrics_row.get("worst_month_return"))),
                ("Average Picks / Month", format_float(metrics_row.get("avg_monthly_picks"), 1)),
            ],
            columns=["Metric", "Value"],
        )
        st.dataframe(summary_table, use_container_width=True, hide_index=True)

        st.subheader("Classification Report")
        st.code(metrics_row.get("classification_report", "N/A"))

        if benchmark_summary is not None:
            st.subheader("Benchmark Comparison")
            benchmark_label = benchmark_summary.get("benchmark_symbol", "Benchmark")
            benchmark_table = pd.DataFrame(
                [
                    (f"{benchmark_label} cumulative return", format_pct(benchmark_summary.get("benchmark_cumulative_return"))),
                    ("Portfolio excess return", format_pct(benchmark_summary.get("excess_cumulative_return"))),
                    (f"Average monthly {benchmark_label} return", format_pct(benchmark_summary.get("avg_monthly_benchmark_return"))),
                    ("Average monthly excess return", format_pct(benchmark_summary.get("avg_monthly_excess_return"))),
                ],
                columns=["Metric", "Value"],
            )
            st.dataframe(benchmark_table, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
