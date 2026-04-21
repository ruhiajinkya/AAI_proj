from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")


def plot_cumulative_returns(portfolio_series: pd.DataFrame, output_path: str | Path, title: str):
    output_path = Path(output_path)
    data = portfolio_series.copy()
    data["YearMonth"] = pd.to_datetime(data["YearMonth"])

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.lineplot(
        data=data,
        x="YearMonth",
        y="cumulative_return",
        hue="model_name",
        style="evaluation_name",
        linewidth=2.2,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel("Cumulative Return")
    ax.axhline(0, color="black", linewidth=1, alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_portfolio_vs_benchmark(
    comparison_df: pd.DataFrame,
    output_path: str | Path,
    title: str,
):
    output_path = Path(output_path)
    data = comparison_df.copy()
    if data.empty:
        return

    data["YearMonth"] = pd.to_datetime(data["YearMonth"])
    data["run_label"] = data["model_name"] + " (" + data["evaluation_name"] + ")"

    portfolio_lines = data[["YearMonth", "run_label", "cumulative_return"]].rename(
        columns={"cumulative_return": "value"}
    )
    portfolio_lines["series_type"] = "Portfolio"

    benchmark_lines = data[["YearMonth", "run_label", "benchmark_cumulative_return"]].rename(
        columns={"benchmark_cumulative_return": "value"}
    )
    benchmark_lines["series_type"] = data["benchmark_symbol"].iloc[0]

    plot_df = pd.concat([portfolio_lines, benchmark_lines], ignore_index=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.lineplot(
        data=plot_df,
        x="YearMonth",
        y="value",
        hue="run_label",
        style="series_type",
        linewidth=2.2,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel("Cumulative Return")
    ax.axhline(0, color="black", linewidth=1, alpha=0.5)
    ax.legend(title="Series", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(
    metrics_df: pd.DataFrame,
    output_path: str | Path,
    title: str,
    value_columns,
):
    output_path = Path(output_path)
    plot_df = metrics_df[["model_name"] + list(value_columns)].copy()
    plot_df["label"] = plot_df["model_name"]
    long_df = plot_df.melt(id_vars="label", value_vars=list(value_columns), var_name="metric", value_name="value")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=long_df, x="metric", y="value", hue="label", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Run", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
