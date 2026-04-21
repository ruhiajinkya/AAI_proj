# Stock Ranking Project

This project builds a monthly stock-selection pipeline for S&P 100 equities, evaluates several machine learning models, and exposes the results through a Streamlit demo application.

The core idea is:

1. Collect S&P 100 daily market data.
2. Convert it into monthly features.
3. Build an ML dataset for a top-20% stock-selection task.
4. Train and evaluate classification models.
5. Convert model scores into monthly portfolio returns.
6. Export metrics, stock recommendations, and plots.
7. Present the walk-forward results in a Streamlit app.

## Project Goals

- Predict which stocks are likely to be in the top 20% by next-month return.
- Rank stocks each month by predicted probability.
- Form a portfolio from the top-ranked names.
- Evaluate both classification quality and portfolio performance.
- Compare walk-forward portfolio performance with SPY when benchmark data is available.

## Main Workflow

The pipeline is orchestrated by [main.py]

### 1. Data Collection

[data_loader.py]
- `fetch_sp100_tickers()`
  - Scrapes the S&P 100 member list from Wikipedia.
- `download_daily_data()`
  - Downloads daily OHLCV data with `yfinance`.
- `reshape_to_long_format()`
  - Converts Yahoo Finance output into a long table with one row per ticker-date pair.

### 2. Monthly Aggregation

[preprocess.py]

- `build_monthly_features_from_daily()`
  - Converts daily data into monthly return, monthly volatility, and average monthly volume.

### 3. Market Cap Enrichment

[market_cap.py]
- `build_monthly_market_cap_from_daily()`
  - Uses month-end price and `sharesOutstanding` from Yahoo Finance to estimate market cap.
- `add_market_cap_to_monthly()`
  - Merges the market cap data into the monthly table.
- `clean_and_prepare()`
  - Cleans invalid values and adds `Log_Market_Cap`.

### 4. ML Feature Engineering

[feature_engineering.py]

`build_ml_dataset()` creates the modeling dataset. The engineered features include:

- Momentum features:
  - `ret_1m`, `ret_3m`, `ret_6m`, `ret_12m`
  - `momentum_accel`
- Volatility features:
  - `vol_1m`, `vol_3m`, `vol_6m`
  - `downside_vol_3m`
- Volume and liquidity features:
  - `avg_volume_1m`, `avg_volume_3m`
  - `volume_change_1m`, `volume_ratio_1m_3m`
  - `log_volume_change_1m`
- Size features:
  - `market_cap_1m`, `market_cap_3m_avg`
  - `market_cap_change_1m`, `market_cap_ratio_1m_3m`
  - `log_market_cap_1m`, `log_market_cap_change_1m`
- Cross-sectional rank features:
  - monthly percentile ranks of return, volatility, volume, and market-cap features
- Interaction feature:
  - `momentum_volume_interaction = ret_3m_rank * vol_3m_rank`

The target variables are:

- `future_return_1m`
- `label_top20`
  - equals `1` if the stock is in the top 20% of next-month returns within that month

## Modeling

[model_module.py]

This module defines the standard models and the common model utilities.

### Models

- Logistic Regression
- Random Forest
- XGBoost

### Feature Sets

- `BASE_FEATURE_COLUMNS`
  - used for Logistic Regression
- `RANDOM_FOREST_FEATURE_COLUMNS`
  - extends the base set with additional nonlinear-friendly features
- `XGBOOST_FEATURE_COLUMNS`
  - currently matches the Random Forest feature set

### Standard Evaluation

`run_standard_models()` runs:

- Logistic Regression with `train_test_split`
- Random Forest with a time split
- XGBoost with a time split

For each model, the pipeline computes:

- accuracy
- ROC AUC
- classification report
- average return of the top-20% ranked stocks
- overall average return
- portfolio metrics derived from the ranked stock picks

## Walk-Forward Evaluation

[walk_forward_model.py]

This is the main evaluation mode used by the demo application.

### Why Walk-Forward

Walk-forward validation better reflects a real investment workflow:

- train on past months
- test on the next month
- roll the window forward
- repeat until the sample is exhausted

### Key Functions

- `prepare_walk_forward_data()`
  - filters rows with missing required features
- `build_walk_forward_slices()`
  - creates rolling train/test windows
- `choose_threshold_from_training_predictions()`
  - chooses a classification threshold
- `run_walk_forward_validation()`
  - trains the model fold by fold and collects predictions and metrics

### Threshold Strategy

The default threshold strategy is `target_rate`, which tries to match the positive rate in training data rather than always using `0.5`.

Other supported strategies:

- `balanced_accuracy`
- `f1`

## Portfolio Construction and Metrics

[metrics_module.py]

This module converts ranked predictions into portfolio returns.

### Logic

- Rank each stock within each month by model score.
- Select stocks with `pred_rank >= 0.8`.
- Average their realized next-month return.
- Compound monthly returns into cumulative return.

### Reported Portfolio Metrics

- `portfolio_months`
- `avg_monthly_return`
- `monthly_return_std`
- `monthly_win_rate`
- `cumulative_return`
- `annualized_sharpe`
- `best_month_return`
- `worst_month_return`
- `avg_monthly_picks`

## Benchmark Comparison

[benchmark_module.py]

This module builds a monthly SPY benchmark series and aligns it with each model portfolio.

### Functions

- `fetch_benchmark_monthly_returns()`
  - downloads monthly SPY returns
- `build_portfolio_benchmark_comparison()`
  - aligns portfolio returns and benchmark returns by month
- `summarize_portfolio_benchmark_comparison()`
  - creates summary statistics such as excess cumulative return

The current export logic applies the portfolio-vs-SPY comparison to walk-forward portfolios only.

## Plotting

[plot_module.py]

This module generates:

- cumulative portfolio return plots
- portfolio vs SPY comparison plots
- model comparison bar charts

## Demo Application

The end-user UI is implemented with Streamlit.

### Data Layer

[demo_data_module.py]
This module:

- loads exported artifact CSV files
- filters them to `walk_forward`
- provides helper functions for:
  - available models
  - available target months
  - top-ranked stock recommendations
  - portfolio history
  - benchmark history
  - metrics lookup

### Streamlit UI

[streamlit_app.py]

The app allows an end user to:

- choose a model
- choose a target month
- view recommended top-rank stocks
- inspect historical cumulative portfolio performance
- inspect walk-forward evaluation metrics
- inspect SPY benchmark comparison when available

## Generated Outputs

The project writes outputs into `artifacts/`.

### Metrics

Stored in `artifacts/metrics/`:

- `model_metrics_summary.csv`
  - combined standard and walk-forward model metrics
- `walk_forward_metrics_summary.csv`
  - walk-forward metrics only
- `selected_stock.csv`
  - monthly ranked stock recommendations
- `portfolio_time_series.csv`
  - monthly portfolio return path
- `portfolio_vs_spy.csv`
  - walk-forward portfolio vs SPY monthly comparison
- `portfolio_vs_spy_summary.csv`
  - aggregated portfolio vs SPY comparison

### Plots

Stored in `artifacts/plots/`:

- `cumulative_returns.png`
- `portfolio_vs_spy.png`
- `model_comparison.png`

## Configuration

[config.py]

This file defines:

- data directories
- artifact directories
- date range
- default file paths

It also contains a fallback so that if the processed ML dataset does not exist in `data/processed/`, the code can reuse the local legacy file:

- `sp100_monthly_ml_dataset.csv`

## How To Run

### 1. Run the pipeline

From the `src` directory:

```powershell
python main.py
```

To force a fresh rebuild:

```powershell
python main.py --rebuild-data
```

### 2. Launch the demo app

```powershell
streamlit run streamlit_app.py
```

## Expected Dependencies

Based on the imports used in the codebase, the project depends on:

- `pandas`
- `numpy`
- `requests`
- `yfinance`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `streamlit`
- `xgboost` optional

## Notes About Current Behavior

- The demo is driven by exported artifact CSV files, not by live model training inside Streamlit.
- Walk-forward results are the primary user-facing evaluation mode.
- The cumulative return comparison plot is currently limited to walk-forward models.
- If benchmark download fails, the pipeline skips SPY comparison and still completes.
- If the local ML CSV is missing `momentum_volume_interaction`, `main.py` backfills it in memory.

## Suggested Reading Order

If you want to understand the project quickly, read the code in this order:

1. [main.py]
2. [feature_engineering.py]
3. [model_module.py]
4. [walk_forward_model.py]
5. [metrics_module.py]
6. [streamlit_app.py]