# Stock Ranking ML App

This app converts the notebook in `MLproject.ipynb` into a usable Streamlit application.

## What it does
- Uploads the stock dataset as a CSV
- Recreates the notebook's time-based train/test split
- Trains Logistic Regression, Random Forest, XGBoost, and LightGBM when available
- Compares ROC-AUC, PR-AUC, F1, average precision@K, and average return of the top-K ranked stocks
- Lets you inspect the top-ranked stocks for each month in the test period

## Files
- `app.py` — Streamlit app
- `MLproject.ipynb` — original notebook

## How to run
```bash
pip install streamlit pandas numpy scikit-learn xgboost lightgbm
streamlit run app.py
```

## Required dataset columns
Your CSV should contain at least these columns:
- `Ticker`
- `YearMonth`
- `label_top20`
- `future_return_1m`

All remaining columns are treated as numeric model features.

## Notes
The original notebook references `sample_data/sp100_monthly_ml_dataset.csv`, but that CSV was not included with the upload. The app therefore uses a file uploader so you can plug in the dataset directly.
