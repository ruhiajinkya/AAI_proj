import numpy as np
import pandas as pd


def rolling_cum_return(x, window):
    return (1 + x).rolling(window=window).apply(np.prod, raw=True) - 1


def downside_vol(series, window):
    neg = series.where(series < 0, 0.0)
    return neg.rolling(window=window).std()


def build_ml_dataset(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    # --------------------------
    # step1: basic cleaning
    required_cols = [
        "Ticker", "YearMonth", "Monthly_Return",
        "Monthly_Volatility", "Avg_Monthly_Volume", "Log_Volume",
        "MonthEnd_Market_Cap", "Log_Market_Cap"
    ]
    missing_cols = [c for c in required_cols if c not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    data["YearMonth"] = pd.to_datetime(data["YearMonth"].astype(str)) + pd.offsets.MonthEnd(0)
    data = data.sort_values(["Ticker", "YearMonth"]).reset_index(drop=True)

    # --------------------------
    # step2: momentum features
    data["ret_1m"] = data["Monthly_Return"]
    data["ret_3m"] = data.groupby("Ticker")["ret_1m"].transform(lambda x: rolling_cum_return(x, 3))
    data["ret_6m"] = data.groupby("Ticker")["ret_1m"].transform(lambda x: rolling_cum_return(x, 6))
    data["ret_12m"] = data.groupby("Ticker")["ret_1m"].transform(lambda x: rolling_cum_return(x, 12))


    data["momentum_accel"] = data.groupby("Ticker")["ret_1m"].diff()

    # --------------------------
    # step3: volatility features
    data["vol_1m"] = data["Monthly_Volatility"]
    data["vol_3m"] = data.groupby("Ticker")["ret_1m"].transform(lambda x: x.rolling(3).std())
    data["vol_6m"] = data.groupby("Ticker")["ret_1m"].transform(lambda x: x.rolling(6).std())
    data["downside_vol_3m"] = data.groupby("Ticker")["ret_1m"].transform(lambda x: downside_vol(x, 3))

    # --------------------------
    # step4: volume / liquidity features
    data["avg_volume_1m"] = data["Avg_Monthly_Volume"]
    data["avg_volume_3m"] = data.groupby("Ticker")["avg_volume_1m"].transform(lambda x: x.rolling(3).mean())
    data["volume_change_1m"] = data.groupby("Ticker")["avg_volume_1m"].pct_change()
    data["volume_ratio_1m_3m"] = data["avg_volume_1m"] / data["avg_volume_3m"]
    data["log_volume_change_1m"] = data.groupby("Ticker")["Log_Volume"].diff()

    # --------------------------
    # step5: market cap / size features
    data["market_cap_1m"] = data["MonthEnd_Market_Cap"]
    data["log_market_cap_1m"] = data["Log_Market_Cap"]
    data["market_cap_3m_avg"] = data.groupby("Ticker")["market_cap_1m"].transform(lambda x: x.rolling(3).mean())
    data["market_cap_change_1m"] = data.groupby("Ticker")["market_cap_1m"].pct_change()
    data["market_cap_ratio_1m_3m"] = data["market_cap_1m"] / data["market_cap_3m_avg"]
    data["log_market_cap_change_1m"] = data.groupby("Ticker")["log_market_cap_1m"].diff()

    # --------------------------
    # step6: cross-sectional rank features
    rank_cols = [
        "ret_1m", "ret_3m", "ret_6m", "ret_12m",
        "vol_1m", "vol_3m", "vol_6m", "downside_vol_3m",
        "avg_volume_1m", "volume_ratio_1m_3m", "Log_Volume",
        "market_cap_1m", "market_cap_ratio_1m_3m", "log_market_cap_1m"
    ]

    for col in rank_cols:
        data[f"{col}_rank"] = data.groupby("YearMonth")[col].rank(pct=True)

    data["momentum_volume_interaction"] = data["ret_3m_rank"] * data["vol_3m_rank"]

    # --------------------------
    # step7: future return and label
    data["future_return_1m"] = data.groupby("Ticker")["ret_1m"].shift(-1)
    data["future_return_rank"] = data.groupby("YearMonth")["future_return_1m"].rank(pct=True)
    data["label_top20"] = (data["future_return_rank"] >= 0.8).astype(int)

   
    data = data.replace([np.inf, -np.inf], np.nan)

    # --------------------------
    # keep rows after enough history exists
    feature_cols = [
        "ret_1m", "ret_3m", "ret_6m", "ret_12m",
        "momentum_accel",
        "vol_1m", "vol_3m", "vol_6m", "downside_vol_3m",
        "avg_volume_1m", "avg_volume_3m", "volume_change_1m",
        "volume_ratio_1m_3m", "Log_Volume", "log_volume_change_1m",
        "market_cap_1m", "log_market_cap_1m", "market_cap_3m_avg",
        "market_cap_change_1m", "market_cap_ratio_1m_3m", "log_market_cap_change_1m",
        "ret_1m_rank", "ret_3m_rank", "ret_6m_rank", "ret_12m_rank",
        "vol_1m_rank", "vol_3m_rank", "vol_6m_rank", "downside_vol_3m_rank",
        "avg_volume_1m_rank", "volume_ratio_1m_3m_rank", "Log_Volume_rank",
        "market_cap_1m_rank", "market_cap_ratio_1m_3m_rank", "log_market_cap_1m_rank",
        "momentum_volume_interaction",
    ]

    final_cols = ["Ticker", "YearMonth"] + feature_cols + ["future_return_1m", "label_top20"]
    ml_df = data[final_cols].dropna().reset_index(drop=True)

    return ml_df
