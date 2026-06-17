from pathlib import Path
import numpy as np
import pandas as pd


def load_raw_quote(path):
    df = pd.read_csv(path, parse_dates=["datetime_utc"])
    df = df.sort_values("datetime_utc").reset_index(drop=True)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0].copy()
    return df


def add_log_returns(df, interval_minutes=5, drop_large_gaps=True):
    cleaned = df.copy()
    cleaned["log_close"] = np.log(cleaned["close"])
    cleaned["gap_minutes"] = (
        cleaned["datetime_utc"].diff().dt.total_seconds().div(60)
    )
    cleaned["log_return"] = cleaned["log_close"].diff()

    if drop_large_gaps:
        max_allowed_gap = interval_minutes * 1.5
        cleaned.loc[cleaned["gap_minutes"] > max_allowed_gap, "log_return"] = np.nan

    return cleaned.dropna(subset=["log_return"]).reset_index(drop=True)


def add_intraday_deseasonalized_returns(df):
    cleaned = df.copy()
    cleaned["time_of_day"] = cleaned["datetime_utc"].dt.strftime("%H:%M")
    slot_mean = cleaned.groupby("time_of_day")["log_return"].transform("mean")
    slot_std = cleaned.groupby("time_of_day")["log_return"].transform("std")
    fallback = cleaned["log_return"].std(ddof=1)
    slot_std = slot_std.replace(0, np.nan).fillna(fallback)
    cleaned["deseasonalized_return"] = (cleaned["log_return"] - slot_mean) / slot_std
    return cleaned


def build_wide_panel(frames, value_column):
    pieces = []
    for frame in frames:
        ticker = frame["ticker"].iloc[0]
        piece = frame[["datetime_utc", value_column]].rename(columns={value_column: ticker})
        pieces.append(piece.set_index("datetime_utc"))
    return pd.concat(pieces, axis=1).sort_index()


def summarize_returns(panel):
    rows = []
    for ticker in panel.columns:
        series = panel[ticker].dropna()
        rows.append(
            {
                "ticker": ticker,
                "observations": int(series.shape[0]),
                "mean": float(series.mean()),
                "std": float(series.std(ddof=1)),
                "skew": float(series.skew()),
                "excess_kurtosis": float(series.kurt()),
                "min": float(series.min()),
                "max": float(series.max()),
            }
        )
    return pd.DataFrame(rows)
