import sys
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import yfinance as yf


TICKERS = ["SPY", "QQQ", "AAPL", "MSFT"]
PERIOD = "60d"
INTERVAL = "5m"


def main():
    output_dir = PROJECT_ROOT / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    for ticker in TICKERS:
        print(f"downloading {ticker}...")
        data = yf.download(ticker, period=PERIOD, interval=INTERVAL,
                           auto_adjust=False, prepost=False,
                           progress=False, threads=False)

        if data.empty:
            print(f"  no data for {ticker}, skipped")
            continue

        # Recent yfinance versions return a two-level column index even for one ticker.
        if getattr(data.columns, "nlevels", 1) > 1:
            data.columns = data.columns.droplevel(-1)

        data = data.reset_index()
        dt_col = "Datetime" if "Datetime" in data.columns else "Date"
        data = data.rename(columns={
            dt_col: "datetime_utc",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })

        data["ticker"] = ticker
        data["currency"] = ""
        data["exchange_timezone"] = ""
        data["datetime_utc"] = data["datetime_utc"].apply(to_utc)

        cols = ["ticker", "datetime_utc", "open", "high", "low", "close",
                "volume", "currency", "exchange_timezone"]
        out = output_dir / f"{ticker.replace('-', '_')}_{INTERVAL}_{PERIOD}.csv"
        data[cols].to_csv(out, index=False)
        print(f"  saved {len(data)} rows to {out}")
        time.sleep(0.7)


def to_utc(value):
    if getattr(value, "tzinfo", None) is None:
        return value.tz_localize("UTC")
    return value.tz_convert("UTC")


if __name__ == "__main__":
    main()
