import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (
    add_intraday_deseasonalized_returns,
    add_log_returns,
    build_wide_panel,
    load_raw_quote,
    summarize_returns,
)


INTERVAL_MINUTES = 5


def main():
    raw_dir = PROJECT_ROOT / "data" / "raw"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    table_dir = PROJECT_ROOT / "outputs" / "tables"
    processed_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for path in sorted(raw_dir.glob("*.csv")):
        raw = load_raw_quote(path)
        returns = add_log_returns(raw, interval_minutes=INTERVAL_MINUTES)
        returns = add_intraday_deseasonalized_returns(returns)
        output_path = processed_dir / path.name.replace(".csv", "_returns.csv")
        returns.to_csv(output_path, index=False)
        frames.append(returns)
        print(f"prepared {path.name}: {len(returns)} returns")

    if len(frames) == 0:
        raise RuntimeError("No raw CSV files found. Run scripts/01_download_quotes.py first.")

    returns_panel = build_wide_panel(frames, "log_return")
    deseasonalized_panel = build_wide_panel(frames, "deseasonalized_return")
    log_price_panel = build_wide_panel(frames, "log_close")

    returns_panel.to_csv(processed_dir / "log_returns_panel.csv")
    deseasonalized_panel.to_csv(processed_dir / "deseasonalized_returns_panel.csv")
    log_price_panel.to_csv(processed_dir / "log_prices_panel.csv")
    summarize_returns(returns_panel).to_csv(table_dir / "return_summary.csv", index=False)
    summarize_returns(deseasonalized_panel).to_csv(
        table_dir / "deseasonalized_return_summary.csv", index=False
    )
    print(f"saved panel: {processed_dir / 'log_returns_panel.csv'}")
    print(f"saved deseasonalized panel: {processed_dir / 'deseasonalized_returns_panel.csv'}")
    print(f"saved summary: {table_dir / 'return_summary.csv'}")


if __name__ == "__main__":
    main()
