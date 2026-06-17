from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def plot_price_panel(log_prices, output_path):
    prices = log_prices.apply(lambda column: column - column.dropna().iloc[0])
    ax = prices.plot(figsize=(11, 6), linewidth=1)
    ax.set_title("Normalized log-prices")
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Log-price change from first observation")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_return_histograms(returns, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.ravel()
    for ax, ticker in zip(axes, returns.columns):
        series = returns[ticker].dropna()
        ax.hist(series, bins=80, density=True, alpha=0.75)
        ax.set_title(ticker)
        ax.grid(alpha=0.25)
    fig.suptitle("5-minute log-return distributions")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_abs_return_autocorr(returns, output_path, max_lag=160):
    fig, ax = plt.subplots(figsize=(11, 6))
    for ticker in returns.columns:
        series = returns[ticker].dropna().abs()
        acf = [series.autocorr(lag=lag) for lag in range(1, max_lag + 1)]
        ax.plot(range(1, max_lag + 1), acf, label=ticker, linewidth=1)
    ax.set_title("Autocorrelation of absolute returns")
    ax.set_xlabel("Lag in 5-minute bars")
    ax.set_ylabel("Autocorrelation")
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main():
    processed_dir = PROJECT_ROOT / "data" / "processed"
    figures_dir = PROJECT_ROOT / "outputs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    returns = pd.read_csv(processed_dir / "log_returns_panel.csv", index_col=0, parse_dates=True)
    log_prices = pd.read_csv(processed_dir / "log_prices_panel.csv", index_col=0, parse_dates=True)

    plot_price_panel(log_prices, figures_dir / "normalized_log_prices.png")
    plot_return_histograms(returns, figures_dir / "return_histograms.png")
    plot_abs_return_autocorr(returns, figures_dir / "absolute_return_autocorrelation.png")
    print(f"saved figures to {figures_dir}")


if __name__ == "__main__":
    main()
