import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
cache_dir = PROJECT_ROOT / ".matplotlib_cache"
cache_dir.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(PROJECT_ROOT))

from src.dfa import dfa, mfdfa
from src.fbm import fractional_gaussian_noise


Q = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4], dtype=float)
HURSTS = [0.3, 0.5, 0.7]
MC_RUNS = 50
CONTROL_RUNS = 30
RNG_SEED = 20260617


def main():
    processed_dir = PROJECT_ROOT / "data" / "processed"
    figures_dir = PROJECT_ROOT / "outputs" / "figures" / "report"
    tables_dir = PROJECT_ROOT / "outputs" / "tables" / "report"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    returns = pd.read_csv(processed_dir / "deseasonalized_returns_panel.csv",
                          index_col=0, parse_dates=True)
    raw_returns = pd.read_csv(processed_dir / "log_returns_panel.csv",
                              index_col=0, parse_dates=True)
    sample_info(raw_returns, tables_dir)

    dfa_results = {}
    mfdfa_results = {}
    summary_rows = []

    for ticker in returns.columns:
        x = clean_array(returns[ticker])
        dfa_results[ticker] = dfa(x)
        mfdfa_results[ticker] = mfdfa(x, Q)
        summary_rows.append(summary_row(ticker, "real_deseasonalized",
                                        len(x), dfa_results[ticker], mfdfa_results[ticker]))

    n = int(returns.count().median())
    benchmark_results, benchmark_summary = monte_carlo_benchmarks(n)
    controls = control_experiments(returns)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(tables_dir / "report_summary.csv", index=False)
    pd.DataFrame(benchmark_summary).to_csv(tables_dir / "benchmark_mc_summary.csv", index=False)
    pd.DataFrame(controls).to_csv(tables_dir / "control_summary.csv", index=False)
    make_long_tables(mfdfa_results, benchmark_results, tables_dir)

    plot_dfa_scaling(dfa_results, figures_dir / "dfa_scaling_real_assets.png")
    plot_hq(mfdfa_results, figures_dir / "mfdfa_hq_real_assets.png")
    plot_tau(mfdfa_results, figures_dir / "mfdfa_tau_real_assets.png")
    plot_spectra(mfdfa_results, figures_dir / "mfdfa_spectra_real_assets.png")
    plot_spectra({**mfdfa_results, **benchmark_results},
                 figures_dir / "mfdfa_spectra_with_benchmarks.png")
    plot_widths(summary, benchmark_summary, figures_dir / "mfdfa_spectrum_widths.png")
    plot_control_widths(summary, controls, figures_dir / "mfdfa_control_widths.png")

    write_notes(summary, benchmark_summary, controls, tables_dir / "report_notes.md")
    print(f"saved report figures to {figures_dir}")
    print(f"saved report tables to {tables_dir}")


def clean_array(series):
    x = series.dropna().to_numpy(dtype=float)
    return x[np.isfinite(x)]


def sample_info(raw_returns, tables_dir):
    rows = []
    for ticker in raw_returns.columns:
        s = raw_returns[ticker].dropna()
        rows.append({
            "ticker": ticker,
            "first_timestamp_utc": s.index.min(),
            "last_timestamp_utc": s.index.max(),
            "observations": len(s),
        })
    pd.DataFrame(rows).to_csv(tables_dir / "sample_period.csv", index=False)


def summary_row(name, group, n, dfa_result, mfdfa_result):
    return {
        "series": name,
        "group": group,
        "observations": n,
        "dfa_alpha": dfa_result["alpha"],
        "dfa_r2": dfa_result["r2"],
        "h_minus4": mfdfa_result["hq"][0],
        "h_0": mfdfa_result["hq"][4],
        "h_4": mfdfa_result["hq"][-1],
        "alpha_min": np.min(mfdfa_result["alpha"]),
        "alpha_max": np.max(mfdfa_result["alpha"]),
        "spectrum_width": mfdfa_result["spectrum_width"],
    }


def monte_carlo_benchmarks(n):
    rng = np.random.default_rng(RNG_SEED)
    plotted = {}
    rows = []
    for h in HURSTS:
        dfa_alpha = []
        width = []
        hq = []
        tau = []
        alpha = []
        f_alpha = []
        for _ in range(MC_RUNS):
            seed = int(rng.integers(0, 2**31 - 1))
            x = fractional_gaussian_noise(n, hurst=h, seed=seed)
            dfa_result = dfa(x)
            mfdfa_result = mfdfa(x, Q)
            dfa_alpha.append(dfa_result["alpha"])
            width.append(mfdfa_result["spectrum_width"])
            hq.append(mfdfa_result["hq"])
            tau.append(mfdfa_result["tau"])
            alpha.append(mfdfa_result["alpha"])
            f_alpha.append(mfdfa_result["f_alpha"])

        label = f"fGn H={h:.1f} MC mean"
        plotted[label] = {
            "q": Q,
            "hq": np.mean(hq, axis=0),
            "tau": np.mean(tau, axis=0),
            "alpha": np.mean(alpha, axis=0),
            "f_alpha": np.mean(f_alpha, axis=0),
            "spectrum_width": float(np.mean(width)),
        }
        rows.append({
            "series": f"fGn H={h:.1f}",
            "runs": MC_RUNS,
            "dfa_alpha_mean": np.mean(dfa_alpha),
            "dfa_alpha_std": np.std(dfa_alpha, ddof=1),
            "spectrum_width_mean": np.mean(width),
            "spectrum_width_std": np.std(width, ddof=1),
        })
    return plotted, rows


def control_experiments(returns):
    rng = np.random.default_rng(RNG_SEED + 1)
    rows = []
    for ticker in returns.columns:
        x = clean_array(returns[ticker])
        original_width = mfdfa(x, Q)["spectrum_width"]
        shuffled = []
        phase_randomized = []
        for _ in range(CONTROL_RUNS):
            shuffled.append(mfdfa(rng.permutation(x), Q)["spectrum_width"])
            phase_randomized.append(mfdfa(phase_randomize(x, rng), Q)["spectrum_width"])
        shuffle_mean = np.mean(shuffled)
        shuffle_std = np.std(shuffled, ddof=1)
        shuffle_z = (original_width - shuffle_mean) / shuffle_std if shuffle_std > 0 else np.nan
        rows.append({
            "series": ticker,
            "runs": CONTROL_RUNS,
            "real_width": original_width,
            "shuffle_width_mean": shuffle_mean,
            "shuffle_width_std": shuffle_std,
            "shuffle_z": shuffle_z,
            "phase_width_mean": np.mean(phase_randomized),
            "phase_width_std": np.std(phase_randomized, ddof=1),
        })
    return rows


def phase_randomize(x, rng):
    centered = x - np.mean(x)
    spectrum = np.fft.rfft(centered)
    phases = rng.uniform(0, 2 * np.pi, len(spectrum))
    phases[0] = 0.0
    if len(x) % 2 == 0:
        phases[-1] = 0.0
    new_spectrum = np.abs(spectrum) * np.exp(1j * phases)
    y = np.fft.irfft(new_spectrum, n=len(x))
    y = y - np.mean(y)
    if np.std(y, ddof=1) > 0:
        y = y / np.std(y, ddof=1) * np.std(x, ddof=1)
    return y


def make_long_tables(real_results, benchmark_results, tables_dir):
    hq_rows = []
    spectrum_rows = []
    for name, result in {**real_results, **benchmark_results}.items():
        group = "model_mc_mean" if name.startswith("fGn") else "real_deseasonalized"
        for q, hq, tau in zip(result["q"], result["hq"], result["tau"]):
            hq_rows.append({"series": name, "group": group, "q": q, "h_q": hq, "tau_q": tau})
        for alpha, f_alpha in zip(result["alpha"], result["f_alpha"]):
            spectrum_rows.append({"series": name, "group": group, "alpha": alpha, "f_alpha": f_alpha})

    pd.DataFrame(hq_rows).to_csv(tables_dir / "hq_tau_long.csv", index=False)
    pd.DataFrame(spectrum_rows).to_csv(tables_dir / "singularity_spectrum_long.csv", index=False)


def plot_dfa_scaling(dfa_results, path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (ticker, result) in zip(axes.ravel(), dfa_results.items()):
        s = result["scales"]
        f = result["fluctuations"]
        fit = np.exp(result["intercept"]) * s ** result["alpha"]
        ax.loglog(s, f, "o", markersize=4, label="DFA fluctuation")
        ax.loglog(s, fit, "-", label=f"slope={result['alpha']:.3f}, R2={result['r2']:.3f}")
        ax.set_title(ticker)
        ax.set_xlabel("scale s")
        ax.set_ylabel("F(s)")
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=8)
    fig.suptitle("DFA scaling for deseasonalized 5-minute returns")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_hq(results, path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for ticker, result in results.items():
        ax.plot(result["q"], result["hq"], marker="o", linewidth=1.6, label=ticker)
    ax.axhline(0.5, color="black", linewidth=1, linestyle=":", label="0.5 reference")
    ax.set_title("Generalized Hurst exponents after intraday deseasonalization")
    ax.set_xlabel("q")
    ax.set_ylabel("h(q)")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_tau(results, path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for ticker, result in results.items():
        ax.plot(result["q"], result["tau"], marker="o", linewidth=1.6, label=ticker)
    ax.set_title("Mass exponent function after deseasonalization")
    ax.set_xlabel("q")
    ax.set_ylabel("tau(q)")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_spectra(results, path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, result in results.items():
        linestyle = "--" if name.startswith("fGn") else "-"
        ax.plot(result["alpha"], result["f_alpha"], marker="o", markersize=3,
                linewidth=1.5, linestyle=linestyle, label=name)
    ax.set_title("MF-DFA singularity spectra")
    ax.set_xlabel("alpha")
    ax.set_ylabel("f(alpha)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_widths(summary, benchmark_summary, path):
    rows = []
    for _, row in summary.iterrows():
        rows.append({
            "series": row["series"],
            "width": row["spectrum_width"],
            "err": 0.0,
            "group": "real"
        })
    for row in benchmark_summary:
        rows.append({
            "series": row["series"],
            "width": row["spectrum_width_mean"],
            "err": row["spectrum_width_std"],
            "group": "model"
        })
    df = pd.DataFrame(rows)
    colors = ["#4c78a8" if g == "real" else "#999999" for g in df["group"]]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df["series"], df["width"], yerr=df["err"], capsize=4, color=colors)
    ax.set_title("Width of singularity spectrum")
    ax.set_xlabel("series")
    ax.set_ylabel("max(alpha) - min(alpha)")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_control_widths(summary, controls, path):
    real = summary.set_index("series")["spectrum_width"]
    controls = pd.DataFrame(controls).set_index("series")
    tickers = list(real.index)
    x = np.arange(len(tickers))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width, [real[t] for t in tickers], width, label="real deseasonalized")
    ax.bar(x, controls.loc[tickers, "shuffle_width_mean"], width,
           yerr=controls.loc[tickers, "shuffle_width_std"], capsize=3, label="shuffle")
    ax.bar(x + width, controls.loc[tickers, "phase_width_mean"], width,
           yerr=controls.loc[tickers, "phase_width_std"], capsize=3, label="phase randomized")
    ax.set_xticks(x)
    ax.set_xticklabels(tickers)
    ax.set_title("MF-DFA width controls")
    ax.set_ylabel("spectrum width")
    ax.grid(alpha=0.25, axis="y")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def write_notes(summary, benchmark_summary, controls, path):
    real = summary.copy()
    widest = real.sort_values("spectrum_width", ascending=False).iloc[0]
    narrowest = real.sort_values("spectrum_width").iloc[0]
    mean_alpha = real["dfa_alpha"].mean()
    h05 = [r for r in benchmark_summary if r["series"] == "fGn H=0.5"][0]
    control_df = pd.DataFrame(controls)
    lines = [
        "# Short notes for the report",
        "",
        f"- The average DFA alpha across deseasonalized real assets is {mean_alpha:.3f}.",
        f"- The widest real-data spectrum is observed for {widest['series']} "
        f"(width {widest['spectrum_width']:.3f}).",
        f"- The narrowest real-data spectrum is observed for {narrowest['series']} "
        f"(width {narrowest['spectrum_width']:.3f}).",
        f"- The fGn H=0.5 Monte Carlo spectrum-width mean is "
        f"{h05['spectrum_width_mean']:.3f} with std {h05['spectrum_width_std']:.3f}.",
        "- Control means by ticker:",
    ]
    for _, row in control_df.iterrows():
        lines.append(
            f"  - {row['series']}: real={row['real_width']:.3f}, "
            f"shuffle={row['shuffle_width_mean']:.3f}, "
            f"phase={row['phase_width_mean']:.3f}, "
            f"shuffle_z={row['shuffle_z']:+.2f}"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
