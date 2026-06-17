# Multifractal Finance Analysis

Small research project for the seminar assignment on multifractal analysis of
financial data. The code downloads intraday Yahoo Finance quotes, prepares
log-return series, removes the strongest intraday volatility seasonality, runs
DFA/MF-DFA, and compares real data with fractional Gaussian noise benchmarks and
shuffle / phase-randomization controls.

## Workflow

1. Download real intraday quotes from Yahoo Finance.
2. Convert prices into log-returns and deseasonalized log-returns.
3. Check basic stylized facts of financial returns.
4. Apply DFA and MF-DFA.
5. Generate Monte Carlo fractional Gaussian noise benchmarks.
6. Compare singularity spectra of real data, model data, and controls.

## Data

The default assets are:

- `SPY`: broad US equity market ETF.
- `QQQ`: Nasdaq-100 ETF.
- `AAPL`: Apple stock.
- `MSFT`: Microsoft stock.

The default sampling scheme is `5m` bars over `60d`. The saved data used in the
report cover 2026-03-23 13:35 UTC to 2026-06-16 19:55 UTC and contain 4620
cleaned intraday returns per asset. Re-running the downloader later may produce
a different rolling 60-day window.

## Project Structure

```text
data/
  raw/                 Raw Yahoo Finance OHLCV files.
  processed/           Cleaned log-prices and log-returns.
outputs/
  figures/             Plots for the report.
  tables/              CSV summaries for the report.
scripts/
  01_download_quotes.py
  02_prepare_returns.py
  03_make_exploratory_figures.py
  04_run_mfdfa.py
  05_make_report_figures.py
src/
  preprocessing.py
  dfa.py
  fbm.py
```

## How To Run

From the project root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python scripts/01_download_quotes.py
.venv/bin/python scripts/02_prepare_returns.py
.venv/bin/python scripts/03_make_exploratory_figures.py
.venv/bin/python scripts/04_run_mfdfa.py
.venv/bin/python scripts/05_make_report_figures.py
```

The scripts write CSV files to `data/` and report-ready figures/tables to
`outputs/`.

## Notes

The preprocessing step removes returns across large time gaps. Otherwise the
overnight jump between two trading days would be mixed with ordinary 5-minute
returns. For final DFA/MF-DFA calculations, returns are also normalized by
time-of-day volatility to reduce the deterministic intraday U-shaped volatility
pattern.
