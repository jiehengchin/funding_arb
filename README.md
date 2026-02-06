# Beta-Neutral Funding Arbitrage Strategy

This project implements a quantitative trading strategy designed to capture funding rate arbitrage opportunities in the Binance Perpetual Futures market while maintaining beta-neutrality to Bitcoin (or the broader market).

## Overview

The strategy identifies assets with high predicted funding rates and constructs a long/short portfolio. It hedges market risk (beta) by optimizing position weights or using a specific hedge asset (like BTC).

Key components:
- **Funding Rate Forecasting**: Uses AutoRegressive (AR) models to predict future funding rates.
- **Beta Neutrality**: Calculates realized betas and constructs a portfolio with near-zero net beta.
- **Shrinkage Estimators**: Applies statistical shrinkage to beta estimates to improve out-of-sample stability.

## Project Structure

- **`funding_arb_framework.py`**: The core library containing the strategy logic, data bundle structures, and backtesting engine.
- **`manual_funding_arb_allocation.py`**: Main script for live trading. Calculates the target portfolio allocation for the current day.
- **`fetch_binance_data.py`**: Utilities to download historical price and funding rate data from Binance. (for `manual_funding_arb_allocation.py` only)
- **`generate_report.py`**: Generates performance metrics and reports from backtest results.
- **`funding_arb_walkforward.ipynb`**: Notebook for running walk-forward backtests.
- **`beta_neu_fund_arb_EDA.ipynb`**: Exploratory data analysis notebook.

## Installation

1. Clone the repository.
2. Install the required Python packages:

```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels binance-connector cvxpy
```

## Usage

### 1. Data Collection

Before running the strategy or backtests, you need historical data. The default directories are `data_1h_price` and `data_fundingrate`.

You can use the `importdatabinance_1h.ipynb` and `importfundingratedata.ipynb`  to download data.

### 2. Generating Daily Allocations (Live Trading)

To generate the target portfolio for the current day, use the `manual_funding_arb_allocation.py` script.

```bash
python manual_funding_arb_allocation.py --equity 1000 --lookback 120
```

**Arguments:**
- `--equity`: Total portfolio equity in USD (default: 100.0).
- `--lookback`: Number of days of history to use for analysis (default: 120).
- `--skip-fetch`: If set, skips downloading fresh data and uses existing local files.
- `--price-dir`: Directory for price data (default: `.../data_1h_price`).
- `--funding-dir`: Directory for funding data (default: `.../data_fundingrate`).

**Output:**
The script will print the target portfolio weights and expected funding rates. It creates a "Target Portfolio" section indicating which assets to Long/Short and their respective sizes.

### 3. Backtesting

To run a backtest / walk-forward analysis:

1. Open `funding_arb_walkforward.ipynb`.
2. Configure the `BinanceDataLoader` paths to point to your data.
3. Run the notebook cells to execute the strategy over the historical period.
4. The notebook will save detailed records to CSV files (e.g., `detailed_records_framework_wf.csv`).

### 4. Performance Reporting

After running a backtest (or if you have a trade record CSV), you can generate a performance report using:

```bash
python generate_report.py
```

This script reads `detailed_records_framework_wf.csv` (by default) and produces:
- `strategy_report.txt`: A text summary of returns, Sharpe ratio, drawdowns, and risk metrics.
- `daily_strategy_metrics.csv`: Aggregated daily performance stats.
- Plots/Figures in the `figures/` directory.

## Strategy Configuration

The strategy parameters are defined in the `FundingArbParams` class in `funding_arb_framework.py`. Key parameters include:

- `ar_window`: Lookback window for AR model training.
- `beta_window`: Lookback window for beta calculation.
- `portfolio_size_each_side`: Number of assets to hold on Long/Short sides.
- `beta_limit`: Maximum allowed portfolio beta (for optimization constraints).
- `tc_bps`: Transaction costs in basis points (for simulation).
