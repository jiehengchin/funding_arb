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
- **`beta_neu_fund_arb_EDA.ipynb`**: Exploratory data analysis notebook. *Note: This notebook is for exploratory purposes and may use simplified or different beta calculation methods/parameters compared to the main `funding_arb_framework.py` for clarity and rapid prototyping. It's not intended for direct strategy execution.*

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

### 5. Understanding the Figures

The `generate_report.py` script produces several visualizations in the `figures/` directory to help diagnose strategy performance:

- **`cumulative_returns.png`**:
  - **What it shows:** Total cumulative return vs. its components (Price PnL and Funding PnL).
  - **What to look for:** Ideally, **Funding PnL (green/dashed)** should drive the total return. **Price PnL (blue/dashed)** should be relatively flat or oscillate around zero, indicating effective hedging.

- **`exposures.png`**:
  - **What it shows:** Gross and Net leverage over time, and the breakdown of Long vs. Short positions.
  - **What to look for:** **Net Exposure** should stay close to zero (market neutral). **Gross Exposure** shows your total leverage utilization.

- **`drawdown.png`**:
  - **What it shows:** The percentage decline from the strategy's historical peak equity.
  - **What to look for:** Periods of deep drawdowns indicate risks (e.g., correlation breakdowns, liquidation cascades).

- **`weighted_beta.png` (Ex-Ante)**:
  - **What it shows:** The portfolio's expected beta *before* trades were executed, based on historical data.
  - **What to look for:** This should be extremely close to 0.0, confirming the optimizer satisfied the neutrality constraints.

- **`ex_post_beta.png` (Realized)**:
  - **What it shows:** The actual **Rolling 30-Day Beta** of the strategy's *Price Return* against BTC.
  - **What to look for:** This measures "beta slippage." Ideally, it should hover near 0. If it deviates significantly (e.g., > 0.3 or < -0.3), the historical beta estimates failed to predict future correlations.

- **`hedging_efficiency.png`**:
  - **What it shows:** Overlays the Rolling Realized Beta against the Cumulative Price PnL. Red shaded regions indicate days where `|Beta| > 0.3`.
  - **What to look for:** Use this to check if large losses in Price PnL align with periods of high realized beta (hedging failure). If Price PnL drops while Beta is high, market exposure caused the loss.

## Strategy Configuration

The strategy parameters are defined in the `FundingArbParams` class in `funding_arb_framework.py`. Key parameters include:

- `ar_window`: Lookback window for AR model training.
- `beta_window`: Lookback window for beta calculation.
- `portfolio_size_each_side`: Number of assets to hold on Long/Short sides.
- `beta_limit`: Maximum allowed portfolio beta (for optimization constraints).
- `tc_bps`: Transaction costs in basis points (for simulation).
