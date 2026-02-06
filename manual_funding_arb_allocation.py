import pandas as pd
import numpy as np
import os
import time
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from binance.client import Client
from typing import List, Optional

# Import project modules
# Assuming these are in the same directory
try:
    from fetch_binance_data import fetch_binance_futures_data, fetch_binance_funding_rates
    from funding_arb_framework import (
        FundingDataBundle, 
        FundingArbParams, 
        FundingArbStrategy, 
        BetaNeutralWeighting
    )
    from binance_data_loader import BinanceDataLoader
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure fetch_binance_data.py, funding_arb_framework.py, and binance_data_loader.py are in the current directory.")
    exit(1)


def main():
    parser = argparse.ArgumentParser(description="Manual Funding Arbitrage Allocation Calculator")
    parser.add_argument("--equity", type=float, default=100.0, help="Total equity in USD (default: 100.0)")
    parser.add_argument("--lookback", type=int, default=120, help="Lookback days for data fetching (default: 120)")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip data fetching (use existing local data)")
    parser.add_argument("--price-dir", type=str, default="/Users/chinjieheng/Documents/research/funding_arb/data_1h_price", help="Directory for price data")
    parser.add_argument("--funding-dir", type=str, default="/Users/chinjieheng/Documents/research/funding_arb/data_fundingrate", help="Directory for funding data")
    
    args = parser.parse_args()
    
    # 1. Fetch Data
    if not args.skip_fetch:
        print("--- Step 1: Fetching Market Data ---")
        # Fetch Price (1h)
        fetch_binance_futures_data(
            symbols=None, # All symbols
            frequency='1h',
            lookback_days=args.lookback,
            data_folder=args.price_dir
        )
        # Fetch Funding
        fetch_binance_funding_rates(
            symbols=None, # All symbols
            lookback_days=args.lookback,
            data_folder=args.funding_dir
        )
    else:
        print("Skipping data fetching, using local data...")

    # 2. Load Data
    print("\n--- Step 2: Loading & Processing Data ---")
    loader = BinanceDataLoader(
        data_directory=args.price_dir,
        funding_rate_directory=args.funding_dir,
        timeframe='1h',
        min_records=args.lookback * 24 * 0.75, # Ensure enough data (~80% of lookback)
        min_volume=1e4, # Lower volume filter for manual check? Or keep 1e6?
        start_date=(datetime.now() - timedelta(days=args.lookback)).strftime("%Y-%m-%d"),
        end_date=None
    )
    
    # Prepare Data Matrices (mimicking notebook logic)
    print("Resampling data to Daily/Hourly...")
    price_hf = loader.get_price_matrix()
    # Loader returns 1h data (since we asked for 1h), so no need to resample for HF
    # But ensure it is sorted
    price_hf.sort_index(inplace=True)
    
    # Get Volume Data for Framework
    volume_hf = loader.get_volume_matrix(vol_30d=False)
    volume_hf.sort_index(inplace=True)
    
    returns_df_hf = price_hf.pct_change(fill_method=None)
    
    # Prepare Daily Data for Strategy Heartbeat
    price_daily = price_hf.resample('D').last()
    returns_daily = price_daily.pct_change(fill_method=None)
    
    # Process Funding Data (Aggregate to Daily)
    funding_long = loader.get_funding_long_form()
    daily_funding = (
        funding_long['fundingRate']
        .unstack(level=0)
        .sort_index()
        .resample('D').sum(min_count=1)
    )
    
    # Create Bundle
    bundle = FundingDataBundle(
        price_df=price_daily,
        funding_df=daily_funding,
        returns_df=returns_daily,
        returns_df_hf=returns_df_hf,
        hf_window_multiplier=24, # 1h data -> 24 bars per day
        min_hist_days=30,
        hf_resample_rule='D', # Resample HF betas to Daily
        volume_df=volume_hf  # Pass raw volume matrix
    )
    
    print(f"\n--- Data Verification ---")
    print(f"Last Price Data Date: {bundle.price_df.index.max()}")
    print(f"Last Funding Data Date: {bundle.funding_df.index.max()}")
    print(f"Last available date for strategy: {pd.Timestamp(bundle.dates[-1]).date()}")

    # 3. Initialize Strategy with Best Params
    print("\n--- Step 3: Running Strategy ---")
    # Best params from notebook output
    params = FundingArbParams(
        ar_window=30,
        beta_window=30,
        portfolio_size_each_side=6,
        beta_limit=1e-4,
        beta_type="adaptive",
        gross_exposure_limit=0.3,
        tc_bps=5.0,
        min_positions=6,
        min_weight=0.037,
        use_shrinkage=True,
        prior_beta_window=90,
        min_volume=100000.0 # Match notebook filter (100k ADV)
    )
    
    strategy = FundingArbStrategy(params)
    weighting = BetaNeutralWeighting()
    
    # Prepare Strategy (Precompute AR forecasts, Betas, etc.)
    print("Precomputing signals (this may take a moment)...")
    strategy.prepare(bundle)
    
    # 4. Generate Allocation for LATEST available date
    # We want the last date where we have data.
    # Check last date in price and funding
    last_date = bundle.dates[-1]
    last_date_ts = pd.Timestamp(last_date)
    
    print(f"Generating allocation for date: {last_date_ts.date()}")

    # --- Data Freshness Check ---
    # Check if data is stale (older than 48 hours)
    time_since_last = datetime.now(timezone.utc) - last_date_ts.replace(tzinfo=timezone.utc)
    if time_since_last > timedelta(hours=48):
        print(f"\n[WARNING] DATA MIGHT BE STALE!")
        print(f"The latest data point is from {last_date_ts.date()}, which is {time_since_last.days} days ago.")
        print("Please check if the download step completed successfully.")
        # We proceed, but user is warned.

    # Universe Mask
    # We need to use index -1 (last available closed bar) to predict for "Next Day"
    # In backtest: signal at 'i' (today) uses data up to 'i'. Allocation is for 'i' to 'i+1'.
    idx = len(bundle.dates) - 1
    decision_date = pd.Timestamp(bundle.dates[idx])

    signals = strategy.signals(idx, bundle)

    # Real-time trading: use current-day data only (no next-day dependency)
    p_curr = bundle.price_df.loc[decision_date].to_numpy()
    has_price = np.isfinite(p_curr)

    if decision_date in bundle.funding_df.index:
        f_curr = bundle.funding_df.loc[decision_date].to_numpy()
        has_funding = np.isfinite(f_curr)
    else:
        has_funding = np.zeros(len(bundle.tickers), dtype=bool)

    base_mask = has_price & has_funding

    # Require finite alpha and beta (match stricter notebook-style mask)
    beta = signals["beta"]
    alpha = signals["alpha"]
    valid_beta = np.isfinite(beta)
    valid_alpha = np.isfinite(alpha)
    uni_mask = base_mask & valid_beta & valid_alpha

    print(f"Coins with valid price data: {np.sum(has_price)} / {len(bundle.tickers)}")
    print(f"Coins with valid funding data: {np.sum(has_funding)} / {len(bundle.tickers)}")
    print(f"Coins with valid beta: {np.sum(valid_beta)} / {len(bundle.tickers)}")
    print(f"Coins with valid alpha: {np.sum(valid_alpha)} / {len(bundle.tickers)}")

    # Calculate Volume Mask (Match Backtest)
    if params.min_volume > 0 and bundle.volume_df is not None:
         # Use same logic as backtest: rolling 30d sum / 30, aligned to price_df index
         vol_df = bundle.volume_df.copy()
         if not isinstance(vol_df.index, pd.DatetimeIndex):
             try:
                 vol_df.index = pd.to_datetime(vol_df.index)
             except Exception:
                 print("[WARNING] Volume DF index not datetime; skipping volume filter.")
                 vol_df = None

         if vol_df is not None:
             avg_daily_vol = vol_df.rolling('30D', min_periods=1).sum() / 30.0
             avg_daily_vol = avg_daily_vol.reindex(bundle.price_df.index).fillna(0.0)

             if decision_date in avg_daily_vol.index:
                 current_vol = avg_daily_vol.loc[decision_date].to_numpy()
                 current_vol = np.nan_to_num(current_vol, 0.0)
                 vol_mask = (current_vol >= params.min_volume)
                 uni_mask = uni_mask & vol_mask
                 print(f"Coins meeting volume criteria (> ${params.min_volume:,.0f}): {np.sum(vol_mask)}")
             else:
                 print("[WARNING] Could not find volume data for decision date, skipping volume filter.")

    print(f"Coins eligible for trading (Price + Signal + Vol): {np.sum(uni_mask)}")
    
    weights = weighting.weights(idx, signals, bundle, uni_mask, params)
    
    # 5. Output Results
    print(f"\n{'='*40}")
    print(f"TARGET ALLOCATION (Equity: ${args.equity:,.2f})")
    print(f" { '='*40}")
    print(f" { 'Symbol':<15} {'Side':<6} {'Weight':<10} {'USD Amount':<15}")
    print(f"{'-'*46}")
    
    allocations = []
    
    for i, w in enumerate(weights):
        if abs(w) > 1e-4: # Filter small dust
            symbol = bundle.tickers[i]
            side = "LONG" if w > 0 else "SHORT"
            usd_amt = w * args.equity
            allocations.append({
                'Symbol': symbol,
                'Side': side,
                'Weight': w,
                'USD': usd_amt,
                'Abs_USD': abs(usd_amt)
            })
    
    # Sort by Absolute USD Amount (Desc)
    allocations.sort(key=lambda x: x['Abs_USD'], reverse=True)
    
    total_long = 0
    total_short = 0
    
    for item in allocations:
        print(f"{item['Symbol']:<15} {item['Side']:<6} {item['Weight']:<10.4f} ${item['USD']:<15.2f}")
        if item['Weight'] > 0:
            total_long += item['USD']
        else:
            total_short += item['USD']
            
    print(f"{'-'*46}")
    print(f"Total Long:  ${total_long:,.2f}")
    print(f"Total Short: ${total_short:,.2f}")
    print(f"Net Exposure: ${total_long + total_short:,.2f}")
    print(f"Gross Exposure: ${total_long - total_short:,.2f}")
    print(f" { '='*40}")

if __name__ == "__main__":
    main()
