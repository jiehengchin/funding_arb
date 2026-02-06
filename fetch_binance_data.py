import pandas as pd
import os
import time
from datetime import datetime, timedelta, timezone
from binance.client import Client
from binance.enums import HistoricalKlinesType
from pathlib import Path
from typing import List, Optional

def fetch_binance_futures_data(
    symbols=None, 
    frequency='1h', 
    lookback_days=30, 
    data_folder="/Users/chinjieheng/Documents/research/funding_arb/data_1h_price"
):
    """
    Simplified workflow to fetch Binance Futures Perpetual data and save to Parquet.
    
    Args:
        symbols (list): List of symbols like ['BTCUSDT']. If None, fetches all USDT perpetuals.
        frequency (str): Kline interval ('1m', '5m', '15m', '1h', '1d').
        lookback_days (int): Number of days to look back from today 00:00 UTC.
        data_folder (str): Directory where parquet files will be stored.
    """
    client = Client()
    
    # Ensure data folder exists
    Path(data_folder).mkdir(parents=True, exist_ok=True)
    
    # Test connection
    try:
        client.ping()
    except Exception as e:
        print(f"[Error] Could not connect to Binance API: {e}")
        print("Skipping download. Check your internet connection or VPN.")
        return

    # Calculate end date: Today 00:00 UTC
    end_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    # Calculate start date: End date - lookback
    start_date = end_date - timedelta(days=lookback_days)
    
    start_str = start_date.strftime("%d %b %Y %H:%M:%S")
    end_str = end_date.strftime("%d %b %Y %H:%M:%S")
    
    print(f"Fetching {frequency} data for {lookback_days} days.")
    print(f"Range: {start_str} to {end_str}")

    # If no symbols provided, fetch all USDT Perpetual contracts
    if symbols is None:
        print("Scanning for all USDT-Future Perpetual symbols...")
        info = client.futures_exchange_info()
        symbols = [
            s['symbol'] for s in info['symbols'] 
            if s['contractType'] == 'PERPETUAL' and s["status"] == "TRADING" and s['symbol'].endswith('USDT')
        ]
        print(f"Found {len(symbols)} symbols to download.")

    for i, symbol in enumerate(symbols):
        try:
            print(f"[{i+1}/{len(symbols)}] Downloading {symbol}...", end="\r")
            
            klines = client.get_historical_klines(
                symbol, 
                frequency, 
                start_str, 
                end_str,
                klines_type=HistoricalKlinesType.FUTURES
            )
            
            if not klines:
                continue
                
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 
                'volume', 'close_time', 'quote_av', 'trades', 
                'tb_base_av', 'tb_quote_av', 'ignore'
            ])
            
            # Data Cleaning & Formatting
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Convert numeric columns
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            
            # Drop unnecessary columns
            if 'ignore' in df.columns:
                df.drop('ignore', axis=1, inplace=True)
            
            # Save to Parquet (more efficient than CSV)
            filename = Path(data_folder) / f"{symbol}-{frequency}-data.parquet"
            df.to_parquet(filename)
            
            # Respect rate limits
            time.sleep(0.5)
            
        except Exception as e:
            print(f"\nFailed to fetch {symbol}: {e}")

    print("\nAll data fetching tasks completed.")


def fetch_binance_funding_rates(
    symbols: Optional[List[str]] = None,
    lookback_days: int = 90,
    data_folder: str = "/Users/chinjieheng/Documents/research/funding_arb/data_fundingrate"
):
    """
    Fetches historical funding rate data for Binance Futures.
    """
    client = Client()
    Path(data_folder).mkdir(parents=True, exist_ok=True)
    
    # Test connection
    try:
        client.ping()
    except Exception as e:
        print(f"[Error] Could not connect to Binance API: {e}")
        print("Skipping funding rate fetch. Check your internet connection or VPN.")
        return
    
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=lookback_days)
    start_ts = int(start_date.timestamp() * 1000)
    end_ts = int(end_date.timestamp() * 1000)
    
    print(f"Fetching funding rates for {lookback_days} days...")
    
    if symbols is None:
        print("Scanning for all USDT-Future Perpetual symbols...")
        info = client.futures_exchange_info()
        symbols = [
            s['symbol'] for s in info['symbols'] 
            if s['contractType'] == 'PERPETUAL' and s["status"] == "TRADING" and s['symbol'].endswith('USDT')
        ]
        print(f"Found {len(symbols)} symbols.")

    total_symbols = len(symbols)
    for i, symbol in enumerate(symbols):
        try:
            print(f"[{i+1}/{total_symbols}] Fetching funding for {symbol}...", end="\r")
            
            # Fetch all funding rates with pagination
            all_rates = []
            current_start = start_ts
            
            while True:
                rates = client.futures_funding_rate(
                    symbol=symbol,
                    startTime=current_start,
                    endTime=end_ts,
                    limit=1000
                )
                
                if not rates:
                    break
                    
                all_rates.extend(rates)
                
                # Update start time for next batch
                last_ts = rates[-1]['fundingTime']
                if last_ts >= end_ts or len(rates) < 1000:
                    break
                current_start = last_ts + 1
            
            if not all_rates:
                continue
                
            df = pd.DataFrame(all_rates)
            
            # Formatting
            df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
            df['fundingRate'] = pd.to_numeric(df['fundingRate'])
            df.set_index('fundingTime', inplace=True)
            
            # Estimate interval in hours (usually 8, sometimes 4 or 1)
            # Calculate mode of time difference
            if len(df) > 1:
                diffs = df.index.to_series().diff().dt.total_seconds() / 3600
                interval = diffs.mode().iloc[0] if not diffs.mode().empty else 8
            else:
                interval = 8
            
            df['fundingIntervalHours'] = interval
            
            # Save to Parquet
            filename = Path(data_folder) / f"{symbol}-funding-data.parquet"
            df.to_parquet(filename)
            
            time.sleep(0.1) # Rate limit niceness
            
        except Exception as e:
            print(f"\nFailed to fetch funding for {symbol}: {e}")
            
    print("\nFunding rate fetching completed.")


if __name__ == "__main__":
    # --- USAGE CONFIGURATION ---
    # Change these values to adjust your workflow
    FREQ = '1h'           # Options: '1m', '5m', '15m', '1h', '4h', '1d'
    LOOKBACK = 1         # Number of days to fetch
    TARGET_FOLDER = "/Users/chinjieheng/Documents/research/funding_arb/data_1h_price"
    
    # To fetch specific coins, pass a list: symbols=['BTCUSDT', 'ETHUSDT']
    # To fetch all available USDT-Perp futures, leave symbols=None
        # Fetch Funding Rates
    FUNDING_FOLDER = "/Users/chinjieheng/Documents/research/funding_arb/data_fundingrate"
    fetch_binance_funding_rates(
        symbols=None,
        lookback_days=LOOKBACK,
        data_folder=FUNDING_FOLDER
    )
    fetch_binance_futures_data(
        symbols=None, 
        frequency=FREQ, 
        lookback_days=LOOKBACK, 
        data_folder=TARGET_FOLDER
    )


