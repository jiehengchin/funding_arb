import pandas as pd
import numpy as np
import os

def compute_max_drawdown(series):
    cum = (1 + series).cumprod()
    peak = cum.expanding(min_periods=1).max()
    dd = (cum - peak) / peak
    return dd.min()

def compute_sharpe(series, periods=365):
    if series.std() == 0: return np.nan
    return series.mean() / series.std() * np.sqrt(periods)

def generate_report():
    print("Loading detailed records...")
    try:
        df = pd.read_csv('detailed_records_framework_wf.csv')
    except FileNotFoundError:
        print("Error: detailed_records_framework_wf.csv not found.")
        return

    df['date'] = pd.to_datetime(df['date'])
    
    # 1. Daily Strategy Metrics
    # Group by date to aggregate portfolio level stats
    print("Calculating daily metrics...")
    
    def agg_daily(x):
        # Contribution sum
        total_price_ret = x['price_return_contrib'].sum()
        total_funding_ret = x['funding_return_contrib'].sum()
        
        # Exposure
        long_mask = x['weight'] > 0
        short_mask = x['weight'] < 0
        
        long_exp = x.loc[long_mask, 'weight'].sum()
        short_exp = x.loc[short_mask, 'weight'].sum()
        
        # Weighted Beta (Ex-ante) - handle NaNs in beta by filling with 0 for aggregation
        # (Assuming missing beta means 0 exposure to market risk or data error)
        w_beta = (x['weight'] * x['beta'].fillna(0)).sum()
        
        # Long/Short Return Breakdown
        long_ret_contrib = (x.loc[long_mask, 'price_return_contrib'] + x.loc[long_mask, 'funding_return_contrib']).sum()
        short_ret_contrib = (x.loc[short_mask, 'price_return_contrib'] + x.loc[short_mask, 'funding_return_contrib']).sum()
        
        return pd.Series({
            'strategy_return': total_price_ret + total_funding_ret,
            'price_return': total_price_ret,
            'funding_return': total_funding_ret,
            'long_exposure': long_exp,
            'short_exposure': short_exp,
            'gross_exposure': x['weight'].abs().sum(),
            'net_exposure': x['weight'].sum(),
            'weighted_beta': w_beta,
            'long_return_contribution': long_ret_contrib,
            'short_return_contribution': short_ret_contrib,
            'long_positions': long_mask.sum(),
            'short_positions': short_mask.sum()
        })

    daily_stats = df.groupby('date').apply(agg_daily).sort_index()
    
    # Initialize column for ex-post beta
    daily_stats['rolling_ex_post_beta'] = np.nan

    # 2. Load BTC Data for Beta calculation
    btc_path = '/Users/chinjieheng/Documents/data/binance_1Hdata/BTCUSDT-1h-data.parquet'
    realized_beta = np.nan
    correlation = np.nan
    
    if os.path.exists(btc_path):
        print(f"Loading BTC data from {btc_path}...")
        try:
            btc_df = pd.read_parquet(btc_path)
            
            # Ensure index is datetime
            if not isinstance(btc_df.index, pd.DatetimeIndex):
                if 'open_time' in btc_df.columns:
                    btc_df['open_time'] = pd.to_datetime(btc_df['open_time'])
                    btc_df.set_index('open_time', inplace=True)
                elif 'date' in btc_df.columns:
                    btc_df['date'] = pd.to_datetime(btc_df['date'])
                    btc_df.set_index('date', inplace=True)
            
            # Resample to Daily Close and calculate returns
            if 'close' in btc_df.columns:
                btc_daily = btc_df['close'].resample('D').last().pct_change().dropna()
                btc_daily.name = 'btc_return'
                
                # Merge with strategy returns
                merged = daily_stats.join(btc_daily, how='inner')
                
                if not merged.empty:
                    # Calculate Realized Beta (Scalar)
                    cov = merged['price_return'].cov(merged['btc_return'])
                    var = merged['btc_return'].var()
                    realized_beta = cov / var if var != 0 else np.nan
                    correlation = merged['price_return'].corr(merged['btc_return'])
                    print(f"Computed Realized Beta vs BTC: {realized_beta:.4f}")
                    
                    # Calculate Rolling Realized Beta (Ex-Post) - 30 Day Window
                    rolling_cov = merged['price_return'].rolling(window=30).cov(merged['btc_return'])
                    rolling_var = merged['btc_return'].rolling(window=30).var()
                    
                    # Assign back to daily_stats
                    # Use reindex to align with daily_stats index (filling missing dates with NaN)
                    rolling_beta_series = (rolling_cov / rolling_var)
                    daily_stats.loc[rolling_beta_series.index, 'rolling_ex_post_beta'] = rolling_beta_series
            else:
                print("Column 'close' not found in BTC data.")
        except Exception as e:
            print(f"Failed to process BTC data: {e}")
    else:
        print("BTC data not found. Skipping realized beta calculation.")

    # 3. Aggregate Performance Stats
    total_ret = (1 + daily_stats['strategy_return']).cumprod().iloc[-1] - 1
    days = (daily_stats.index[-1] - daily_stats.index[0]).days
    annualized_ret = (1 + total_ret) ** (365 / days) - 1 if days > 0 else 0
    sharpe = compute_sharpe(daily_stats['strategy_return'])
    max_dd = compute_max_drawdown(daily_stats['strategy_return'])
    
    # 4. Formatting Report
    lines = []
    lines.append("="*50)
    lines.append("FUNDING ARBITRAGE STRATEGY REPORT")
    lines.append("="*50)
    lines.append(f"Date Range: {daily_stats.index.min().date()} to {daily_stats.index.max().date()}")
    lines.append(f"Days Traded: {len(daily_stats)}")
    lines.append("-" * 30)
    lines.append("PERFORMANCE METRICS")
    lines.append(f"Total Return:       {total_ret*100:.2f}%")
    lines.append(f"Annualized Return:  {annualized_ret*100:.2f}%")
    lines.append(f"Sharpe Ratio:       {sharpe:.2f}")
    lines.append(f"Max Drawdown:       {max_dd*100:.2f}%")
    lines.append("-" * 30)
    lines.append("RISK METRICS")
    if not np.isnan(realized_beta):
        lines.append(f"Realized Beta (vs BTC): {realized_beta:.4f}")
        lines.append(f"Correlation (vs BTC):   {correlation:.4f}")
    lines.append(f"Avg Daily Weighted Beta: {daily_stats['weighted_beta'].mean():.4f}")
    lines.append("-" * 30)
    lines.append("EXPOSURE (Average)")
    lines.append(f"Gross Exposure:     {daily_stats['gross_exposure'].mean():.2f}")
    lines.append(f"Net Exposure:       {daily_stats['net_exposure'].mean():.2f}")
    lines.append(f"Long Exposure:      {daily_stats['long_exposure'].mean():.2f}")
    lines.append(f"Short Exposure:     {daily_stats['short_exposure'].mean():.2f}")
    lines.append("-" * 30)
    lines.append("RETURN ATTRIBUTION (Cumulative Sum)")
    lines.append(f"Price Return:       {daily_stats['price_return'].sum()*100:.2f}%")
    lines.append(f"Funding Return:     {daily_stats['funding_return'].sum()*100:.2f}%")
    lines.append("-" * 30)
    lines.append("LONG/SHORT ATTRIBUTION (Cumulative Contribution)")
    lines.append(f"Long Positions:     {daily_stats['long_return_contribution'].sum()*100:.2f}%")
    lines.append(f"Short Positions:    {daily_stats['short_return_contribution'].sum()*100:.2f}%")
    lines.append("="*50)
    
    report_text = "\n".join(lines)
    print(report_text)
    
    # Save to file
    with open("strategy_report.txt", "w") as f:
        f.write(report_text)
    
    daily_stats.to_csv("daily_strategy_metrics.csv")
    print("\nReport saved to 'strategy_report.txt'")
    print("Daily metrics saved to 'daily_strategy_metrics.csv'")

    # 5. Generate Figures
    print("\nGenerating figures...")
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        # Create figures directory
        if not os.path.exists('figures'):
            os.makedirs('figures')
            
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # A. Cumulative Returns Breakdown
        fig, ax = plt.subplots(figsize=(12, 6))
        cum_total = (1 + daily_stats['strategy_return']).cumprod()
        cum_price = daily_stats['price_return'].cumsum()
        cum_funding = daily_stats['funding_return'].cumsum()
        
        ax.plot(cum_total.index, cum_total, label='Total Return (Compound)', linewidth=2)
        ax.plot(cum_price.index, cum_price, label='Price Return (Sum)', linestyle='--', alpha=0.7)
        ax.plot(cum_funding.index, cum_funding, label='Funding Return (Sum)', linestyle='--', alpha=0.7)
        
        ax.set_title('Strategy Cumulative Returns Breakdown')
        ax.set_ylabel('Return')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig('figures/cumulative_returns.png', dpi=300)
        plt.close(fig)
        
        # B. Exposure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Gross/Net
        ax1.plot(daily_stats.index, daily_stats['gross_exposure'], label='Gross Exposure', color='black')
        ax1.plot(daily_stats.index, daily_stats['net_exposure'], label='Net Exposure', color='blue', alpha=0.6)
        ax1.set_title('Portfolio Exposure')
        ax1.set_ylabel('Exposure')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Long/Short
        ax2.fill_between(daily_stats.index, daily_stats['long_exposure'], 0, color='green', alpha=0.3, label='Long')
        ax2.fill_between(daily_stats.index, daily_stats['short_exposure'], 0, color='red', alpha=0.3, label='Short')
        ax2.plot(daily_stats.index, daily_stats['long_exposure'], color='green', linewidth=1)
        ax2.plot(daily_stats.index, daily_stats['short_exposure'], color='red', linewidth=1)
        ax2.set_ylabel('Long/Short Exposure')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        fig.tight_layout()
        fig.savefig('figures/exposures.png', dpi=300)
        plt.close(fig)
        
        # C. Drawdown
        fig, ax = plt.subplots(figsize=(12, 4))
        cum = (1 + daily_stats['strategy_return']).cumprod()
        peak = cum.expanding(min_periods=1).max()
        dd = (cum - peak) / peak
        
        ax.fill_between(dd.index, dd, 0, color='red', alpha=0.3)
        ax.plot(dd.index, dd, color='red', linewidth=1)
        ax.set_title('Strategy Drawdown')
        ax.set_ylabel('Drawdown')
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        fig.savefig('figures/drawdown.png', dpi=300)
        plt.close(fig)
        
        # D. Beta (Weighted)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(daily_stats.index, daily_stats['weighted_beta'], color='purple', linewidth=1.5)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax.set_title('Portfolio Weighted Beta (Ex-Ante)')
        ax.set_ylabel('Beta')
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        fig.savefig('figures/weighted_beta.png', dpi=300)
        plt.close(fig)

        # E. Rolling Realized Beta (Ex-Post)
        if 'rolling_ex_post_beta' in daily_stats.columns and daily_stats['rolling_ex_post_beta'].notna().any():
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(daily_stats.index, daily_stats['rolling_ex_post_beta'], color='darkorange', linewidth=1.5, label='30-Day Rolling Beta')
            ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
            ax.set_title('Strategy Realized Beta vs BTC (Ex-Post)')
            ax.set_ylabel('Beta')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            fig.tight_layout()
            fig.savefig('figures/ex_post_beta.png', dpi=300)
            plt.close(fig)

        print("Figures saved to 'figures/' directory.")

    except ImportError:
        print("Matplotlib not installed. Skipping figure generation.")
    except Exception as e:
        print(f"Error generating figures: {e}")

if __name__ == "__main__":
    generate_report()