"""
Funding Arbitrage Backtest Framework.

Refactored from beta_neu_fund_arb_01.py into a class-based structure.
Designed to be compatible with walk-forward validation workflows.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import cvxpy as cvx
from statsmodels.tsa.ar_model import AutoReg

from beta_shrink_func import shrink_beta_estimate

EPS = 1e-8
PERIODS_PER_YEAR = 365  # Daily data

# --- Scoring helpers ---

def compute_sharpe(ret_series: pd.Series, periods_per_year: float = PERIODS_PER_YEAR) -> float:
    if ret_series.empty or ret_series.size < 2:
        return float("nan")
    std = ret_series.std(ddof=1)
    if std < 1e-10 or not np.isfinite(std):
        return float("nan")
    return float((ret_series.mean() / std) * np.sqrt(periods_per_year))


def compute_sortino_ratio(ret_series: pd.Series, risk_free_rate: float = 0.0, periods_per_year: float = PERIODS_PER_YEAR) -> float:
    if ret_series.empty or ret_series.size < 2:
        return float("nan")
    daily_rf = risk_free_rate / periods_per_year
    excess_returns = ret_series - daily_rf
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0:
        return 100.0 if ret_series.mean() > daily_rf else float("nan")
    downside_std = downside_returns.std(ddof=1)
    if downside_std < 1e-10 or not np.isfinite(downside_std):
        return float("nan")
    return float((ret_series.mean() / downside_std) * np.sqrt(periods_per_year))


def compute_calmar_ratio(ret_series: pd.Series, equity_series: Optional[pd.Series] = None, min_periods: int = 30, periods_per_year: float = PERIODS_PER_YEAR) -> float:
    if ret_series.empty or ret_series.size < min_periods:
        return float("nan")
    if equity_series is None:
        equity_series = (1 + ret_series).cumprod()
    running_max = equity_series.expanding().max()
    drawdown = (equity_series - running_max) / running_max
    max_drawdown = abs(drawdown.min())
    annualized_return = ret_series.mean() * periods_per_year
    if max_drawdown == 0:
        return 100.0 if annualized_return > 0 else float("nan")
    if not np.isfinite(max_drawdown) or not np.isfinite(annualized_return):
        return float("nan")
    return float(annualized_return / max_drawdown)


def compute_composite_score(ret_series: pd.Series, equity_series: Optional[pd.Series] = None, w_sortino: float = 0.4, w_sharpe: float = 0.3, w_calmar: float = 0.3, periods_per_year: float = PERIODS_PER_YEAR) -> float:
    if ret_series.empty or ret_series.size < 2:
        return float("nan")
    sortino = compute_sortino_ratio(ret_series, periods_per_year=periods_per_year)
    sharpe = compute_sharpe(ret_series, periods_per_year=periods_per_year)
    calmar = compute_calmar_ratio(ret_series, equity_series, periods_per_year=periods_per_year)
    if np.isnan(sortino) or np.isnan(sharpe) or np.isnan(calmar):
        return float("nan")
    total_weight = w_sortino + w_sharpe + w_calmar
    w_sortino /= total_weight
    w_sharpe /= total_weight
    w_calmar /= total_weight
    return float(w_sortino * sortino + w_sharpe * sharpe + w_calmar * calmar)


def select_score(ret_series: pd.Series, equity_series: Optional[pd.Series] = None, mode: str = "composite", periods_per_year: float = PERIODS_PER_YEAR) -> float:
    mode = (mode or "composite").lower()
    sharpe = compute_sharpe(ret_series, periods_per_year=periods_per_year)
    sortino = compute_sortino_ratio(ret_series, periods_per_year=periods_per_year)
    calmar = compute_calmar_ratio(ret_series, equity_series, periods_per_year=periods_per_year)
    if mode == "sharpe":
        return sharpe
    if mode == "sortino":
        return sortino
    if mode == "calmar":
        return calmar
    if mode == "composite":
        return compute_composite_score(
            ret_series,
            equity_series,
            periods_per_year=periods_per_year,
        )
    raise ValueError(f"Unsupported score_mode: {mode}")

# --- Data Structures ---

class FundingDataBundle:
    """
    Container for price, funding, and return data.
    """

    def __init__(
        self,
        price_df: pd.DataFrame,
        funding_df: pd.DataFrame,
        returns_df: Optional[pd.DataFrame] = None,
        btc_ret: Optional[pd.Series] = None,
        eth_ret: Optional[pd.Series] = None,
        min_hist_days: int = 30,
        beta_window: int = 30,
        # HF Data Support
        returns_df_hf: Optional[pd.DataFrame] = None,
        btc_ret_hf: Optional[pd.Series] = None,
        eth_ret_hf: Optional[pd.Series] = None,
        hf_window_multiplier: int = 1,  # e.g. 24 if hf is hourly and base is daily
        hf_resample_rule: Optional[str] = None, # e.g. 'D' to resample HF betas to Daily,
        volume_df: Optional[pd.DataFrame] = None
    ):
        # Align columns (tickers) across all dataframes to ensure consistent shapes
        # We need the intersection of tickers present in Price AND Funding data
        common_tickers = price_df.columns.intersection(funding_df.columns)
        if returns_df is not None:
            common_tickers = common_tickers.intersection(returns_df.columns)
        if volume_df is not None:
            common_tickers = common_tickers.intersection(volume_df.columns)
        
        # Sort tickers to ensure consistent order
        self.tickers = sorted(list(common_tickers))
        
        self.price_df = price_df[self.tickers]
        self.funding_df = funding_df[self.tickers]
        self.volume_df = volume_df[self.tickers] if volume_df is not None else None
        
        if returns_df is not None:
            self.returns_df = returns_df[self.tickers]
        else:
            self.returns_df = self.price_df.pct_change(fill_method=None)
        
        self.btc_ret = btc_ret
        self.eth_ret = eth_ret
        self.min_hist_days = min_hist_days
        self.beta_window = beta_window

        # HF Data
        self.returns_df_hf = returns_df_hf
        self.btc_ret_hf = btc_ret_hf
        self.eth_ret_hf = eth_ret_hf
        self.hf_window_multiplier = hf_window_multiplier
        self.hf_resample_rule = hf_resample_rule

        self.dates = self.price_df.index.to_numpy()
        
        # Store betas by window: {window: {'btc': df, 'eth': df, ...}}
        self.betas: Dict[int, Dict[str, pd.DataFrame]] = {}
        # Store beta variances by window: {window: {'btc': df, 'eth': df, ...}}
        self.beta_vars: Dict[int, Dict[str, pd.DataFrame]] = {}

        # Store predicted funding by AR window: {ar_window: pd.DataFrame}
        self.predicted_funding: Dict[int, pd.DataFrame] = {}

        # Pre-extract market returns if not provided
        if self.btc_ret is None and returns_df is not None and 'BTCUSDT' in returns_df.columns:
             self.btc_ret = returns_df['BTCUSDT']
        elif self.btc_ret is None and 'BTCUSDT' in self.returns_df.columns:
             self.btc_ret = self.returns_df['BTCUSDT']

        if self.eth_ret is None and returns_df is not None and 'ETHUSDT' in returns_df.columns:
             self.eth_ret = returns_df['ETHUSDT']
        elif self.eth_ret is None and 'ETHUSDT' in self.returns_df.columns:
             self.eth_ret = self.returns_df['ETHUSDT']
             
        # Pre-extract HF market returns if not provided
        if self.returns_df_hf is not None:
            if self.btc_ret_hf is None and 'BTCUSDT' in self.returns_df_hf.columns:
                self.btc_ret_hf = self.returns_df_hf['BTCUSDT']
            if self.eth_ret_hf is None and 'ETHUSDT' in self.returns_df_hf.columns:
                self.eth_ret_hf = self.returns_df_hf['ETHUSDT']
            
            # Align HF data to the selected universe (columns) to avoid shape mismatch
            self.returns_df_hf = self.returns_df_hf[self.tickers]

    def ensure_beta_matrix(self, windows: List[int]):
        """
        Precompute rolling betas against BTC, ETH, and Combined (BTC+ETH)/2 for specific windows.
        Also computes 'adaptive' beta (best fit between BTC and ETH).
        """
        if self.btc_ret is None or self.eth_ret is None:
            # Fallback check for HF
            if self.returns_df_hf is None or self.btc_ret_hf is None or self.eth_ret_hf is None:
                warnings.warn("BTC or ETH returns not found (neither daily nor HF). Betas cannot be computed.")
                return

        for w in windows:
            if w in self.betas:
                continue

            print(f"Precomputing rolling betas (window={w})...")
            
            # Decide whether to use HF or Daily
            use_hf = (self.returns_df_hf is not None) and \
                     (self.btc_ret_hf is not None) and \
                     (self.eth_ret_hf is not None)

            if use_hf:
                print(f"  -> Using High-Frequency Data (multiplier={self.hf_window_multiplier}, resample={self.hf_resample_rule})")
                
                eff_window = w * self.hf_window_multiplier
                combined_rets_hf = (self.btc_ret_hf + self.eth_ret_hf) / 2
                
                def compute_beta(market_ret, data_ret):
                    # Beta
                    rolling_cov = data_ret.rolling(eff_window).cov(market_ret)
                    rolling_mkt_var = market_ret.rolling(eff_window).var()
                    beta = rolling_cov.div(rolling_mkt_var, axis=0)
                    
                    # Beta Variance
                    # Var(beta) = [Var(coin) / Var(mkt)] * [(1 - corr^2) / (N - 2)]
                    rolling_corr = data_ret.rolling(eff_window).corr(market_ret)
                    rolling_coin_var = data_ret.rolling(eff_window).var()
                    n_obs = data_ret.rolling(eff_window).count()
                    
                    var_ratio = rolling_coin_var.div(rolling_mkt_var, axis=0)
                    unexplained_corr = 1 - rolling_corr**2
                    dof = n_obs - 2
                    
                    beta_var = var_ratio * unexplained_corr / dof
                    beta_var[dof <= 0] = np.nan
                    
                    # Resample/Align
                    if self.hf_resample_rule:
                        beta = beta.resample(self.hf_resample_rule).last()
                        beta_var = beta_var.resample(self.hf_resample_rule).last()
                        rolling_corr = rolling_corr.resample(self.hf_resample_rule).last()
                    
                    # Ensure alignment with base price index
                    beta = beta.reindex(self.price_df.index, method='ffill')
                    beta_var = beta_var.reindex(self.price_df.index, method='ffill')
                    rolling_corr = rolling_corr.reindex(self.price_df.index, method='ffill')
                    
                    return beta, beta_var, rolling_corr

                beta_btc, var_btc, corr_btc = compute_beta(self.btc_ret_hf, self.returns_df_hf)
                beta_eth, var_eth, corr_eth = compute_beta(self.eth_ret_hf, self.returns_df_hf)
                beta_combined, var_combined, _ = compute_beta(combined_rets_hf, self.returns_df_hf)
                
            else:
                combined_rets = (self.btc_ret + self.eth_ret) / 2
                
                def compute_beta(market_ret):
                    # Beta
                    rolling_cov = self.returns_df.rolling(w).cov(market_ret)
                    rolling_mkt_var = market_ret.rolling(w).var()
                    beta = rolling_cov.div(rolling_mkt_var, axis=0)
                    
                    # Beta Variance
                    rolling_corr = self.returns_df.rolling(w).corr(market_ret)
                    rolling_coin_var = self.returns_df.rolling(w).var()
                    n_obs = self.returns_df.rolling(w).count()
                    
                    var_ratio = rolling_coin_var.div(rolling_mkt_var, axis=0)
                    unexplained_corr = 1 - rolling_corr**2
                    dof = n_obs - 2
                    
                    beta_var = var_ratio * unexplained_corr / dof
                    beta_var[dof <= 0] = np.nan
                    
                    return beta, beta_var, rolling_corr

                beta_btc, var_btc, corr_btc = compute_beta(self.btc_ret)
                beta_eth, var_eth, corr_eth = compute_beta(self.eth_ret)
                beta_combined, var_combined, _ = compute_beta(combined_rets)

            # Compute Adaptive Beta (Max Correlation)
            # Use abs correlation to determine best fit (Highest R^2)
            corr_btc_abs = corr_btc.abs().fillna(-1.0)
            corr_eth_abs = corr_eth.abs().fillna(-1.0)
            
            mask_btc_better = corr_btc_abs >= corr_eth_abs
            
            beta_adaptive = beta_btc.where(mask_btc_better, beta_eth)
            var_adaptive = var_btc.where(mask_btc_better, var_eth)

            self.betas[w] = {
                'btc': beta_btc,
                'eth': beta_eth,
                'combined': beta_combined,
                'adaptive': beta_adaptive
            }
            
            self.beta_vars[w] = {
                'btc': var_btc,
                'eth': var_eth,
                'combined': var_combined,
                'adaptive': var_adaptive
            }

            # Match original script logic: Explicitly exclude BTC and ETH from beta-neutral trading
            # by setting their betas to NaN in ALL beta types.
            for key in self.betas[w]:
                for ticker in ['BTCUSDT', 'ETHUSDT']:
                    if ticker in self.betas[w][key].columns:
                        self.betas[w][key][ticker] = np.nan
                        self.beta_vars[w][key][ticker] = np.nan


# --- Strategy Interface ---

class Strategy:
    def prepare(self, bundle: FundingDataBundle) -> None:
        pass

    def signals(self, idx: int, bundle: FundingDataBundle) -> Dict[str, Any]:
        raise NotImplementedError

@dataclass
class FundingArbParams:
    ar_window: int = 30
    forecast_horizon: int = 1
    portfolio_size_each_side: int = 5
    beta_limit: float = 1e-4
    gross_exposure_limit: float = 1.0
    tc_bps: float = 5.0
    use_ar_model: bool = True  # If False, maybe use last funding rate (naive)
    beta_type: str = "combined"  # "btc", "eth", "combined", or "adaptive"
    beta_window: int = 30        # Lookback for beta calculation
    min_positions: int = 0       # Minimum number of active positions
    min_weight: float = 0.037    # Minimum absolute weight per active position
    use_shrinkage: bool = False  # Whether to use beta shrinkage
    prior_beta_window: int = 60  # Lookback for prior beta (target)
    min_volume: float = 0.0      # Minimum 30-day rolling average daily dollar volume
    beta_neutral: bool = True    # Whether to enforce beta neutrality

class FundingArbStrategy(Strategy):
    """
    Forecasting strategy using AR(1) model on funding rates.
    """
    def __init__(self, params: FundingArbParams):
        self.params = params

    def prepare(self, bundle: FundingDataBundle) -> None:
        """
        Precompute AR(1) forecasts and Ensure Betas exist.
        """
        # 1. Ensure Betas
        windows = [self.params.beta_window]
        if self.params.use_shrinkage:
            windows.append(self.params.prior_beta_window)
        
        bundle.ensure_beta_matrix(windows)

        # 2. Precompute AR Forecasts
        w = self.params.ar_window
        if w in bundle.predicted_funding:
            return

        print(f"Precomputing AR(1) funding forecasts (window={w}) using AutoReg. This may take a while...")
        
        funding_data = bundle.funding_df
        preds_df = pd.DataFrame(np.nan, index=funding_data.index, columns=funding_data.columns)
        symbols = bundle.funding_df.columns
        
        total_syms = len(symbols)
        for i, sym in enumerate(symbols):
            if i % 10 == 0:
                print(f"Processing symbol {i}/{total_syms}...")
            
            series = funding_data[sym]
            # Get integer locations for the rolling window loop
            # We need at least w data points.
            
            # Optimization: Convert to numpy for faster slicing/checks
            vals = series.values
            dates = series.index
            
            # We iterate starting from index w-1 (which is the w-th element, allowing a window of size w)
            # Actually, to have 'w' elements ending at 't', we need t >= w-1.
            
            for t in range(w - 1, len(vals)):
                # Window: t-(w-1) to t (inclusive) -> length w
                # e.g. w=30. t=29. 29-29=0. 0 to 29. Length 30.
                window_vals = vals[t - w + 1 : t + 1]
                
                # 1. Handle NaNs / Infs
                # The original script does: funding_rate = funding_rate[np.isfinite(funding_rate)]
                # This changes the effective window size if we drop NaNs! 
                # But the original script used a fixed 30-day lookback *date range*.
                # If data was missing in that range, it would get fewer points.
                # If points < 30, it skips.
                
                # Here we are iterating by available bars. 
                # Let's filter valid values within the window.
                valid_vals = window_vals[np.isfinite(window_vals)]
                
                if len(valid_vals) < w:
                    continue
                    
                # 2. Check Variance
                if np.std(valid_vals) < 1e-10:
                    continue
                    
                # 3. Fit AutoReg
                try:
                    # lags=1 requires at least 1+1+constant points? AutoReg is usually robust.
                    model = AutoReg(valid_vals, lags=1, old_names=False).fit()
                    
                    if hasattr(model, 'params') and len(model.params) == 2:
                        intercept = model.params[0]
                        slope = model.params[1]
                        
                        # FIX: Use actual value at t (like standalone), not last valid value
                        # If vals[t] is NaN, prediction should be NaN (filtered out later)
                        current_val = vals[t]
                        if not np.isfinite(current_val):
                            continue
                        pred = intercept + slope * current_val
                        
                        # Store prediction at timestamp t (forecast made at t for t+1)
                        preds_df.iloc[t, i] = pred
                except Exception:
                    continue

        print(f"AR(1) precomputation complete for window {w}.")
        bundle.predicted_funding[w] = preds_df

    def signals(self, idx: int, bundle: FundingDataBundle) -> Dict[str, Any]:
        """
        Returns the alpha (predicted funding) and risk factors (betas) for the day.
        """
        date = bundle.dates[idx]
        w = self.params.ar_window
        
        # Get Predicted Funding
        if w in bundle.predicted_funding and date in bundle.predicted_funding[w].index:
            alpha = bundle.predicted_funding[w].loc[date].to_numpy()
        else:
            alpha = np.full(len(bundle.tickers), np.nan)

        # Get Betas
        beta_type = getattr(self.params, "beta_type", "combined").lower()
        window = self.params.beta_window
        
        # Check if window exists
        if window not in bundle.betas:
            # Should have been prepared, but safe fallback
            beta = np.full(len(bundle.tickers), np.nan)
        else:
            betas_for_window = bundle.betas[window]
            
            if beta_type not in betas_for_window:
                warnings.warn(f"Beta type '{beta_type}' not found for window {window}. Using 'combined' or first available.")
                beta_type = "combined" if "combined" in betas_for_window else list(betas_for_window.keys())[0]

            if date in betas_for_window[beta_type].index:
                raw_beta = betas_for_window[beta_type].loc[date].to_numpy()
                
                # Apply Shrinkage if enabled
                if self.params.use_shrinkage:
                    prior_window = self.params.prior_beta_window
                    
                    # Ensure prior data exists
                    if prior_window in bundle.betas and prior_window in bundle.beta_vars:
                        # Get Empirical Variance
                        if window in bundle.beta_vars and beta_type in bundle.beta_vars[window]:
                            beta_var = bundle.beta_vars[window][beta_type].loc[date].to_numpy()
                        else:
                            beta_var = np.full(len(bundle.tickers), np.nan) # Cannot shrink without var
                            
                        # Get Prior Beta and Variance
                        priors_for_window = bundle.betas[prior_window]
                        prior_vars_for_window = bundle.beta_vars[prior_window]
                        
                        # We use the same beta_type for prior (e.g. beta_60d vs beta_30d of same type)
                        # Summary says "Prior Belief... Metric: 60-Day Rolling Beta". 
                        # It doesn't imply a different type (like sector beta). 
                        # It says "Anchor... assuming beta is mean-reverting".
                        
                        if beta_type in priors_for_window and beta_type in prior_vars_for_window:
                            prior_beta = priors_for_window[beta_type].loc[date].to_numpy()
                            prior_var = prior_vars_for_window[beta_type].loc[date].to_numpy()
                            
                            # Apply Shrinkage
                            # shrink_beta_estimate handles scalars or arrays
                            shrunk_beta, _ = shrink_beta_estimate(raw_beta, beta_var, prior_beta, prior_var)
                            beta = shrunk_beta
                        else:
                             # Missing prior data
                             beta = raw_beta
                    else:
                        # Missing prior window
                        beta = raw_beta
                else:
                    beta = raw_beta
            else:
                beta = np.full(len(bundle.tickers), np.nan)
            
        return {
            "alpha": alpha, 
            "beta": beta,   
            "date": date
        }


# --- Weighting Model ---

class WeightingModel:
    def weights(
        self,
        idx: int,
        signals: Dict[str, Any],
        bundle: FundingDataBundle,
        universe_mask: np.ndarray,
        params: Any,
    ) -> np.ndarray:
        raise NotImplementedError

class BetaNeutralWeighting(WeightingModel):
    """
    Solves convex optimization problem:
    Maximize Expected Funding
    Subject to:
      - Beta Neutrality
      - Gross Exposure <= 1
      - Top/Bottom K selection
    """
    
    def weights(
        self,
        idx: int,
        signals: Dict[str, Any],
        bundle: FundingDataBundle,
        universe_mask: np.ndarray,
        params: FundingArbParams,
    ) -> np.ndarray:
        
        alpha = signals["alpha"] # predicted funding
        beta = signals["beta"]
        
        # Filter valid data - match standalone: use fillna(0) approach
        # Standalone does: s = predicted_funding_rates.loc[idx].fillna(0).values
        # So we should allow NaN alphas but treat them as 0
        mask = universe_mask & np.isfinite(beta)  # Only require valid beta
        available_indices = np.nonzero(mask)[0]
        
        if len(available_indices) < 2 * params.portfolio_size_each_side:
            return np.zeros(len(bundle.tickers))
        
        # Get values for available assets (fill NaN alpha with 0 like standalone)
        alpha_sub = np.nan_to_num(alpha[available_indices], nan=0.0)
            
        # Select Universe: Top K and Bottom K by predicted funding
        # MATCH STANDALONE: sort descending, take head (highest) and tail (lowest)
        
        # Sort descending (like standalone's sort_values(ascending=False))
        sorted_args_desc = np.argsort(-alpha_sub)  # Negative for descending
        
        k = params.portfolio_size_each_side
        
        # highest = head(k) in descending order = first k indices
        # lowest = tail(k) in descending order = last k indices
        top_k_local = sorted_args_desc[:k]      # Highest predicted funding
        bottom_k_local = sorted_args_desc[-k:]  # Lowest predicted funding
        
        # Combine selected local indices
        selected_local = np.union1d(top_k_local, bottom_k_local)
        
        # Map back to global indices
        trade_indices = available_indices[selected_local]
        
        if len(trade_indices) == 0:
            return np.zeros(len(bundle.tickers))
        
        # CRITICAL FIX: Sort trade_indices by SYMBOL NAME (alphabetically)
        # to match standalone's pandas Index.intersection() behavior which sorts alphabetically
        # Standalone: idx = predicted_funding_rates.index.intersection(betas.index)
        # pandas Index intersection returns sorted index
        trade_symbols = [bundle.tickers[j] for j in trade_indices]
        sorted_order = np.argsort(trade_symbols)  # Alphabetical sort
        trade_indices = trade_indices[sorted_order]
            
        # Optimization Data - use fillna(0) like standalone
        s_opt = np.nan_to_num(alpha[trade_indices], nan=0.0)  # Signal (predicted funding)
        b_opt = np.nan_to_num(beta[trade_indices], nan=0.0)   # Beta
        n_opt = len(trade_indices)
        
        # CVXPY Problem
        try:
            # Check if we need MIP (Minimum Positions or Minimum Weight)
            use_mip = (params.min_positions > 0) or (params.min_weight > 0)
            
            if use_mip:
                # If we need more positions than available candidates, return 0 early
                if n_opt < params.min_positions:
                     return np.zeros(len(bundle.tickers))

                # MIP Formulation
                # Split into positive and negative components to handle |w| constraints linearly
                w_pos = cvx.Variable(n_opt, nonneg=True)
                w_neg = cvx.Variable(n_opt, nonneg=True)
                z_pos = cvx.Variable(n_opt, boolean=True)
                z_neg = cvx.Variable(n_opt, boolean=True)
                
                w = w_pos - w_neg
                
                # Big-M: Max weight for any single asset is gross_exposure_limit
                M = params.gross_exposure_limit
                min_w = params.min_weight
                
                constraints = [
                    # Link continuous and binary (Semi-continuous constraints)
                    w_pos <= z_pos * M,
                    w_neg <= z_neg * M,
                    w_pos >= z_pos * min_w,
                    w_neg >= z_neg * min_w,
                    
                    # Mutually exclusive: An asset cannot be both long and short
                    z_pos + z_neg <= 1,
                    
                    # Minimum number of active positions
                    cvx.sum(z_pos + z_neg) >= params.min_positions,
                    
                    # Portfolio Constraints
                    cvx.sum(w_pos + w_neg) <= 1.0, # Gross Exposure hardcoded to 1.0
                ]
                
                if params.beta_neutral:
                     constraints.append(cvx.abs(b_opt @ w) <= params.beta_limit) # Beta Neutral
                
                objective = cvx.Maximize(-s_opt @ w)
                prob = cvx.Problem(objective, constraints)
                
                # Use SCIPY (Highs) solver for MIP
                prob.solve(solver=cvx.SCIPY, verbose=False)
                
            else:
                # Standard LP (Convex)
                w = cvx.Variable(n_opt)
                objective = cvx.Maximize(-s_opt @ w)
                
                constraints = [
                    cvx.norm1(w) <= 1.0, # Leverage limit hardcoded to 1.0
                    cvx.abs(w) <= params.gross_exposure_limit # Individual asset cap
                ]
                
                if params.beta_neutral:
                     constraints.append(cvx.abs(b_opt @ w) <= params.beta_limit) # Beta neutral
                
                prob = cvx.Problem(objective, constraints)
                prob.solve(solver=cvx.CLARABEL, verbose=False)
            
            if w.value is None:
                return np.zeros(len(bundle.tickers))
                
            w_res = w.value
            
            # Map back to full weight vector
            full_weights = np.zeros(len(bundle.tickers))
            full_weights[trade_indices] = w_res
            
            # Cleanup small weights (though MIP should handle min_weight strictly)
            full_weights[np.abs(full_weights) < EPS] = 0.0
            
            return full_weights
            
        except Exception as e:
            # print(f"Optimization failed at {idx}: {e}")
            return np.zeros(len(bundle.tickers))


# --- Backtest Engine ---

class FundingBacktestEngine:
    """
    Specialized engine that accounts for Funding Income.
    Total Return = Price Return + Funding Return
    """

    def __init__(
        self,
        bundle: FundingDataBundle,
        strategy: Strategy,
        weighting: WeightingModel,
        params: FundingArbParams,
    ):
        self.bundle = bundle
        self.strategy = strategy
        self.weighting = weighting
        self.params = params
        
        self.strategy.prepare(bundle)
        
        # Precompute volume eligibility mask
        self.volume_mask_df = None
        if self.params.min_volume > 0 and self.bundle.volume_df is not None:
            # We use a rolling mean of volume to check liquidity using 30-day window
            # "Daily Universe Mask": Rolling 30d Volume (Adjusted for freq) >= min_volume
            # Formula: (Rolling 30D Sum of Volume) / 30
            # This gives Average Daily Volume regardless of input frequency (Hourly/Daily)
            
            # Ensure volume_df index is datetime
            vol_df = self.bundle.volume_df.copy()
            if not isinstance(vol_df.index, pd.DatetimeIndex):
                try:
                    vol_df.index = pd.to_datetime(vol_df.index)
                except Exception:
                    warnings.warn("Volume DF index is not DatetimeIndex, cannot use time-based rolling. Skipping volume filter.")
                    self.volume_mask_df = None
            
            if self.volume_mask_df is None: # Only if not skipped above
                # Rolling 30D 
                # min_periods needs to be reasonable. For 30D, usually want at least some data.
                # If hourly (720 pts), min_periods=0 might trigger early. 
                # Let's say we need at least 1 day of data?
                
                avg_daily_vol = vol_df.rolling('30D', min_periods=1).sum() / 30.0
                
                # Align with price_df index (decision times)
                # reindex ensures we have a row for every backtest row
                avg_daily_vol = avg_daily_vol.reindex(self.bundle.price_df.index).fillna(0.0)
                
                self.volume_mask_df = (avg_daily_vol >= self.params.min_volume)

    def _universe_mask(self, idx: int, next_date: pd.Timestamp) -> np.ndarray:
        # Basic mask: Price exists, not nan
        # Checks Prices exist for current and next period, 
        # and Funding exists for next period.
        
        # Calendar-based: next_date is passed in (current_date + 1 day)
        
        p_curr = self.bundle.price_df.iloc[idx].to_numpy()
        
        # Check if next_date exists in price_df (calendar-based lookup)
        if next_date not in self.bundle.price_df.index:
            return np.zeros(len(self.bundle.tickers), dtype=bool)
            
        p_next = self.bundle.price_df.loc[next_date].to_numpy()
        
        has_price = np.isfinite(p_curr) & np.isfinite(p_next)
        
        # Check funding data availability (calendar-based)
        if next_date in self.bundle.funding_df.index:
            f_next = self.bundle.funding_df.loc[next_date].to_numpy()
            has_funding = np.isfinite(f_next)
        else:
            has_funding = np.zeros(len(self.bundle.tickers), dtype=bool)
            
        base_mask = has_price & has_funding.astype(bool)
        
        # Apply Volume Filter
        if self.volume_mask_df is not None:
             # idx corresponds to current row in price_df (and volume_mask_df)
             vol_mask = self.volume_mask_df.iloc[idx].to_numpy()
             base_mask = base_mask & vol_mask

        return base_mask

    def run(
        self,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
        
        b = self.bundle
        n_dates = len(b.dates)
        if end_idx is None or end_idx >= n_dates:
            end_idx = n_dates - 1
            
        equity_path = []
        return_path = []
        
        # New PnL tracking components
        price_pnl_path = []   # Daily price PnL
        funding_pnl_path = [] # Daily funding PnL
        
        turnover_path = []
        date_path = []
        position_records = []
        
        # Detailed per-symbol records for comparison with standalone
        detailed_records = []
        
        equity = 1.0
        prev_w = np.zeros(len(b.tickers))
        
        # print(f"Starting backtest from idx {start_idx} to {end_idx}...")
        
        for i in range(start_idx, end_idx):
            # Trading logic happens at 'i' (start of day/period)
            # Returns are realized at next calendar day (i + 1 day)
            
            current_date = pd.Timestamp(b.dates[i])
            next_date = current_date + pd.Timedelta(days=1)  # Calendar-based next day
            
            # Check if next_date data exists (calendar-based)
            price_exists = next_date in b.price_df.index
            funding_exists = next_date in b.funding_df.index
            
            if not price_exists or not funding_exists:
                # Missing data for next day -> Record flat performance (like standalone)
                equity_path.append(equity)
                return_path.append(0.0)
                price_pnl_path.append(0.0)
                funding_pnl_path.append(0.0)
                turnover_path.append(0.0)
                date_path.append(next_date)
                # Don't update prev_w - keep previous positions
                continue
            
            uni_mask = self._universe_mask(i, next_date)
            
            # Get Signals
            sigs = self.strategy.signals(i, b)
            
            # Get Weights
            weights = self.weighting.weights(i, sigs, b, uni_mask, self.params)
            
            # Calculate Turnover
            turnover = np.sum(np.abs(weights - prev_w))
            cost = turnover * (self.params.tc_bps / 10000.0)
            
            # Calculate Returns
            # Price Return = Sum(w * (p_next - p_curr)/p_curr)
            # Funding Return = Sum(-w * funding_rate_next)
            
            # Slice for speed
            active = (weights != 0)
            if not np.any(active):
                pnl_price = 0.0
                pnl_funding = 0.0
                daily_ret = 0.0 - cost
            else:
                w_active = weights[active]
                
                # Price Return (calendar-based lookup)
                p_curr = b.price_df.iloc[i].to_numpy()[active]
                p_next = b.price_df.loc[next_date].to_numpy()[active]
                r_price = (p_next - p_curr) / p_curr
                # Handle potential division by zero or nans if mask failed
                r_price = np.nan_to_num(r_price, 0.0)
                
                pnl_price = np.sum(w_active * r_price)
                
                # Funding Return (calendar-based lookup)
                f_next = b.funding_df.loc[next_date].to_numpy()[active]
                f_next = np.nan_to_num(f_next, 0.0)
                # Note: Funding is paid by longs (w>0) if rate>0 -> cost
                # Funding is received by shorts (w<0) if rate>0 -> gain
                # So PnL = -1 * weight * rate
                pnl_funding = np.sum(-w_active * f_next)
                
                daily_ret = pnl_price + pnl_funding - cost

                # Record positions
                long_idx = np.where(weights > 0)[0]
                short_idx = np.where(weights < 0)[0]
                position_records.append({
                    "date": next_date, # Date of return realization
                    "long_tickers": "|".join([b.tickers[j] for j in long_idx]),
                    "short_tickers": "|".join([b.tickers[j] for j in short_idx]),
                    "long_allocations": "|".join([f"{b.tickers[j]}:{weights[j]:.4f}" for j in long_idx]),
                    "short_allocations": "|".join([f"{b.tickers[j]}:{weights[j]:.4f}" for j in short_idx]),
                    "long_positions": len(long_idx),
                    "short_positions": len(short_idx),
                    "total_long": np.sum(weights[long_idx]),
                    "total_short": np.sum(weights[short_idx]),
                    "daily_return": daily_ret,
                    "price_return": pnl_price,
                    "funding_return": pnl_funding,
                    "turnover": turnover
                })
                
                # Detailed per-symbol records for comparison
                alpha = sigs["alpha"]
                beta_arr = sigs["beta"]
                f_next_full = b.funding_df.loc[next_date].to_numpy()
                p_curr_full = b.price_df.iloc[i].to_numpy()
                p_next_full = b.price_df.loc[next_date].to_numpy()
                
                for j in np.where(weights != 0)[0]:
                    sym = b.tickers[j]
                    w_j = weights[j]
                    price_chg = (p_next_full[j] - p_curr_full[j]) / p_curr_full[j] if p_curr_full[j] != 0 else 0.0
                    detailed_records.append({
                        'date': next_date,  # Return realization date
                        'trade_date': current_date,  # Decision date
                        'symbol': sym,
                        'weight': w_j,
                        'predicted_funding': alpha[j] if np.isfinite(alpha[j]) else np.nan,
                        'actual_funding': f_next_full[j] if np.isfinite(f_next_full[j]) else np.nan,
                        'price_change': price_chg,
                        'price_return_contrib': w_j * price_chg,
                        'funding_return_contrib': -w_j * (f_next_full[j] if np.isfinite(f_next_full[j]) else 0.0),
                        'beta': beta_arr[j] if np.isfinite(beta_arr[j]) else np.nan,
                    })

            equity *= (1.0 + daily_ret)
            
            equity_path.append(equity)
            return_path.append(daily_ret)
            
            price_pnl_path.append(pnl_price)
            funding_pnl_path.append(pnl_funding)
            
            turnover_path.append(turnover)
            date_path.append(next_date)
            
            prev_w = weights
            
        # Compile Results
        idx = pd.to_datetime(date_path)
        eq_s = pd.Series(equity_path, index=idx, name="equity")
        ret_s = pd.Series(return_path, index=idx, name="return")
        price_ret_s = pd.Series(price_pnl_path, index=idx, name="price_return")
        funding_ret_s = pd.Series(funding_pnl_path, index=idx, name="funding_return")
        
        turn_s = pd.Series(turnover_path, index=idx, name="turnover")
        pos_df = pd.DataFrame(position_records)
        if not pos_df.empty:
            pos_df['date'] = pd.to_datetime(pos_df['date'])
        
        # Compile detailed records
        detailed_df = pd.DataFrame(detailed_records)
        if not detailed_df.empty:
            detailed_df['date'] = pd.to_datetime(detailed_df['date'])
            detailed_df['trade_date'] = pd.to_datetime(detailed_df['trade_date'])
        
        return eq_s, ret_s, price_ret_s, funding_ret_s, turn_s, pos_df, detailed_df


# --- Walk Forward Runner (Adapted) ---

class FundingWalkForwardRunner:
    """
    Simplified Walk-Forward Runner for Funding Strategies.
    """
    
    def __init__(
        self,
        bundle: FundingDataBundle,
        params_grid: List[FundingArbParams],
        train_span: int,
        test_span: int,
        step_span: int,
        score_mode: str = "sharpe",
        mode: str = "expanding",
        periods_per_year: float = PERIODS_PER_YEAR,
    ):
        self.bundle = bundle
        self.params_grid = params_grid
        self.train_span = train_span
        self.test_span = test_span
        self.step_span = step_span
        self.score_mode = score_mode
        self.mode = mode.lower()
        self.periods_per_year = periods_per_year
        
        # We use a single strategy class
        self.strategy_cls = FundingArbStrategy
        self.weighting_cls = BetaNeutralWeighting
        
    def run(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
        all_dates = self.bundle.dates
        total = len(all_dates)
        
        results = []
        oos_returns_list = []
        # New lists for components
        oos_price_returns_list = []
        oos_funding_returns_list = []
        
        oos_dates_list = []
        all_positions_list = []
        all_detailed_records_list = []  # For detailed per-symbol records
        
        current_end = self.train_span
        iteration = 0
        
        while current_end < total:
            iteration += 1
            
            # Determine Window Type
            if self.mode == "rolling":
                train_start = max(0, current_end - self.train_span)
            else:
                train_start = 0 # Expanding
            
            train_end = current_end
            test_start = current_end
            test_end_inclusive = min(current_end + self.test_span, total) - 1 

            if test_end_inclusive < test_start:
                break
                
            print(f"Iteration {iteration} ({self.mode.title()}): Train [{train_start}:{train_end}], Test [{test_start}:{test_end_inclusive}]")
            
            # 1. Optimize Params (Train)
            best_score = -np.inf
            best_params = None
            best_sharpe = np.nan
            
            # Use joblib here if we want parallelization, but linear for now
            for params in self.params_grid:
                # Instantiate
                strat = self.strategy_cls(params)
                weight = self.weighting_cls()
                engine = FundingBacktestEngine(self.bundle, strat, weight, params)
                
                # Run Train
                _, ret, _, _, _, _, _ = engine.run(start_idx=train_start, end_idx=train_end)
                
                # Score
                score = select_score(ret, mode=self.score_mode, periods_per_year=self.periods_per_year)
                sharpe = compute_sharpe(ret, periods_per_year=self.periods_per_year)
                
                if np.isfinite(score) and score > best_score:
                    best_score = score
                    best_params = params
                    best_sharpe = sharpe
            
            if best_params is None:
                best_params = self.params_grid[0]
                
            # 2. Run OOS (Test)
            strat = self.strategy_cls(best_params)
            weight = self.weighting_cls()
            engine = FundingBacktestEngine(self.bundle, strat, weight, best_params)
            
            _, ret_oos, price_ret_oos, funding_ret_oos, _, pos_oos, detailed_oos = engine.run(start_idx=test_start, end_idx=test_end_inclusive)
            
            if not ret_oos.empty:
                oos_returns_list.extend(ret_oos.values)
                oos_price_returns_list.extend(price_ret_oos.values)
                oos_funding_returns_list.extend(funding_ret_oos.values)
                
                oos_dates_list.extend(ret_oos.index)
                
                oos_score = select_score(ret_oos, mode=self.score_mode, periods_per_year=self.periods_per_year)
                oos_sharpe = compute_sharpe(ret_oos, periods_per_year=self.periods_per_year)
                
                if not pos_oos.empty:
                    pos_oos = pos_oos.copy()
                    pos_oos['iteration'] = iteration
                    all_positions_list.append(pos_oos)
                    
                if not detailed_oos.empty:
                    detailed_oos = detailed_oos.copy()
                    detailed_oos['iteration'] = iteration
                    all_detailed_records_list.append(detailed_oos)
            else:
                oos_score = np.nan
                oos_sharpe = np.nan
                
            results.append({
                "iteration": iteration,
                "train_start": all_dates[train_start],
                "train_end": all_dates[train_end],
                "test_start": all_dates[test_start],
                "test_end": all_dates[test_end_inclusive], 
                "best_params": best_params,
                "is_score": best_score,
                "is_sharpe": best_sharpe,
                "oos_score": oos_score,
                "oos_sharpe": oos_sharpe
            })
            
            current_end += self.step_span
            
        # Aggregate
        oos_s = pd.Series(oos_returns_list, index=oos_dates_list).sort_index()
        oos_s = oos_s[~oos_s.index.duplicated(keep='first')]
        oos_equity = (1 + oos_s).cumprod()
        
        # Aggregate Price and Funding components
        oos_price_s = pd.Series(oos_price_returns_list, index=oos_dates_list).sort_index()
        oos_price_s = oos_price_s[~oos_price_s.index.duplicated(keep='first')]
        
        oos_funding_s = pd.Series(oos_funding_returns_list, index=oos_dates_list).sort_index()
        oos_funding_s = oos_funding_s[~oos_funding_s.index.duplicated(keep='first')]
        
        wf_df = pd.DataFrame(results)
        positions_df = pd.concat(all_positions_list, ignore_index=True) if all_positions_list else pd.DataFrame()
        detailed_df = pd.concat(all_detailed_records_list, ignore_index=True) if all_detailed_records_list else pd.DataFrame()
        
        # Return extended tuple with detailed records
        return wf_df, oos_s, oos_equity, oos_price_s, oos_funding_s, positions_df, detailed_df

    def report(
        self,
        wf_df: pd.DataFrame,
        oos_returns: pd.Series,
        oos_equity: pd.Series,
        oos_price_returns: Optional[pd.Series] = None,
        oos_funding_returns: Optional[pd.Series] = None,
        plot: bool = True,
        fig_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generates a comprehensive performance report, mirroring the backtest_framework structure
        but adapted for Funding Arbitrage (includes PnL decomposition).
        """
        per_year = self.periods_per_year
        fig_path = Path(fig_dir) if fig_dir else None
        if fig_path:
            fig_path.mkdir(parents=True, exist_ok=True)

        out: Dict[str, Any] = {}

        # 1. Aggregate OOS performance
        if oos_returns is not None and not oos_returns.empty:
            agg_sharpe = compute_sharpe(oos_returns, periods_per_year=per_year)
            agg_sortino = compute_sortino_ratio(oos_returns, periods_per_year=per_year)
            agg_calmar = compute_calmar_ratio(oos_returns, oos_equity, periods_per_year=per_year)
            agg_comp = compute_composite_score(oos_returns, oos_equity, periods_per_year=per_year)
            agg_score = select_score(oos_returns, oos_equity, mode=self.score_mode, periods_per_year=per_year)
            agg_total_ret = oos_equity.iloc[-1] - 1
            agg_cagr = (oos_equity.iloc[-1] ** (per_year / len(oos_equity)) - 1) if len(oos_equity) else float("nan")
            
            drawdown = (oos_equity / oos_equity.cummax() - 1.0)
            max_dd = drawdown.min()

            # Additive PnL (Component comparison)
            if oos_price_returns is not None and oos_funding_returns is not None:
                total_pnl_additive = oos_returns.cumsum()
                price_pnl_additive = oos_price_returns.cumsum()
                funding_pnl_additive = oos_funding_returns.cumsum()
            else:
                total_pnl_additive = pd.Series(dtype=float)
                price_pnl_additive = pd.Series(dtype=float)
                funding_pnl_additive = pd.Series(dtype=float)

            print("=" * 80)
            print("AGGREGATED OUT-OF-SAMPLE PERFORMANCE (Funding Arb)")
            print("=" * 80)
            print(f"Mode: {self.mode.upper()}")
            print(f"Total OOS Bars: {len(oos_equity)}")
            if not oos_equity.empty:
                print(f"Date Range: {oos_equity.index[0].date()} to {oos_equity.index[-1].date()}")
            print("--- Risk-Adjusted ---")
            print(f"Selected Score ({self.score_mode}): {agg_score:.3f}")
            print(f"Sharpe:   {agg_sharpe:.3f}")
            print(f"Sortino:  {agg_sortino:.3f}")
            print(f"Calmar:   {agg_calmar:.3f}")
            print(f"Composite:{agg_comp:.3f}")
            print("--- Absolute ---")
            print(f"Total Return: {agg_total_ret*100:.2f}%")
            print(f"CAGR:         {agg_cagr*100:.2f}%")
            print(f"Max Drawdown: {max_dd*100:.2f}%")
            
            out.update({
                "agg_sharpe": agg_sharpe,
                "agg_sortino": agg_sortino,
                "agg_calmar": agg_calmar,
                "agg_composite": agg_comp,
                "agg_score": agg_score,
                "agg_total_return": agg_total_ret,
                "agg_cagr": agg_cagr,
                "agg_max_dd": max_dd,
                "combined_equity": oos_equity,
                "combined_drawdown": drawdown,
                "total_pnl_additive": total_pnl_additive,
                "price_pnl_additive": price_pnl_additive,
                "funding_pnl_additive": funding_pnl_additive,
            })
        else:
            print("No OOS returns available")

        # 2. Parameter selection summary
        if wf_df is not None and not wf_df.empty:
            best_params_df = wf_df.copy()
            # Extract relevant FundingArbParams
            best_params_df["best_ar_window"] = best_params_df["best_params"].apply(lambda p: getattr(p, "ar_window", None))
            best_params_df["best_beta_window"] = best_params_df["best_params"].apply(lambda p: getattr(p, "beta_window", None))
            best_params_df["best_portfolio_size"] = best_params_df["best_params"].apply(lambda p: getattr(p, "portfolio_size_each_side", None))
            best_params_df["best_beta_type"] = best_params_df["best_params"].apply(lambda p: getattr(p, "beta_type", None))
            best_params_df["best_min_positions"] = best_params_df["best_params"].apply(lambda p: getattr(p, "min_positions", None))
            best_params_df["best_min_weight"] = best_params_df["best_params"].apply(lambda p: getattr(p, "min_weight", None))
            best_params_df["best_use_shrinkage"] = best_params_df["best_params"].apply(lambda p: getattr(p, "use_shrinkage", None))
            best_params_df["best_prior_beta_window"] = best_params_df["best_params"].apply(lambda p: getattr(p, "prior_beta_window", None))
            best_params_df["best_gross_exposure"] = best_params_df["best_params"].apply(lambda p: getattr(p, "gross_exposure_limit", None))
            best_params_df["best_beta_neutral"] = best_params_df["best_params"].apply(lambda p: getattr(p, "beta_neutral", None))
            # best_params_df["best_use_ar"] = best_params_df["best_params"].apply(lambda p: getattr(p, "use_ar_model", None))

            print("\nParameter Selection Counts:")
            print("AR window selection:\n", best_params_df["best_ar_window"].value_counts().sort_index())
            print("Beta window selection:\n", best_params_df["best_beta_window"].value_counts().sort_index())
            print("Portfolio size selection:\n", best_params_df["best_portfolio_size"].value_counts().sort_index())
            print("Beta type selection:\n", best_params_df["best_beta_type"].value_counts().sort_index())
            print("Min positions selection:\n", best_params_df["best_min_positions"].value_counts().sort_index())
            print("Min weight selection:\n", best_params_df["best_min_weight"].value_counts().sort_index())
            print("Use shrinkage selection:\n", best_params_df["best_use_shrinkage"].value_counts().sort_index())
            print("Prior beta window selection:\n", best_params_df["best_prior_beta_window"].value_counts().sort_index())
            print("Gross exposure limit selection:\n", best_params_df["best_gross_exposure"].value_counts().sort_index())
            print("Beta neutral selection:\n", best_params_df["best_beta_neutral"].value_counts().sort_index())

            out["best_params_df"] = best_params_df
        else:
            print("wf_df is empty; no parameter summary available.")

        # 3. Recompute per-iteration OOS (to capture turnover and verify consistency, mirroring backtest_framework)
        iter_stats: List[Dict[str, Any]] = []
        combined_returns: List[float] = []
        combined_dates: List[pd.Timestamp] = []
        combined_turnover: List[float] = []
        
        # Funding specific re-computes
        combined_price_returns: List[float] = []
        combined_funding_returns: List[float] = []

        dates_idx = pd.Index(self.bundle.dates)
        
        if wf_df is not None and not wf_df.empty:
            print("\\nRecomputing OOS metrics for turnover tracking...")
            for row in wf_df.itertuples(index=False):
                params = row.best_params
                
                # Re-instantiate
                strat = self.strategy_cls(params)
                weight = self.weighting_cls()
                engine = FundingBacktestEngine(self.bundle, strat, weight, params)
                
                test_start = pd.Timestamp(row.test_start)
                test_end = pd.Timestamp(row.test_end)
                
                # Check for dates availability
                # Note: run() takes date indices.
                if test_start not in dates_idx or test_end not in dates_idx:
                   continue
                
                # We need to find the integer indices for start and end of this OOS block
                # Engine.run uses integer start_idx and end_idx.
                # It simulates from start_idx to end_idx.
                # However, backtest_framework optimizes this by running from 0 to test_end_idx to get proper history?
                # Actually, backtest_framework does: engine.run(start_idx=0, end_idx=test_end_idx)
                # This is safer to ensure all lookbacks (AR model, Vol, etc) are correct.
                
                test_start_idx = dates_idx.get_loc(test_start)
                test_end_idx = dates_idx.get_loc(test_end)

                # Run OOS for this iteration (full run up to test_end to ensure warmup)
                _, ret_full, price_ret_full, funding_ret_full, turn_full, _, _ = engine.run(start_idx=0, end_idx=test_end_idx)
                
                # Slice logic same as backtest_framework
                # ret_full is indexed by DATE.
                
                # We want only the [test_start, test_end] portion.
                # Using loc on DatetimeIndex is inclusive.
                oos_slice = slice(test_start, test_end)
                
                oos_ret = ret_full.loc[oos_slice].dropna()
                oos_price_ret = price_ret_full.loc[oos_slice]
                oos_funding_ret = funding_ret_full.loc[oos_slice]
                # Ensure turnover aligns with returns
                oos_turn = turn_full.reindex(oos_ret.index).fillna(0)
                
                if oos_ret.empty:
                    continue
                    
                oos_eq = (1 + oos_ret).cumprod()

                combined_returns.extend(oos_ret.values)
                combined_price_returns.extend(oos_price_ret.values)
                combined_funding_returns.extend(oos_funding_ret.values)
                combined_dates.extend(oos_ret.index)
                combined_turnover.extend(oos_turn.values)

                iter_stats.append(
                    dict(
                        iteration=row.iteration,
                        test_start=test_start,
                        test_end=test_end,
                        ar_window=getattr(params, "ar_window", None),
                        beta_window=getattr(params, "beta_window", None),
                        beta_type=getattr(params, "beta_type", None),
                        oos_sharpe=compute_sharpe(oos_ret, periods_per_year=per_year),
                        oos_sortino=compute_sortino_ratio(oos_ret, periods_per_year=per_year),
                        oos_calmar=compute_calmar_ratio(oos_ret, oos_eq, periods_per_year=per_year),
                        oos_total_return=oos_eq.iloc[-1] - 1,
                        oos_avg_turnover=oos_turn.mean(),
                    )
                )

        combined_returns_series = pd.Series(combined_returns, index=combined_dates, name="OOS_Returns_Recomputed").sort_index()
        combined_returns_series = combined_returns_series[~combined_returns_series.index.duplicated(keep="first")]
        
        combined_turnover_series = pd.Series(combined_turnover, index=combined_returns_series.index, name="OOS_Turnover") if not combined_returns_series.empty else pd.Series(dtype=float)
        
        iter_stats_df = pd.DataFrame(iter_stats)

        out.update(
            dict(
                combined_returns_recomputed=combined_returns_series,
                combined_turnover=combined_turnover_series,
                iter_stats=iter_stats_df,
            )
        )

        # 4. Plots
        if plot:
            try:
                import matplotlib.pyplot as plt
                plt.style.use("seaborn-v0_8")
                
                # A. Equity & Drawdown (with Component breakdown)
                if not oos_equity.empty:
                    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True, gridspec_kw={"height_ratios": [2, 1, 2]})
                    
                    # 1. Equity
                    axes[0].plot(oos_equity.index, oos_equity.values, label="Total Equity", color="tab:blue", linewidth=1.5)
                    
                    # Add BTC Benchmark if available
                    if self.bundle.btc_ret is not None:
                        # Construct BTC Equity for OOS period
                         btc_ret_oos = self.bundle.btc_ret.reindex(oos_equity.index).fillna(0)
                         btc_equity = (1 + btc_ret_oos).cumprod()
                         axes[0].plot(btc_equity.index, btc_equity.values, label="BTC Buy & Hold", color="tab:gray", linestyle="--", alpha=0.6)

                    axes[0].set_ylabel("Equity")
                    axes[0].set_title("Walk-Forward Equity Curve (Funding Arb)")
                    axes[0].legend()
                    axes[0].grid(True, alpha=0.3)
                    
                    # 2. Drawdown
                    dd_pct = (oos_equity / oos_equity.cummax() - 1.0) * 100
                    axes[1].plot(dd_pct.index, dd_pct.values, label="Drawdown %", color="tab:red")
                    axes[1].fill_between(dd_pct.index, dd_pct.values, 0, color="tab:red", alpha=0.2)
                    axes[1].set_ylabel("DD (%)")
                    axes[1].legend()
                    axes[1].grid(True, alpha=0.3)

                    # 3. Additive Decomposition
                    if "total_pnl_additive" in out and not out["total_pnl_additive"].empty:
                        t_pnl = out["total_pnl_additive"]
                        p_pnl = out["price_pnl_additive"]
                        f_pnl = out["funding_pnl_additive"]
                        
                        axes[2].plot(t_pnl.index, t_pnl.values, label="Total PnL (Additive)", color="tab:gray", linestyle="--", alpha=0.8)
                        axes[2].plot(p_pnl.index, p_pnl.values, label="Price Component PnL", color="tab:blue", alpha=0.8)
                        axes[2].plot(f_pnl.index, f_pnl.values, label="Funding Component PnL", color="tab:orange", alpha=0.8)
                        
                        axes[2].set_title("PnL Attribution (Cumulative %)")
                        axes[2].set_xlabel("Date")
                        axes[2].legend()
                        axes[2].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    if fig_path:
                        fig.savefig(fig_path / "oos_performance_summary.png", dpi=150)
                        plt.close(fig)
                    else:
                        plt.show()

                # B. Parameter Selection
                if "best_params_df" in out:
                    bp = out["best_params_df"]
                    fig, axes = plt.subplots(5, 2, figsize=(12, 20))
                    plots = [
                        ("best_ar_window", "AR Window"),
                        ("best_beta_window", "Beta Window"),
                        ("best_portfolio_size", "Portfolio Size"),
                        ("best_beta_type", "Beta Type"),
                        ("best_min_positions", "Min Positions"),
                        ("best_min_weight", "Min Weight"),
                        ("best_use_shrinkage", "Use Shrinkage"),
                        ("best_prior_beta_window", "Prior Beta Window"),
                        ("best_gross_exposure", "Gross Exposure Limit"),
                        ("best_beta_neutral", "Beta Neutral"),
                    ]
                    
                    flat_axes = axes.flatten()
                    for i, ax in enumerate(flat_axes):
                        if i < len(plots):
                            col, title = plots[i]
                            if col in bp.columns:
                                bp[col].value_counts().sort_index().plot(kind="bar", ax=ax, color="tab:blue")
                                ax.set_title(title)
                                ax.grid(alpha=0.3)
                        else:
                            ax.axis('off')
                            
                    plt.tight_layout()
                    if fig_path:
                        fig.savefig(fig_path / "parameter_counts.png", dpi=150)
                        plt.close(fig)
                    else:
                        plt.show()

                # C. IS vs OOS Sharpe
                if wf_df is not None and not wf_df.empty:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(wf_df["iteration"], wf_df["is_sharpe"], label="IS Sharpe", marker="o", linestyle="-", alpha=0.7)
                    ax.plot(wf_df["iteration"], wf_df["oos_sharpe"], label="OOS Sharpe", marker="o", linestyle="-", alpha=0.7)
                    ax.axhline(0, color="black", linestyle="--", alpha=0.3)
                    ax.set_xlabel("Iteration")
                    ax.set_ylabel("Sharpe Ratio")
                    ax.set_title("In-Sample vs Out-of-Sample Sharpe")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    if fig_path:
                        fig.savefig(fig_path / "is_oos_sharpe.png", dpi=150)
                        plt.close(fig)
                    else:
                        plt.show()

                # D. Turnover
                if not combined_turnover_series.empty:
                    fig, axes = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={"height_ratios": [2, 1]})
                    axes[0].plot(combined_turnover_series.index, combined_turnover_series.values * 100, color="purple", linewidth=0.8)
                    axes[0].set_title("Daily Turnover (%)")
                    axes[0].set_ylabel("Turnover (%)")
                    axes[0].grid(alpha=0.3)
                    axes[1].hist(combined_turnover_series.values * 100, bins=40, color="purple", alpha=0.7)
                    axes[1].set_xlabel("Turnover (%)")
                    axes[1].set_ylabel("Frequency")
                    plt.tight_layout()
                    if fig_path:
                        fig.savefig(fig_path / "turnover.png", dpi=150)
                        plt.close(fig)
                    else:
                        plt.show()

            except ImportError:
                print("matplotlib not available; skipping plots.")
        
        return out

