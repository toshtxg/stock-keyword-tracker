import pandas as pd
import numpy as np
from scipy import stats

from analysis.normalization import min_max_normalize


def align_series(
    stock_prices: pd.Series,
    mention_scores: pd.Series,
    resample_rule: str = "1D",
) -> tuple[pd.Series, pd.Series]:
    """Align stock price and mention score to the same datetime index."""
    stock = stock_prices.resample(resample_rule).last().ffill()
    mentions = mention_scores.resample(resample_rule).mean().interpolate(
        method="linear", limit_direction="both"
    )

    common_idx = stock.index.intersection(mentions.index)
    if common_idx.empty:
        combined = pd.DataFrame({"stock": stock, "mentions": mentions})
        combined = combined.interpolate(method="linear", limit_direction="both").dropna()
        return combined["stock"], combined["mentions"]

    return stock.loc[common_idx], mentions.loc[common_idx]


def get_aligned_returns(
    stock_prices: pd.Series,
    mention_scores: pd.Series,
    resample_rule: str = "1D",
) -> tuple[pd.Series, pd.Series]:
    """Get aligned returns/changes series, cleaned of inf/nan, for scatter plots."""
    stock_aligned, mentions_aligned = align_series(
        stock_prices, mention_scores, resample_rule
    )

    stock_returns = stock_aligned.pct_change().dropna()
    mention_changes = mentions_aligned.pct_change().dropna()

    common = stock_returns.index.intersection(mention_changes.index)
    if len(common) < 3:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    sr = stock_returns.loc[common].replace([np.inf, -np.inf], np.nan).dropna()
    mc = mention_changes.loc[common].replace([np.inf, -np.inf], np.nan).dropna()
    common_clean = sr.index.intersection(mc.index)

    return sr.loc[common_clean], mc.loc[common_clean]


def compute_correlation(
    stock_prices: pd.Series,
    mention_scores: pd.Series,
    resample_rule: str = "1D",
) -> dict:
    """Compute Pearson and Spearman correlation between stock returns and mention changes."""
    sr, mc = get_aligned_returns(stock_prices, mention_scores, resample_rule)

    if len(sr) < 5:
        return {
            "pearson_r": None, "pearson_p": None,
            "spearman_r": None, "spearman_p": None,
            "n_samples": len(sr),
        }

    pearson_r, pearson_p = stats.pearsonr(sr, mc)
    spearman_r, spearman_p = stats.spearmanr(sr, mc)

    return {
        "pearson_r": round(pearson_r, 4),
        "pearson_p": round(pearson_p, 4),
        "spearman_r": round(spearman_r, 4),
        "spearman_p": round(spearman_p, 4),
        "n_samples": len(sr),
    }


def compute_lag_correlation(
    stock_prices: pd.Series,
    mention_scores: pd.Series,
    resample_rule: str = "1D",
    max_lag: int = 7,
) -> pd.DataFrame:
    """Compute Pearson correlation at different lags."""
    stock_aligned, mentions_aligned = align_series(
        stock_prices, mention_scores, resample_rule
    )

    if len(stock_aligned) < 10:
        return pd.DataFrame(columns=["lag", "pearson_r", "pearson_p"])

    stock_returns = stock_aligned.pct_change().dropna()

    results = []
    for lag in range(-max_lag, max_lag + 1):
        shifted_mentions = mentions_aligned.shift(lag).pct_change().dropna()
        common = stock_returns.index.intersection(shifted_mentions.index)

        if len(common) < 5:
            results.append({"lag": lag, "pearson_r": 0, "pearson_p": 1.0})
            continue

        sr = stock_returns.loc[common].replace([np.inf, -np.inf], np.nan)
        mc = shifted_mentions.loc[common].replace([np.inf, -np.inf], np.nan)
        valid = sr.notna() & mc.notna()

        if valid.sum() < 5:
            results.append({"lag": lag, "pearson_r": 0, "pearson_p": 1.0})
            continue

        r, p = stats.pearsonr(sr[valid], mc[valid])
        results.append({"lag": lag, "pearson_r": round(r, 4), "pearson_p": round(p, 4)})

    return pd.DataFrame(results)


def get_resample_rule(timeframe: str) -> str:
    return {
        "1H": "5min", "1D": "1h", "1M": "1D", "1Q": "1D",
        "1Y": "1W", "2Y": "1W", "5Y": "1ME", "10Y": "1ME", "All": "1ME",
    }.get(timeframe, "1D")


def get_max_lag(timeframe: str) -> int:
    return {
        "1H": 5, "1D": 5, "1M": 7, "1Q": 7,
        "1Y": 7, "2Y": 5, "5Y": 4, "10Y": 3, "All": 3,
    }.get(timeframe, 7)


def get_rolling_window(timeframe: str) -> int:
    """Get appropriate rolling window for rolling correlation."""
    return {
        "1H": 10, "1D": 10, "1M": 10, "1Q": 15,
        "1Y": 12, "2Y": 12, "5Y": 8, "10Y": 6, "All": 6,
    }.get(timeframe, 12)
