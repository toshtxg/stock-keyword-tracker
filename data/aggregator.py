import pandas as pd
import numpy as np


def aggregate_mentions(
    trends_df: pd.DataFrame,
    news_df: pd.DataFrame,
    resample_rule: str | None = None,
) -> pd.DataFrame:
    """Combine Google Trends and News data into a single composite mention score.

    Each source is min-max normalized to 0-100, then averaged across sources.
    Returns a DataFrame with columns for each keyword's composite score.
    """
    if trends_df.empty and news_df.empty:
        return pd.DataFrame()

    keywords = set()
    if not trends_df.empty:
        keywords.update(trends_df.columns)
    if not news_df.empty:
        keywords.update(news_df.columns)

    result = {}
    for kw in keywords:
        sources = []

        if not trends_df.empty and kw in trends_df.columns:
            s = trends_df[kw].copy()
            if resample_rule:
                s = s.resample(resample_rule).mean()
            sources.append(_min_max_normalize(s))

        if not news_df.empty and kw in news_df.columns:
            s = news_df[kw].copy().astype(float)
            if resample_rule:
                s = s.resample(resample_rule).sum()
            sources.append(_min_max_normalize(s))

        if sources:
            combined = pd.concat(sources, axis=1).mean(axis=1)
            result[kw] = combined

    if not result:
        return pd.DataFrame()

    df = pd.DataFrame(result)
    df = df.interpolate(method="linear", limit_direction="both")
    return df


def get_combined_mentions(mention_df: pd.DataFrame) -> pd.Series:
    """Sum all keyword columns into a single combined mention series."""
    if mention_df.empty:
        return pd.Series(dtype=float)
    return mention_df.sum(axis=1)


def _min_max_normalize(s: pd.Series) -> pd.Series:
    min_val = s.min()
    max_val = s.max()
    if max_val == min_val:
        return pd.Series(50.0, index=s.index)
    return (s - min_val) / (max_val - min_val) * 100


RESAMPLE_MAP = {
    "1H": "5min",
    "1D": "1h",
    "1M": "1D",
    "1Q": "1W",
    "1Y": "1W",
    "2Y": "1ME",
    "5Y": "1ME",
    "10Y": "1ME",
    "All": "1ME",
}
