import pandas as pd
import streamlit as st

# Patch urllib3 Retry to accept the deprecated method_whitelist kwarg
# (pytrends 4.9.2 uses it, but urllib3 2.x removed it)
import urllib3.util.retry as _retry_module

_OrigRetry = _retry_module.Retry
class _PatchedRetry(_OrigRetry):
    def __init__(self, *args, **kwargs):
        if "method_whitelist" in kwargs:
            kwargs["allowed_methods"] = kwargs.pop("method_whitelist")
        super().__init__(*args, **kwargs)

_retry_module.Retry = _PatchedRetry

from pytrends.request import TrendReq

TRENDS_TIMEFRAME_MAP = {
    "1H": "now 1-H",
    "1D": "now 1-d",
    "1M": "today 1-m",
    "1Q": "today 3-m",
    "1Y": "today 12-m",
    "2Y": "today 5-y",   # pytrends doesn't support 2y directly; use 5y and trim
    "5Y": "today 5-y",
    "10Y": "all",
    "All": "all",
}

_TRIM_2Y = {"2Y"}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_trends(keywords: list[str], timeframe: str) -> pd.DataFrame:
    """Fetch Google Trends interest over time for given keywords.

    Returns a DataFrame with datetime index and one column per keyword (0-100 scale).
    """
    if not keywords:
        return pd.DataFrame()

    tf = TRENDS_TIMEFRAME_MAP.get(timeframe, "today 12-m")

    try:
        pytrends = TrendReq(hl="en-US", tz=360, retries=3, backoff_factor=1)
        batch_size = 5
        frames = []
        for i in range(0, len(keywords), batch_size):
            batch = keywords[i : i + batch_size]
            pytrends.build_payload(batch, timeframe=tf, geo="", gprop="")
            df = pytrends.interest_over_time()
            if not df.empty and "isPartial" in df.columns:
                df = df.drop(columns=["isPartial"])
            frames.append(df)

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, axis=1)
        result = result.loc[:, ~result.columns.duplicated()]

        if timeframe in _TRIM_2Y and not result.empty:
            cutoff = result.index.max() - pd.DateOffset(years=2)
            result = result[result.index >= cutoff]

        result.index = result.index.tz_localize(None) if result.index.tz is not None else result.index
        return result

    except Exception as e:
        st.warning(f"Google Trends error: {e}")
        return pd.DataFrame()
