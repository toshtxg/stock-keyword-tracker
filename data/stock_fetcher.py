import yfinance as yf
import pandas as pd
import streamlit as st

TIMEFRAME_MAP = {
    "1H": {"period": "1d", "interval": "1m"},
    "1D": {"period": "5d", "interval": "5m"},
    "1M": {"period": "1mo", "interval": "1d"},
    "1Q": {"period": "3mo", "interval": "1d"},
    "1Y": {"period": "1y", "interval": "1wk"},
    "2Y": {"period": "2y", "interval": "1wk"},
    "5Y": {"period": "5y", "interval": "1mo"},
    "10Y": {"period": "10y", "interval": "1mo"},
    "All": {"period": "max", "interval": "1mo"},
}

INTRADAY_TIMEFRAMES = {"1H", "1D"}


@st.cache_data(ttl=60, show_spinner=False)
def _fetch_intraday(ticker: str, period: str, interval: str) -> pd.DataFrame:
    return _download(ticker, period, interval)


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_daily(ticker: str, period: str, interval: str) -> pd.DataFrame:
    return _download(ticker, period, interval)


def _download(ticker: str, period: str, interval: str) -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    df = tk.history(period=period, interval=interval)
    if df.empty:
        return df
    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
    return df


def fetch_stock(ticker: str, timeframe: str) -> pd.DataFrame:
    params = TIMEFRAME_MAP[timeframe]
    if timeframe in INTRADAY_TIMEFRAMES:
        return _fetch_intraday(ticker, **params)
    return _fetch_daily(ticker, **params)


def get_ticker_info(ticker: str) -> dict:
    """Get basic info about a ticker: name, sector, market cap, etc."""
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        return {
            "name": info.get("shortName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap"),
            "currency": info.get("currency", "USD"),
            "exchange": info.get("exchange", "N/A"),
            "description": info.get("longBusinessSummary", ""),
        }
    except Exception:
        return {"name": ticker, "sector": "N/A", "industry": "N/A",
                "market_cap": None, "currency": "USD", "exchange": "N/A",
                "description": ""}


def validate_ticker(ticker: str) -> bool:
    try:
        tk = yf.Ticker(ticker)
        info = tk.fast_info
        return info is not None and hasattr(info, "last_price")
    except Exception:
        return False
