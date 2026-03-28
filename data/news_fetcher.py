import urllib.parse
from datetime import datetime, timedelta

import feedparser
import pandas as pd
import streamlit as st

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_news_mentions(keywords: list[str], timeframe: str) -> pd.DataFrame:
    """Fetch article counts from Google News RSS for each keyword.

    Returns a DataFrame with datetime index and one column per keyword
    containing article counts per time bucket.
    """
    if not keywords:
        return pd.DataFrame()

    resample_rule = _get_resample_rule(timeframe)
    lookback = _get_lookback(timeframe)
    cutoff = datetime.now() - lookback

    all_series = {}
    for kw in keywords:
        dates = _fetch_article_dates(kw)
        dates = [d for d in dates if d >= cutoff]
        if not dates:
            continue
        s = pd.Series(1, index=pd.DatetimeIndex(dates), name=kw)
        counts = s.resample(resample_rule).sum().fillna(0)
        all_series[kw] = counts

    if not all_series:
        return pd.DataFrame()

    df = pd.DataFrame(all_series).fillna(0).astype(int)
    return df


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_news_headlines(keywords: list[str], max_per_keyword: int = 20) -> list[dict]:
    """Fetch recent news headlines matching keywords.

    Returns list of dicts with: title, source, published, keyword, link.
    """
    if not keywords:
        return []

    headlines = []
    for kw in keywords:
        query = urllib.parse.quote_plus(kw)
        url = GOOGLE_NEWS_RSS.format(query=query)
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_per_keyword]:
                pub_date = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    pub_date = datetime(*entry.published_parsed[:6])

                source = ""
                if hasattr(entry, "source") and hasattr(entry.source, "title"):
                    source = entry.source.title
                elif " - " in entry.get("title", ""):
                    source = entry.title.rsplit(" - ", 1)[-1]

                headlines.append({
                    "title": entry.get("title", "").rsplit(" - ", 1)[0] if " - " in entry.get("title", "") else entry.get("title", ""),
                    "source": source,
                    "published": pub_date,
                    "keyword": kw,
                    "link": entry.get("link", ""),
                })
        except Exception:
            continue

    headlines.sort(key=lambda x: x["published"] or datetime.min, reverse=True)
    return headlines


def _fetch_article_dates(keyword: str) -> list[datetime]:
    """Parse Google News RSS feed and extract publication dates."""
    query = urllib.parse.quote_plus(keyword)
    url = GOOGLE_NEWS_RSS.format(query=query)
    try:
        feed = feedparser.parse(url)
        dates = []
        for entry in feed.entries:
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                dt = datetime(*entry.published_parsed[:6])
                dates.append(dt)
        return dates
    except Exception:
        return []


def _get_resample_rule(timeframe: str) -> str:
    return {
        "1H": "5min",
        "1D": "1h",
        "1M": "1D",
        "1Q": "1W",
        "1Y": "1W",
        "2Y": "1ME",
        "5Y": "1ME",
        "10Y": "1ME",
        "All": "1ME",
    }.get(timeframe, "1D")


def _get_lookback(timeframe: str) -> timedelta:
    return {
        "1H": timedelta(hours=1),
        "1D": timedelta(days=1),
        "1M": timedelta(days=30),
        "1Q": timedelta(days=90),
        "1Y": timedelta(days=365),
        "2Y": timedelta(days=730),
        "5Y": timedelta(days=1825),
        "10Y": timedelta(days=3650),
        "All": timedelta(days=3650),
    }.get(timeframe, timedelta(days=30))
