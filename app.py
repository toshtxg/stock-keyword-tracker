import streamlit as st
import pandas as pd

from data.stock_fetcher import fetch_stock, validate_ticker, get_ticker_info
from data.trends_fetcher import fetch_trends
from data.news_fetcher import fetch_news_mentions, fetch_news_headlines
from data.aggregator import aggregate_mentions, get_combined_mentions, RESAMPLE_MAP
from analysis.correlation import (
    compute_correlation,
    compute_lag_correlation,
    get_aligned_returns,
    get_resample_rule,
    get_max_lag,
    get_rolling_window,
)
from visualization.charts import (
    create_stock_chart,
    create_mentions_chart,
    create_correlation_chart,
    create_scatter_plot,
    create_rolling_correlation_chart,
    create_lag_chart,
    format_market_cap,
)

# --- Page Config ---
st.set_page_config(
    page_title="Stock & Keyword Tracker",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
<style>
    /* Tighter main content padding */
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }

    /* Metric card styling */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2d3436;
        border-radius: 10px;
        padding: 12px 16px;
    }
    [data-testid="stMetricLabel"] { font-size: 0.8rem; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px;
        border-radius: 8px 8px 0 0;
    }

    /* Sidebar section headers */
    .sidebar-header {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #636e72;
        margin-top: 1rem;
        margin-bottom: 0.3rem;
    }

    /* News headline cards */
    .news-card {
        background: #1a1a2e;
        border-left: 3px solid #ffa502;
        padding: 8px 12px;
        margin-bottom: 6px;
        border-radius: 0 6px 6px 0;
        font-size: 0.85rem;
    }
    .news-source {
        color: #636e72;
        font-size: 0.72rem;
    }

    /* Welcome hero */
    .hero {
        text-align: center;
        padding: 3rem 1rem;
    }
    .hero h1 { font-size: 2.5rem; margin-bottom: 0.5rem; }
    .hero p { color: #b2bec3; font-size: 1.1rem; }

    /* Correlation badge */
    .corr-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .corr-strong-pos { background: rgba(0, 212, 170, 0.2); color: #00d4aa; }
    .corr-moderate-pos { background: rgba(0, 212, 170, 0.1); color: #00d4aa; }
    .corr-weak { background: rgba(99, 110, 114, 0.2); color: #b2bec3; }
    .corr-moderate-neg { background: rgba(255, 71, 87, 0.1); color: #ff4757; }
    .corr-strong-neg { background: rgba(255, 71, 87, 0.2); color: #ff4757; }
</style>
""", unsafe_allow_html=True)

# --- Session State ---
if "ticker" not in st.session_state:
    st.session_state.ticker = ""
if "keywords" not in st.session_state:
    st.session_state.keywords = []
if "ticker_info" not in st.session_state:
    st.session_state.ticker_info = {}

# --- Sidebar ---
with st.sidebar:
    st.markdown("## Stock & Keyword Tracker")
    st.markdown("---")

    # Ticker
    st.markdown('<p class="sidebar-header">Stock Ticker</p>', unsafe_allow_html=True)
    ticker_input = st.text_input(
        "Ticker symbol",
        placeholder="e.g. AAPL, TSLA, MSFT",
        value=st.session_state.ticker,
        label_visibility="collapsed",
    ).upper().strip()

    if ticker_input and ticker_input != st.session_state.ticker:
        with st.spinner("Validating..."):
            if validate_ticker(ticker_input):
                st.session_state.ticker = ticker_input
                st.session_state.ticker_info = get_ticker_info(ticker_input)
                st.success(f"Tracking {ticker_input}")
            else:
                st.error(f"Invalid ticker: {ticker_input}")

    # Show ticker info if available
    if st.session_state.ticker_info.get("name"):
        info = st.session_state.ticker_info
        st.caption(f"**{info['name']}** — {info.get('sector', 'N/A')}")

    st.markdown("---")

    # Keywords
    st.markdown('<p class="sidebar-header">Keywords to Track</p>', unsafe_allow_html=True)
    keyword_input = st.text_input(
        "Keywords",
        placeholder="e.g. iPhone, Apple AI",
        label_visibility="collapsed",
    )
    if st.button("Add Keywords", width="stretch") and keyword_input:
        new_keywords = [k.strip() for k in keyword_input.split(",") if k.strip()]
        for kw in new_keywords:
            if kw not in st.session_state.keywords:
                st.session_state.keywords.append(kw)
        st.rerun()

    if st.session_state.keywords:
        for i, kw in enumerate(st.session_state.keywords):
            col1, col2 = st.columns([4, 1])
            col1.markdown(f"`{kw}`")
            if col2.button("✕", key=f"rm_{i}"):
                st.session_state.keywords.pop(i)
                st.rerun()

    st.markdown("---")

    # Timeframe
    st.markdown('<p class="sidebar-header">Timeframe</p>', unsafe_allow_html=True)
    timeframe = st.select_slider(
        "Timeframe",
        options=["1H", "1D", "1M", "1Q", "1Y", "2Y", "5Y", "10Y", "All"],
        value="1Y",
        label_visibility="collapsed",
    )

    # Chart options
    st.markdown('<p class="sidebar-header">Chart Options</p>', unsafe_allow_html=True)
    chart_type = st.selectbox(
        "Chart type", ["line", "candlestick", "area"],
        label_visibility="collapsed",
    )
    col_a, col_b = st.columns(2)
    show_volume = col_a.checkbox("Volume", value=True)
    show_sma = col_b.checkbox("SMA", value=True)
    show_bollinger = st.checkbox("Bollinger Bands", value=False)

    # Sources
    st.markdown('<p class="sidebar-header">Mention Sources</p>', unsafe_allow_html=True)
    sources = st.multiselect(
        "Platforms",
        ["Google Trends", "Google News"],
        default=["Google Trends", "Google News"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    if st.button("Refresh All Data", width="stretch", type="primary"):
        st.cache_data.clear()
        st.rerun()


# --- Welcome Screen ---
if not st.session_state.ticker:
    st.markdown("""
    <div class="hero">
        <h1>Stock & Keyword Correlation Tracker</h1>
        <p>Track stock prices alongside keyword mentions to discover patterns</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 1. Add a Stock")
        st.markdown("Enter any ticker symbol (AAPL, TSLA, GOOGL...) in the sidebar to start tracking its price across multiple timeframes.")
    with col2:
        st.markdown("### 2. Add Keywords")
        st.markdown("Add keywords you want to track. The app monitors Google Trends search interest and Google News article counts.")
    with col3:
        st.markdown("### 3. See Correlation")
        st.markdown("Overlay stock price with keyword mentions to visually and statistically analyze their relationship.")

    st.markdown("---")
    st.caption("All data sourced from free APIs. No API keys required. Deploy free on Streamlit Community Cloud.")
    st.stop()

# --- Data Fetching ---
ticker = st.session_state.ticker
keywords = st.session_state.keywords

with st.spinner(f"Fetching {ticker} data..."):
    stock_df = fetch_stock(ticker, timeframe)

if stock_df.empty:
    st.error(f"No data found for {ticker} with timeframe {timeframe}.")
    st.stop()

# Mention data
trends_df = pd.DataFrame()
news_df = pd.DataFrame()
headlines = []

if keywords:
    with st.spinner("Fetching keyword mention data..."):
        if "Google Trends" in sources:
            trends_df = fetch_trends(keywords, timeframe)
        if "Google News" in sources:
            news_df = fetch_news_mentions(keywords, timeframe)
            headlines = fetch_news_headlines(keywords, max_per_keyword=15)

    resample_rule = RESAMPLE_MAP.get(timeframe, "1D")
    mention_df = aggregate_mentions(trends_df, news_df, resample_rule)
else:
    mention_df = pd.DataFrame()

# --- Header ---
info = st.session_state.ticker_info
current_price = stock_df["Close"].iloc[-1]
prev_price = stock_df["Close"].iloc[0]
change = current_price - prev_price
change_pct = (change / prev_price) * 100

header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.markdown(f"## {info.get('name', ticker)} ({ticker})")
    if info.get("sector") != "N/A":
        st.caption(f"{info.get('sector', '')} · {info.get('industry', '')} · {info.get('exchange', '')}")
with header_col2:
    delta_color = "normal"
    st.metric(
        "Current Price",
        f"${current_price:.2f}",
        f"{change_pct:+.2f}% ({timeframe})",
    )

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Stock Price", "🔍 Keyword Mentions", "🔗 Correlation"])

# ==================== Tab 1: Stock Price ====================
with tab1:
    fig = create_stock_chart(
        stock_df, ticker,
        chart_type=chart_type,
        show_volume=show_volume,
        show_sma=show_sma,
        show_bollinger=show_bollinger,
    )
    st.plotly_chart(fig, width="stretch")

    # Summary metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Open", f"${stock_df['Open'].iloc[-1]:.2f}")
    col2.metric("High", f"${stock_df['High'].max():.2f}")
    col3.metric("Low", f"${stock_df['Low'].min():.2f}")
    col4.metric("Volume", f"{stock_df['Volume'].iloc[-1]:,.0f}" if "Volume" in stock_df else "N/A")
    col5.metric("Mkt Cap", format_market_cap(info.get("market_cap")))

    # Data export
    with st.expander("Raw Data & Export"):
        st.dataframe(
            stock_df[["Open", "High", "Low", "Close", "Volume"]].tail(50),
            width="stretch",
        )
        csv = stock_df[["Open", "High", "Low", "Close", "Volume"]].to_csv()
        st.download_button(
            "Download CSV",
            csv,
            file_name=f"{ticker}_{timeframe}_stock_data.csv",
            mime="text/csv",
        )

# ==================== Tab 2: Keyword Mentions ====================
with tab2:
    if not keywords:
        st.info("Add keywords in the sidebar to track mentions across Google Trends and Google News.")
    elif mention_df.empty:
        st.warning("No mention data found for the selected keywords and timeframe. Try different keywords or a different timeframe.")
    else:
        fig = create_mentions_chart(mention_df, keywords)
        st.plotly_chart(fig, width="stretch")

        # Summary stats
        st.subheader("Mention Statistics")
        stats_cols = st.columns(min(len(keywords), 4))
        for i, kw in enumerate(keywords):
            if kw in mention_df.columns:
                col = stats_cols[i % len(stats_cols)]
                series = mention_df[kw]
                col.metric(
                    kw,
                    f"{series.iloc[-1]:.1f}",
                    f"{series.iloc[-1] - series.mean():+.1f} vs avg",
                )

        # Platform breakdown
        left_col, right_col = st.columns(2)
        with left_col:
            if not trends_df.empty:
                with st.expander("Google Trends Data", expanded=False):
                    st.dataframe(trends_df.tail(20), width="stretch")
        with right_col:
            if not news_df.empty:
                with st.expander("Google News Article Counts", expanded=False):
                    st.dataframe(news_df.tail(20), width="stretch")

        # News headlines
        if headlines:
            st.subheader("Recent Headlines")
            for h in headlines[:15]:
                pub = h["published"].strftime("%b %d, %H:%M") if h["published"] else "Unknown"
                st.markdown(
                    f'<div class="news-card">'
                    f'<a href="{h["link"]}" target="_blank" style="color: #dfe6e9; text-decoration: none;">{h["title"]}</a>'
                    f'<br><span class="news-source">{h["source"]} · {pub} · keyword: {h["keyword"]}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # Mention data export
        with st.expander("Export Mention Data"):
            csv = mention_df.to_csv()
            st.download_button(
                "Download Mentions CSV",
                csv,
                file_name=f"mentions_{timeframe}.csv",
                mime="text/csv",
            )

# ==================== Tab 3: Correlation ====================
with tab3:
    if not keywords:
        st.info("Add keywords in the sidebar to analyze correlation with stock price movements.")
    elif mention_df.empty:
        st.warning("No mention data available for correlation analysis.")
    else:
        combined_mentions = get_combined_mentions(mention_df)
        stock_close = stock_df["Close"]
        keyword_label = " + ".join(keywords)
        resample_rule = get_resample_rule(timeframe)

        # Compute correlation
        corr_results = compute_correlation(stock_close, combined_mentions, resample_rule)

        # Dual-axis overlay chart
        fig = create_correlation_chart(
            stock_close, combined_mentions, ticker, keyword_label,
            correlation_value=corr_results.get("pearson_r"),
        )
        st.plotly_chart(fig, width="stretch")

        if corr_results["pearson_r"] is not None:
            # Correlation metrics with visual badges
            st.subheader("Correlation Metrics")

            r = corr_results["pearson_r"]
            abs_r = abs(r)
            if abs_r >= 0.7:
                strength, badge_class = "Strong", "corr-strong-pos" if r > 0 else "corr-strong-neg"
            elif abs_r >= 0.3:
                strength, badge_class = "Moderate", "corr-moderate-pos" if r > 0 else "corr-moderate-neg"
            else:
                strength, badge_class = "Weak", "corr-weak"

            direction = "Positive" if r > 0 else "Negative"
            sig = "Significant" if corr_results["pearson_p"] < 0.05 else "Not Significant"
            sig_icon = "✓" if corr_results["pearson_p"] < 0.05 else "✗"

            st.markdown(
                f'<span class="corr-badge {badge_class}">'
                f'{strength} {direction} Correlation · {sig_icon} {sig} (p={corr_results["pearson_p"]:.4f})'
                f'</span>',
                unsafe_allow_html=True,
            )
            st.markdown("")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Pearson r", f"{corr_results['pearson_r']:.4f}")
            col2.metric("Spearman ρ", f"{corr_results['spearman_r']:.4f}")
            col3.metric("p-value", f"{corr_results['pearson_p']:.4f}")
            col4.metric("Data Points", f"{corr_results['n_samples']}")

            # Two-column layout for scatter + rolling
            st.markdown("---")
            scatter_col, rolling_col = st.columns(2)

            # Scatter plot
            with scatter_col:
                st.subheader("Return vs Mention Scatter")
                stock_ret, mention_chg = get_aligned_returns(
                    stock_close, combined_mentions, resample_rule
                )
                if len(stock_ret) > 3:
                    fig = create_scatter_plot(stock_ret, mention_chg, ticker, keyword_label)
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info("Not enough data points for scatter plot.")

            # Rolling correlation
            with rolling_col:
                st.subheader("Rolling Correlation")
                window = get_rolling_window(timeframe)
                fig = create_rolling_correlation_chart(
                    stock_close, combined_mentions, window=window
                )
                if fig:
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info("Not enough data for rolling correlation.")

            # Lag analysis
            st.markdown("---")
            st.subheader("Lag Analysis")
            st.caption(
                "Do keyword mentions **lead** or **follow** price changes? "
                "Positive lag = mentions happened before price moved."
            )
            max_lag = get_max_lag(timeframe)
            lag_df = compute_lag_correlation(
                stock_close, combined_mentions, resample_rule, max_lag
            )
            if not lag_df.empty:
                fig = create_lag_chart(lag_df)
                st.plotly_chart(fig, width="stretch")

                best_lag = lag_df.loc[lag_df["pearson_r"].abs().idxmax()]
                lag_val = int(best_lag["lag"])
                if lag_val > 0:
                    lag_meaning = f"mentions tend to **lead** price by ~{lag_val} periods"
                elif lag_val < 0:
                    lag_meaning = f"price tends to **lead** mentions by ~{abs(lag_val)} periods"
                else:
                    lag_meaning = "mentions and price move **simultaneously**"

                st.markdown(
                    f"**Strongest correlation at lag {lag_val}** "
                    f"(r = {best_lag['pearson_r']:.4f}) — {lag_meaning}"
                )
        else:
            st.warning(
                f"Not enough aligned data points ({corr_results['n_samples']}) "
                f"to compute correlation. Try a longer timeframe or different keywords."
            )

        st.divider()
        st.caption(
            "Correlation does not imply causation. These metrics show statistical "
            "association between keyword mentions and stock price movements, not causal relationships. "
            "Many confounding factors may drive both variables independently."
        )

# --- Footer ---
st.markdown("---")
st.caption(
    "Data: Yahoo Finance · Google Trends · Google News RSS | "
    "All APIs are free, no keys required | "
    "Built with Streamlit"
)
