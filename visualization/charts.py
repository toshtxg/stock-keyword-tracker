import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis.normalization import min_max_normalize

# Consistent color palette
COLORS = {
    "stock": "#00d4aa",
    "stock_up": "#00d4aa",
    "stock_down": "#ff4757",
    "mention": "#ffa502",
    "volume": "rgba(100, 140, 200, 0.3)",
    "sma20": "#ff6348",
    "sma50": "#7bed9f",
    "bollinger": "rgba(128, 128, 128, 0.15)",
    "keywords": ["#ffa502", "#ff6348", "#7bed9f", "#70a1ff", "#a29bfe", "#fd79a8"],
    "positive": "#00d4aa",
    "negative": "#ff4757",
    "neutral": "#636e72",
}


def create_stock_chart(
    df: pd.DataFrame,
    ticker: str,
    chart_type: str = "line",
    show_volume: bool = True,
    show_sma: bool = True,
    show_bollinger: bool = False,
) -> go.Figure:
    """Create a stock price chart with optional indicators."""
    row_heights = [0.75, 0.25] if show_volume else [1.0]
    rows = 2 if show_volume else 1

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
    )

    # Main price chart
    if chart_type == "candlestick" and all(c in df.columns for c in ["Open", "High", "Low", "Close"]):
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name=ticker,
                increasing_line_color=COLORS["stock_up"],
                decreasing_line_color=COLORS["stock_down"],
            ),
            row=1, col=1,
        )
    elif chart_type == "area":
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Close"],
                mode="lines",
                name=ticker,
                line=dict(color=COLORS["stock"], width=2),
                fill="tozeroy",
                fillcolor="rgba(0, 212, 170, 0.1)",
            ),
            row=1, col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Close"],
                mode="lines",
                name=ticker,
                line=dict(color=COLORS["stock"], width=2),
            ),
            row=1, col=1,
        )

    # Moving averages
    if show_sma and len(df) >= 20:
        sma20 = df["Close"].rolling(window=20).mean()
        fig.add_trace(
            go.Scatter(
                x=df.index, y=sma20,
                mode="lines", name="SMA 20",
                line=dict(color=COLORS["sma20"], width=1, dash="dot"),
                opacity=0.7,
            ),
            row=1, col=1,
        )
        if len(df) >= 50:
            sma50 = df["Close"].rolling(window=50).mean()
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=sma50,
                    mode="lines", name="SMA 50",
                    line=dict(color=COLORS["sma50"], width=1, dash="dot"),
                    opacity=0.7,
                ),
                row=1, col=1,
            )

    # Bollinger Bands
    if show_bollinger and len(df) >= 20:
        sma = df["Close"].rolling(window=20).mean()
        std = df["Close"].rolling(window=20).std()
        upper = sma + 2 * std
        lower = sma - 2 * std

        fig.add_trace(
            go.Scatter(
                x=df.index, y=upper, mode="lines",
                name="BB Upper", line=dict(width=0),
                showlegend=False,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=lower, mode="lines",
                name="BB Lower", line=dict(width=0),
                fill="tonexty", fillcolor=COLORS["bollinger"],
                showlegend=False,
            ),
            row=1, col=1,
        )

    # Volume bars
    if show_volume and "Volume" in df.columns:
        vol_colors = [
            COLORS["stock_up"] if close >= open_ else COLORS["stock_down"]
            for close, open_ in zip(df["Close"], df["Open"])
        ]
        fig.add_trace(
            go.Bar(
                x=df.index, y=df["Volume"],
                name="Volume",
                marker_color=vol_colors,
                opacity=0.4,
                showlegend=False,
            ),
            row=2, col=1,
        )
        fig.update_yaxes(title_text="Volume", row=2, col=1)

    fig.update_layout(
        title=f"{ticker} Stock Price",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        hovermode="x unified",
        height=550 if show_volume else 450,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_mentions_chart(
    mention_df: pd.DataFrame,
    keywords: list[str],
) -> go.Figure:
    """Create a mention trends line chart with area fills for all keywords."""
    fig = go.Figure()

    for i, kw in enumerate(keywords):
        if kw in mention_df.columns:
            color = COLORS["keywords"][i % len(COLORS["keywords"])]
            fig.add_trace(
                go.Scatter(
                    x=mention_df.index,
                    y=mention_df[kw],
                    mode="lines",
                    name=kw,
                    line=dict(color=color, width=2),
                    fill="tozeroy",
                    fillcolor=_hex_to_rgba(color, 0.1),
                )
            )

    fig.update_layout(
        title="Keyword Mention Trends",
        xaxis_title="Date",
        yaxis_title="Mention Score (0-100)",
        template="plotly_dark",
        hovermode="x unified",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_correlation_chart(
    stock_prices: pd.Series,
    mention_scores: pd.Series,
    ticker: str,
    keyword_label: str,
    correlation_value: float | None = None,
) -> go.Figure:
    """Create a dual-axis overlay chart: stock price + mention score."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Stock price on left axis
    fig.add_trace(
        go.Scatter(
            x=stock_prices.index,
            y=stock_prices,
            mode="lines",
            name=f"{ticker} Price",
            line=dict(color=COLORS["stock"], width=2),
        ),
        secondary_y=False,
    )

    # Mention score on right axis (normalized to 0-100)
    normalized_mentions = min_max_normalize(mention_scores)
    fig.add_trace(
        go.Scatter(
            x=normalized_mentions.index,
            y=normalized_mentions,
            mode="lines",
            name=f'Mentions: "{keyword_label}"',
            line=dict(color=COLORS["mention"], width=2),
            fill="tozeroy",
            fillcolor="rgba(255, 165, 2, 0.08)",
            opacity=0.85,
        ),
        secondary_y=True,
    )

    title = f"{ticker} Price vs Keyword Mentions"
    if correlation_value is not None:
        title += f"  (r = {correlation_value:.3f})"

    fig.update_layout(
        title=title,
        template="plotly_dark",
        hovermode="x unified",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Price (USD)", secondary_y=False)
    fig.update_yaxes(title_text="Mention Score (0-100)", secondary_y=True)

    return fig


def create_scatter_plot(
    stock_returns: pd.Series,
    mention_changes: pd.Series,
    ticker: str,
    keyword_label: str,
) -> go.Figure:
    """Create a scatter plot of stock returns vs mention changes."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=mention_changes,
            y=stock_returns,
            mode="markers",
            marker=dict(
                color=COLORS["stock"],
                size=6,
                opacity=0.6,
                line=dict(width=0.5, color="white"),
            ),
            name="Data Points",
        )
    )

    # Add trend line
    valid = np.isfinite(stock_returns) & np.isfinite(mention_changes)
    if valid.sum() > 2:
        x_clean = mention_changes[valid].values
        y_clean = stock_returns[valid].values
        z = np.polyfit(x_clean, y_clean, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
        fig.add_trace(
            go.Scatter(
                x=x_line, y=p(x_line),
                mode="lines",
                name="Trend Line",
                line=dict(color=COLORS["mention"], width=2, dash="dash"),
            )
        )

    fig.update_layout(
        title=f"{ticker} Returns vs Mention Changes",
        xaxis_title=f'Mention Change ("{keyword_label}")',
        yaxis_title=f"{ticker} Return",
        template="plotly_dark",
        height=400,
    )
    return fig


def create_rolling_correlation_chart(
    stock_prices: pd.Series,
    mention_scores: pd.Series,
    window: int = 20,
) -> go.Figure:
    """Create a rolling correlation chart over time."""
    stock_ret = stock_prices.pct_change()
    mention_chg = mention_scores.pct_change()

    combined = pd.DataFrame({"stock": stock_ret, "mentions": mention_chg}).dropna()
    if len(combined) < window:
        return None

    rolling_corr = combined["stock"].rolling(window=window).corr(combined["mentions"])

    fig = go.Figure()

    # Background shading for positive/negative correlation
    fig.add_hrect(y0=0, y1=1, fillcolor=COLORS["positive"], opacity=0.05, line_width=0)
    fig.add_hrect(y0=-1, y1=0, fillcolor=COLORS["negative"], opacity=0.05, line_width=0)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.add_trace(
        go.Scatter(
            x=rolling_corr.index,
            y=rolling_corr,
            mode="lines",
            name=f"Rolling Correlation ({window}-period)",
            line=dict(color=COLORS["stock"], width=2),
        )
    )

    fig.update_layout(
        title=f"Rolling {window}-Period Correlation",
        xaxis_title="Date",
        yaxis_title="Pearson r",
        yaxis_range=[-1, 1],
        template="plotly_dark",
        height=350,
    )
    return fig


def create_lag_chart(lag_df: pd.DataFrame) -> go.Figure:
    """Create a bar chart showing correlation at different lags."""
    colors = [
        COLORS["positive"] if r > 0 else COLORS["negative"]
        for r in lag_df["pearson_r"]
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=lag_df["lag"],
            y=lag_df["pearson_r"],
            marker_color=colors,
            marker_line_color="white",
            marker_line_width=0.5,
            hovertemplate=(
                "Lag: %{x}<br>"
                "Correlation: %{y:.4f}<br>"
                "<extra></extra>"
            ),
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title="Lag Correlation Analysis",
        xaxis_title="Lag (positive = mentions lead price)",
        yaxis_title="Pearson r",
        template="plotly_dark",
        height=350,
    )
    return fig


def _hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    """Convert a hex color like '#ffa502' to 'rgba(255, 165, 2, alpha)'."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def format_market_cap(cap: int | None) -> str:
    """Format market cap into human-readable string."""
    if cap is None:
        return "N/A"
    if cap >= 1e12:
        return f"${cap / 1e12:.2f}T"
    if cap >= 1e9:
        return f"${cap / 1e9:.2f}B"
    if cap >= 1e6:
        return f"${cap / 1e6:.2f}M"
    return f"${cap:,.0f}"
