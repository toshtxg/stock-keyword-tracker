# Stock & Keyword Mention Correlation Tracker

Track stock prices alongside keyword mentions (Google Trends + Google News) and analyze their statistical correlation.

## Features

- **Stock Price Charts** — Line, candlestick, or area charts with volume bars, SMA (20/50), and Bollinger Bands
- **Keyword Mention Tracking** — Google Trends search interest + Google News article counts
- **Correlation Analysis** — Dual-axis overlay, Pearson/Spearman correlation, scatter plots, rolling correlation, and lag analysis
- **News Headlines** — Browse recent headlines matching your keywords
- **Data Export** — Download stock and mention data as CSV
- **9 Timeframes** — 1H, 1D, 1M, 1Q, 1Y, 2Y, 5Y, 10Y, All

## Run Locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy Free on Streamlit Community Cloud

1. Push to a public GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Select your repo, branch `main`, file `app.py`
4. Click Deploy

No API keys required — all data sources are free and keyless.

## Data Sources

| Source | What it provides | Limits |
|---|---|---|
| Yahoo Finance (yfinance) | Stock OHLCV data | No key needed |
| Google Trends (pytrends) | Search interest 0-100 | No key, cached 1hr |
| Google News RSS | Article counts + headlines | No key, cached 30min |
