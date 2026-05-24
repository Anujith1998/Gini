import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from textblob import TextBlob
import requests
import plotly.graph_objects as go
from datetime import timedelta
import xml.etree.ElementTree as ET

# --- Configuration ---
st.set_page_config(
    page_title="ProQuant AI",
    layout="centered",
    page_icon="📈"
)
st.title("ProQuant AI 📈")
st.write("Advanced Market Analysis Engine")

# --- Token Handling ---
hf_token = st.secrets.get("HF_TOKEN", None)

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Controls")

if not hf_token:
    hf_token = st.sidebar.text_input(
        "HF Token (Optional)",
        type="password"
    )

st.sidebar.markdown("---")
st.sidebar.header("📝 Watchlist")
wl_input = st.sidebar.text_input(
    "Tickers:",
    value="AAPL, NVDA, TSLA, AMD, MSFT"
)

parsed_wl = []
for t in wl_input.split(","):
    if t.strip():
        parsed_wl.append(t.strip().upper())
watchlist_tuple = tuple(parsed_wl)

st.sidebar.markdown("---")
st.sidebar.header("🚨 Alerts")
alert_threshold = st.sidebar.number_input(
    "Alert Trigger (%)",
    min_value=1.0,
    max_value=50.0,
    value=4.0
)

max_price_filter = st.sidebar.slider(
    "Max Price ($)",
    min_value=10,
    max_value=1000,
    value=1000
)

# --- Functions ---
def resolve_company_name(query):
    cleaned = query.strip()
    if len(cleaned) <= 5 and cleaned.isalpha() and cleaned.isupper():
        return cleaned

    url = "https://query1.finance.yahoo.com/v1/finance/search"
    headers = {"User-Agent": "Mozilla/5.0"}
    params = {"q": cleaned, "lang": "en-US", "region": "US"}

    try:
        res = requests.get(url, params=params, headers=headers, timeout=5)
        data = res.json()
        if data.get("quotes"):
            return data["quotes"][0]["symbol"].upper()
    except:
        pass

    return cleaned.upper()

def get_resilient_news(stock_obj, ticker_str):
    news_data = []
    try:
        news_data = stock_obj.news
    except:
        pass

    if not news_data:
        try:
            url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker_str}"
            headers = {"User-Agent": "Mozilla/5.0"}
            res = requests.get(url, headers=headers, timeout=5)
            if res.status_code == 200:
                root = ET.fromstring(res.content)
                for item in root.findall(".//item"):
                    title = item.find("title")
                    if title is not None and title.text:
                        news_data.append({"title": title.text})
        except:
            pass

    return news_data if news_data else []

def query_finbert_api(text_list, token):
    url = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        res = requests.post(url, headers=headers, json={"inputs": text_list}, timeout=10)
        return res.json()
    except:
        return None

@st.cache_data(ttl=900, show_spinner=False)
def scan_market_leaders_fast(watchlist):
    scanned = []
    q_list = list(watchlist)
    if not q_list:
        return pd.DataFrame()

    try:
        df = yf.download(q_list, period="5d", progress=False)
        if df.empty:
            return pd.DataFrame()

        for t in q_list:
            try:
                if isinstance(df.columns, pd.MultiIndex):
                    col = df["Close"][t] if t in df["Close"].columns else None
                else:
                    col = df["Close"] if "Close" in df.columns else None

                if col is not None:
                    series = col.dropna()
                    if len(series) >= 2:
                        today = float(series.iloc[-1])
                        prev = float(series.iloc[-2])
                        change = ((today - prev) / prev) * 100
                        scanned.append({"ticker": t, "price": today, "change": change})
            except:
                continue
    except:
        pass

    return pd.DataFrame(scanned)

# --- UI Momentum Watchlist ---
st.markdown("### 🔥 Live Momentum Watchlist")
if not parsed_wl:
    st.warning("Enter tickers in the sidebar controls.")
else:
    with st.spinner("Scanning..."):
        scanner_df = scan_market_leaders_fast(watchlist_tuple)

    if not scanner_df.empty:
        f_df = scanner_df[scanner_df["price"] <= max_price_filter]
        f_df = f_df.sort_values(by="change", ascending=False)

        for _, row in f_df.iterrows():
            if abs(row["change"]) >= alert_threshold:
                alert_msg = f"{row['ticker']} shifted {row['change']:+.2f}%"
                st.toast(alert_msg)

        display_count = min(4, len(f_df))
        if display_count > 0:
            cols = st.columns(display_count)
            for i, row in enumerate(f_df.head(display_count).itertuples()):
                lab = row.ticker
                val = f"${row.price:.2f}"
                delta = f"{row.change:+.2f}%"
                cols[i].metric(label=lab, value=val, delta=delta)

st.markdown("---")
st.markdown("### 🔍 Run Deep AI Multi-Week Analysis")
user_input = st.text_input("Company Name or Ticker:", "Apple")

if st.button("Run Master Multi-Week Analysis"):
    try:
        with st.status("Running Engine...", expanded=True) as status:
            st.write("Resolving asset identifier...")
            ticker = resolve_company_name(user_input)
            st.write(f"Target Confirmed: {ticker}")

            stock = yf.Ticker(ticker)
            data = stock.history(period="1y")
            intraday = stock.history(period="5d", interval="5m")

            if data.empty:
                status.update(label="Error", state="error")
                st.error("No market data found for this asset.")
            else:
                feats = ["Open", "High", "Low", "Close", "Volume"]
                data = data.dropna(subset=feats).copy()

                if data.empty:
                    status.update(label="Error", state="error")
                    st.error("Not enough clean market data found for this asset.")
                else:
                    st.write("Training predictive models...")
                    current_price = float(data["Close"].iloc[-1])

                    data["SMA_20"] = data["Close"].rolling(window=20).mean()
                    data["SMA_50"] = data["Close"].rolling(window=50).mean()
                    recent = data.dropna(subset=["SMA_20", "SMA_50"]).iloc[-90:].copy()

                    if len(data) < 20 or recent.empty:
                        status.update(label="Error", state="error")
                        st.error("Not enough historical data for the selected analysis window.")
                    else:
                        latest_data = data[feats].iloc[-1:]

                        def fit_forecast(shift_n):
                            df_w = data.copy()
                            df_w["Target"] = df_w["Close"].shift(-shift_n)
                            df_w = df_w.dropna(subset=feats + ["Target"])
                            if len(df_w) < 20:
                                return None
                            model = RandomForestRegressor(n_estimators=100, random_state=42)
                            model.fit(df_w[feats], df_w["Target"])
                            return float(model.predict(latest_data)[0])

                        forecast_w1 = fit_forecast(5)
                        forecast_w2 = fit_forecast(10)
                        forecast_w3 = fit_forecast(15)

                        if None in (forecast_w1, forecast_w2, forecast_w3):
                            status.update(label="Error", state="error")
                            st.error("Unable to build one or more forecasts from the available data.")
                        else:
                            st.write("Scanning global sentiment...")
                            news = get_resilient_news(stock, ticker)
                            bull_s = 0.0
                            bear_s = 0.0
                            num_hl = 0
                            engine = "TextBlob Engine"

                            if news:
                                texts = [i.get("title", "") for i in news[:5] if i.get("title")]
                                num_hl = len(texts)

                                if texts:
                                    if hf_token:
                                        api_res = query_finbert_api(texts, hf_token)
                                        valid_res = isinstance(api_res, list) and api_res and "error" not in str(api_res).lower()

                                        if valid_res:
                                            engine = "FinBERT Engine"
                                            for r_list in api_res:
                                                if isinstance(r_list, list):
                                                    for item in r_list:
                                                        lbl = item.get("label")
                                                        scr = item.get("score", 0)
                                                        if lbl == "positive":
                                                            bull_s += scr
                                                        elif lbl == "negative":
                                                            bear_s += scr
                                        else:
                                            hf_token = None

                                    if not hf_token:
                                        for txt in texts:
                                            pol = TextBlob(txt).sentiment.polarity
                                            if pol > 0.1:
                                                bull_s += pol
                                            elif pol < -0.1:
                                                bear_s +=
