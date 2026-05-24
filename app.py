import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from textblob import TextBlob
import requests
import plotly.graph_objects as go
from datetime import timedelta
import xml.etree.ElementTree as ET

st.set_page_config(
    page_title="ProQuant AI",
    layout="centered",
    page_icon="📈"
)
st.title("ProQuant AI 📈")
st.write("Advanced Market Analysis Engine")

hf_token = st.secrets.get("HF_TOKEN", None)

st.sidebar.header("⚙️ Controls")
if not hf_token:
    hf_token = st.sidebar.text_input("HF Token (Optional)", type="password")

st.sidebar.markdown("---")
st.sidebar.header("📝 Watchlist")
wl_input = st.sidebar.text_input("Tickers:", value="AAPL, NVDA, TSLA, AMD, MSFT")

parsed_wl = [t.strip().upper() for t in wl_input.split(",") if t.strip()]
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
    except Exception:
        pass

    return cleaned.upper()

def get_resilient_news(stock_obj, ticker_str):
    news_data = []
    try:
        news_data = stock_obj.news or []
    except Exception:
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
        except Exception:
            pass

    return news_data if news_data else []

def query_finbert_api(text_list, token):
    url = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        res = requests.post(url, headers=headers, json={"inputs": text_list}, timeout=10)
        return res.json()
    except Exception:
        return None

@st.cache_data(ttl=900, show_spinner=False)
def scan_market_leaders_fast(watchlist):
    if not watchlist:
        return pd.DataFrame()

    try:
        df = yf.download(list(watchlist), period="5d", progress=False, group_by="column")
        if df.empty:
            return pd.DataFrame()

        scanned = []
        for t in watchlist:
            try:
                if isinstance(df.columns, pd.MultiIndex):
                    col = df["Close"][t] if "Close" in df.columns.get_level_values(0) and t in df["Close"].columns else None
                else:
                    col = df["Close"] if "Close" in df.columns else None

                if col is not None:
                    series = col.dropna()
                    if len(series) >= 2:
                        today = float(series.iloc[-1])
                        prev = float(series.iloc[-2])
                        change = ((today - prev) / prev) * 100
                        scanned.append({"ticker": t, "price": today, "change": change})
            except Exception:
                continue

        return pd.DataFrame(scanned)
    except Exception:
        return pd.DataFrame()

st.markdown("### 🔥 Live Momentum Watchlist")
if not parsed_wl:
    st.warning("Enter tickers in the sidebar controls.")
else:
    with st.spinner("Scanning..."):
        scanner_df = scan_market_leaders_fast(watchlist_tuple)

    if not scanner_df.empty:
        f_df = scanner_df[scanner_df["price"] <= max_price_filter].sort_values(by="change", ascending=False)

        for _, row in f_df.iterrows():
            if abs(row["change"]) >= alert_threshold:
                st.toast(f"{row['ticker']} shifted {row['change']:+.2f}%")

        display_count = min(4, len(f_df))
        if display_count > 0:
            cols = st.columns(display_count)
            for i, row in enumerate(f_df.head(display_count).itertuples()):
                cols[i].metric(label=row.ticker, value=f"${row.price:.2f}", delta=f"{row.change:+.2f}%")

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
                                                bear_s += abs(pol)

                            status.update(label="Complete!", state="complete", expanded=False)

        if not data.empty:
            tab1, tab2, tab3 = st.tabs([
                "🏦 Multi-Week Forecast",
                "⚡ Intraday Engine",
                "👥 AI Debate Room"
            ])

            with tab1:
                st.markdown(f"### 📊 Forecast Path: {ticker}")

                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Current", f"${current_price:.2f}")
                mc2.metric("Wk 1", f"${forecast_w1:.2f}", f"{forecast_w1 - current_price:+.2f}")
                mc3.metric("Wk 2", f"${forecast_w2:.2f}", f"{forecast_w2 - forecast_w1:+.2f}")
                mc4.metric("Wk 3", f"${forecast_w3:.2f}", f"{forecast_w3 - forecast_w2:+.2f}")

                st.markdown("---")

                h_std = data["Close"].pct_change().std()
                h_std = 0 if pd.isna(h_std) else float(h_std)

                u_w1 = current_price * h_std * (7 ** 0.5)
                u_w2 = current_price * h_std * (14 ** 0.5)
                u_w3 = current_price * h_std * (21 ** 0.5)

                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=recent.index,
                    open=recent["Open"],
                    high=recent["High"],
                    low=recent["Low"],
                    close=recent["Close"],
                    name="Price"
                ))
                fig.add_trace(go.Scatter(
                    x=recent.index,
                    y=recent["SMA_20"],
                    name="20 SMA",
                    line=dict(color="#29B6F6", width=1)
                ))
                fig.add_trace(go.Scatter(
                    x=recent.index,
                    y=recent["SMA_50"],
                    name="50 SMA",
                    line=dict(color="#FFA726", width=1)
                ))

                l_date = recent.index[-1]
                t_line = [
                    l_date,
                    l_date + timedelta(days=7),
                    l_date + timedelta(days=14),
                    l_date + timedelta(days=21)
                ]
                p_vals = [current_price, forecast_w1, forecast_w2, forecast_w3]

                up_y = [
                    current_price,
                    forecast_w1 + u_w1,
                    forecast_w2 + u_w2,
                    forecast_w3 + u_w3
                ]
                dn_y = [
                    current_price,
                    forecast_w1 - u_w1,
                    forecast_w2 - u_w2,
                    forecast_w3 - u_w3
                ]

                fig.add_trace(go.Scatter(
                    x=t_line, y=up_y, mode="lines",
                    showlegend=False, line=dict(width=0)
                ))
                fig.add_trace(go.Scatter(
                    x=t_line, y=dn_y, mode="lines",
                    fill="tonexty", fillcolor="rgba(0,230,118,0.03)",
                    name="Tunnel"
                ))

                c_clr = "#00E676" if forecast_w3 >= current_price else "#FF1744"
                fig.add_trace(go.Scatter(
                    x=t_line, y=p_vals, mode="lines+markers",
                    name="AI Path", line=dict(color=c_clr, width=2, dash="dash")
                ))

                fig.update_layout(
                    xaxis_rangeslider_visible=False,
                    height=400,
                    margin=dict(l=5, r=5, t=5, b=5)
                )
                st.plotly_chart(fig, use_container_width=True)

                m_df = pd.DataFrame({
                    "Horizon": ["Week 1", "Week 2", "Week 3"],
                    "Target": [f"${forecast_w1:.2f}", f"${forecast_w2:.2f}", f"${forecast_w3:.2f}"],
                    "Move": [
                        f"{((forecast_w1 / current_price) - 1) * 100:+.2f}%",
                        f"{((forecast_w2 / current_price) - 1) * 100:+.2f}%",
                        f"{((forecast_w3 / current_price) - 1) * 100:+.2f}%"
                    ]
                })
                st.table(m_df)

                st.markdown("#### AI Diagnostics & Analysis")
                if forecast_w3 >= current_price:
                    st.success("🤖 Cascade Status: NET BULLISH DETECTED")
                else:
                    st.error("🤖 Cascade Status: NET BEARISH DETECTED")

                if num_hl > 0:
                    if bull_s > bear_s:
                        st.success(f"📰 Sentiment Layer: BULLISH ({int(num_hl)} Alerts)")
                    elif bear_s > bull_s:
                        st.error(f"📰 Sentiment Layer: BEARISH ({int(num_hl)} Alerts)")
                    else:
                        st.info(f"📰 Sentiment Layer: NEUTRAL ({int(num_hl)} Alerts)")

            with tab2:
                st.markdown(f"### ⚡ Intraday Diagnostics: {ticker}")
                if not intraday.empty:
                    i_cur = float(intraday["Close"].iloc[-1])
                    i_hi = float(intraday["High"].max())
                    i_lo = float(intraday["Low"].min())

                    cc1, cc2, cc3 = st.columns(3)
                    cc1.metric("Live Execution", f"${i_cur:.2f}")
                    cc2.metric("Session High", f"${i_hi:.2f}")
                    cc3.metric("Session Low", f"${i_lo:.2f}")

                    denom = (i_hi - i_lo) if (i_hi - i_lo) != 0 else 1
                    pos = int(((i_cur - i_lo) / denom) * 100)
                    st.progress(max(0, min(100, pos)) / 100)
                    st.caption(f"Price is at {pos}% of today's bracket.")

            with tab3:
                st.markdown("### 🗳️ 6-Agent AI Consensus Scoreboard")

                l_sma20 = float(data["SMA_20"].iloc[-1])
                l_sma50 = float(data["SMA_50"].iloc[-1])
                l_open = float(data["Open"].iloc[-1])

                v1 = forecast_w3 >= current_price
                v2 = current_price >= l_sma20
                v3 = (bull_s >= bear_s) if num_hl > 0 else True
                v4 = current_price >= l_sma50
                v5 = current_price >= l_open
                v6 = forecast_w1 >= current_price

                votes = [v1, v2, v3, v4, v5, v6]
                bull_votes = votes.count(True)
                stances = ["🟢 BULL" if v else "🔴 BEAR" for v in votes]

                v_df = pd.DataFrame({
                    "Algorithmic Voter Node": [
                        "Agent 1: Cascade ML Matrix (3-Week)",
                        "Agent 2: Short Trend Engine (20-Day SMA)",
                        "Agent 3: Media Sentiment Array",
                        "Agent 4: Macro Baseline (50-Day SMA)",
                        "Agent 5: Intraday Opening Pivot",
                        "Agent 6: Immediate Vector (1-Week Path)"
                    ],
                    "Stance": stances
                })
                st.table(v_df)

                st.markdown("#### ⚖️ Final Arbitration Verdict")
                if bull_votes >= 5:
                    st.success(f"🎯 STRONG BUY ({bull_votes} Bulls)")
                elif bull_votes == 4:
                    st.info(f"⚖️ TACTICAL ACCUMULATE ({bull_votes} Bulls)")
                elif bull_votes == 3:
                    st.warning(f"⚡ EQUAL WEIGHT / HOLD ({bull_votes} Bulls)")
                elif bull_votes == 2:
                    st.error(f"🚨 TACTICAL REDUCE / SHORT ({bull_votes} Bulls)")
                else:
                    st.error(f"💀 STRONG LIQUIDATE ({bull_votes} Bulls)")

    except Exception as e:
        st.error(f"Unexpected error: {e}")
