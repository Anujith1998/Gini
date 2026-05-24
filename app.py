import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from textblob import TextBlob
import requests
import plotly.graph_objects as go
from datetime import timedelta
import xml.etree.ElementTree as ET

st.set_page_config(page_title="ProQuant AI", layout="centered")
st.title("ProQuant AI 📈")

hf_token = st.secrets.get("HF_TOKEN", None)

st.sidebar.header("Controls")
if not hf_token:
    hf_token = st.sidebar.text_input("HF Token", type="password")

wl_input = st.sidebar.text_input("Tickers:", "AAPL, TSLA, NVDA")
parsed_wl = [t.strip().upper() for t in wl_input.split(",") if t.strip()]

st.sidebar.header("Alerts")
alert_threshold = st.sidebar.number_input("Alert Trigger (%)", value=4.0)

# --- Core Functions ---
def get_company_ticker(query):
    query = query.strip()
    if len(query) <= 5 and query.isalpha():
        return query.upper()
    url = "https://query1.finance.yahoo.com/v1/finance/search"
    params = {"q": query, "lang": "en-US", "region": "US"}
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=5)
        return r.json()['quotes'][0]['symbol'].upper()
    except:
        return query.upper()

# --- Main App ---
st.markdown("### 🔍 Run Master Multi-Week Analysis")
user_input = st.text_input("Company or Ticker:", "AAPL")

if st.button("Run Analysis"):
    with st.spinner("Analyzing market data..."):
        ticker = get_company_ticker(user_input)
        stock = yf.Ticker(ticker)
        data = stock.history(period="1y")
        intraday = stock.history(period="5d", interval="5m")
        
        if len(data) < 50:
            st.error("Not enough market data to build ML models.")
            st.stop()
            
        current_price = data['Close'].iloc[-1]
        data['SMA_20'] = data['Close'].rolling(20).mean()
        data['SMA_50'] = data['Close'].rolling(50).mean()
        recent = data.iloc[-90:].copy()
        
        feats = ['Open', 'High', 'Low', 'Close', 'Volume']
        latest_data = data[feats].iloc[-1:]
        
        # ML Models - Wrapped safely to prevent mobile breaks
        df_1 = data.copy()
        df_1['T'] = df_1['Close'].shift(-5)
        df_1.dropna(inplace=True)
        m1 = RandomForestRegressor(n_estimators=50, random_state=42)
        m1.fit(df_1[feats], df_1['T'])
        w1_pred = m1.predict(latest_data)[0]
        
        df_2 = data.copy()
        df_2['T'] = df_2['Close'].shift(-10)
        df_2.dropna(inplace=True)
        m2 = RandomForestRegressor(n_estimators=50, random_state=42)
        m2.fit(df_2[feats], df_2['T'])
        w2_pred = m2.predict(latest_data)[0]
        
        df_3 = data.copy()
        df_3['T'] = df_3['Close'].shift(-15)
        df_3.dropna(inplace=True)
        m3 = RandomForestRegressor(n_estimators=50, random_state=42)
        m3.fit(df_3[feats], df_3['T'])
        w3_pred = m3.predict(latest_data)[0]
        
        # Sentiment Fallback
        bull_s, bear_s, hl_count = 0, 0, 0
        try:
            news = stock.news
            if not news:
                raise Exception("No news")
        except:
            news = []
            
        texts = [i.get('title', '') for i in news[:5]]
        hl_count = len(texts)
        for txt in texts:
            pol = TextBlob(txt).sentiment.polarity
            if pol > 0.1: bull_s += 1
            elif pol < -0.1: bear_s += 1
            
        # Render Tabs
        t1, t2, t3 = st.tabs(["🏦 Forecast", "⚡ Intraday", "👥 Debate Room"])
        
        with t1:
            st.markdown(f"### 📊 3-Week Forecast Path: {ticker}")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Current", f"${current_price:.2f}")
            c2.metric("Wk 1", f"${w1_pred:.2f}", f"{w1_pred-current_price:+.2f}")
            c3.metric("Wk 2", f"${w2_pred:.2f}", f"{w2_pred-w1_pred:+.2f}")
            c4.metric("Wk 3", f"${w3_pred:.2f}", f"{w3_pred-w2_pred:+.2f}")
            
            fig = go.Figure()
            
            # Short line charting to prevent mobile syntax errors
            p_line = dict(color='#29B6F6', width=2)
            p_trace = go.Scatter(x=recent.index, y=recent['Close'], name='Price', line=p_line)
            fig.add_trace(p_trace)
            
            s20_line = dict(color='blue', width=1)
            s20_trace = go.Scatter(x=recent.index, y=recent['SMA_20'], name='20-SMA', line=s20_line)
            fig.add_trace(s20_trace)
            
            s50_line = dict(color='orange', width=1)
            s50_trace = go.Scatter(x=recent.index, y=recent['SMA_50'], name='50-SMA', line=s50_line)
            fig.add_trace(s50_trace)
            
            last_dt = recent.index[-1]
            dt_list = [last_dt, last_dt+timedelta(7), last_dt+timedelta(14), last_dt+timedelta(21)]
            val_list = [current_price, w1_pred, w2_pred, w3_pred]
            
            c_color = '#00E676' if w3_pred >= current_price else '#FF1744'
            f_line = dict(color=c_color, dash='dash', width=2)
            f_trace = go.Scatter(x=dt_list, y=val_list, mode='lines+markers', name='AI Path', line=f_line)
            fig.add_trace(f_trace)
            
            fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
        with t2:
            st.markdown("### ⚡ Intraday Engine")
            if not intraday.empty:
                i_cur = intraday['Close'].iloc[-1]
                i_hi = intraday['High'].max()
                i_lo = intraday['Low'].min()
                st.write(f"**Live:** ${i_cur:.2f} | **High:** ${i_hi:.2f} | **Low:** ${i_lo:.2f}")
                
                spread = i_hi - i_lo if i_hi != i_lo else 1
                pos = max(0, min(100, int(((i_cur - i_lo) / spread) * 100)))
                st.progress(pos / 100)
                st.caption(f"Current Execution is at {pos}% of today's total range.")
                
        with t3:
            st.markdown("### 🗳️ 6-Agent AI Consensus Scoreboard")
            
            l_sma20 = data['SMA_20'].iloc[-1]
            l_sma50 = data['SMA_50'].iloc[-1]
            l_open = data['Open'].iloc[-1]
            
            v1 = w3_pred >= current_price
            v2 = current_price >= l_sma20
            v3 = bull_s >= bear_s
            v4 = current_price >= l_sma50
            v5 = current_price >= l_open
            v6 = w1_pred >= current_price
            
            votes = [v1, v2, v3, v4, v5, v6]
            bull_count = sum(votes)
            
            v_df = pd.DataFrame({
                "Agent Array": ["ML Cascade", "20-SMA", "Sentiment", "50-SMA", "Intraday Pivot", "1-Week Vector"],
                "Stance": ["🟢 BULL" if v else "🔴 BEAR" for v in votes]
            })
            st.table(v_df)
            
            st.markdown(f"#### ⚖️ Arbitration: {bull_count}/6 Bullish")
            
            # Forward Path Consensus Logic
            f_w1 = sum([w3_pred>=w1_pred, w1_pred>=l_sma20, v3, w1_pred>=l_sma50, w1_pred>=current_price, w2_pred>=w1_pred])
            f_w2 = sum([w3_pred>=w2_pred, w2_pred>=l_sma20, v3, w2_pred>=l_sma50, w2_pred>=w1_pred, w3_pred>=w2_pred])
            f_w3 = sum([True, w3_pred>=l_sma20, v3, w3_pred>=l_sma50, w3_pred>=w2_pred, True])
            
            f_scores = [bull_count, f_w1, f_w2, f_w3]
            
            fig_c = go.Figure()
            path_trace = go.Scatter(x=dt_list, y=f_scores, mode='lines+markers', line=dict(color=c_color, width=3))
            fig_c.add_trace(path_trace)
            fig_c.update_layout(yaxis=dict(range=[-0.5, 6.5], title="Bull Votes (0-6)"), height=250)
            
            st.plotly_chart(fig_c, use_container_width=True)

