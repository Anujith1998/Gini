import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from textblob import TextBlob
import requests
import plotly.graph_objects as go
from datetime import timedelta
import xml.etree.ElementTree as ET

# --- Setup ---
st.set_page_config(page_title="ProQuant AI", layout="centered", page_icon="📈")
st.title("ProQuant AI 📈")
st.write("Advanced Market Analysis Engine")

hf_token = st.secrets.get("HF_TOKEN", None)

st.sidebar.header("⚙️ Controls")
if not hf_token:
    hf_token = st.sidebar.text_input("HF Token (Optional)", type="password")

st.sidebar.markdown("---")
st.sidebar.header("📝 Watchlist")
wl_input = st.sidebar.text_input("Tickers:", "AAPL, NVDA, TSLA, AMD, MSFT")

parsed_wl = []
for t in wl_input.split(","):
    if t.strip():
        parsed_wl.append(t.strip().upper())
watchlist_tuple = tuple(parsed_wl)

st.sidebar.markdown("---")
st.sidebar.header("🚨 Alerts")
alert_threshold = st.sidebar.number_input("Alert Trigger (%)", value=4.0)
max_price_filter = st.sidebar.slider("Max Price ($)", 10, 1000, 1000)

# --- Powerful Helper Functions ---
def resolve_company_name(query):
    cleaned = query.strip()
    if len(cleaned) <= 5 and cleaned.isalpha():
        return cleaned.upper()
    url = "https://query1.finance.yahoo.com/v1/finance/search"
    headers = {"User-Agent": "Mozilla/5.0"}
    params = {"q": cleaned, "lang": "en-US", "region": "US"}
    try:
        res = requests.get(url, params=params, headers=headers, timeout=5)
        data = res.json()
        if data.get('quotes'):
            return data['quotes'][0]['symbol'].upper()
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
                for item in root.findall('.//item'):
                    title = item.find('title')
                    if title is not None and title.text:
                        news_data.append({"title": title.text})
        except:
            pass
    return news_data

def query_finbert_api(text_list, token):
    url = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        payload = {"inputs": text_list}
        res = requests.post(url, headers=headers, json=payload, timeout=10)
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
                    col = df['Close'][t] if t in df['Close'].columns else None
                else:
                    col = df['Close'] if 'Close' in df.columns else None
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

# --- UI: Live Momentum Watchlist ---
st.markdown("### 🔥 Live Momentum Watchlist")
if parsed_wl:
    with st.spinner("Scanning..."):
        scanner_df = scan_market_leaders_fast(watchlist_tuple)
    if not scanner_df.empty:
        f_df = scanner_df[scanner_df['price'] <= max_price_filter]
        f_df = f_df.sort_values(by="change", ascending=False)
        for _, row in f_df.iterrows():
            if abs(row['change']) >= alert_threshold:
                st.toast(f"{row['ticker']} shifted {row['change']:+.2f}%")
        
        display_count = min(4, len(f_df))
        if display_count > 0:
            cols = st.columns(display_count)
            for i, row in enumerate(f_df.head(display_count).itertuples()):
                lab = row.ticker
                val = f"${row.price:.2f}"
                delta = f"{row.change:+.2f}%"
                cols[i].metric(label=lab, value=val, delta=delta)
else:
    st.warning("Enter tickers in the sidebar.")

st.markdown("---")

# --- UI: Main Analysis Engine ---
st.markdown("### 🔍 Run Deep AI Multi-Week Analysis")
user_input = st.text_input("Company Name or Ticker:", "Apple")

if st.button("Run Master Multi-Week Analysis"):
    try: 
        with st.status("Running Engine...", expanded=True) as status:
            st.write("Resolving identifier...")
            ticker = resolve_company_name(user_input)
            st.write(f"Confirmed: {ticker}")
            
            stock = yf.Ticker(ticker)
            data = stock.history(period="1y")      
            intraday = stock.history(period="5d", interval="5m")  
            
            if data.empty:
                status.update(label="Error", state="error")
                st.error("No market data found.")
                st.stop()
                
            st.write("Training ML models...")
            feats = ['Open', 'High', 'Low', 'Close', 'Volume']
            latest_data = data[feats].iloc[-1:]
            current_price = data['Close'].iloc[-1]
            
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            recent = data.iloc[-90:].copy()
            
            # Machine Learning 1: 5 Days
            df_w1 = data.copy()
            df_w1['Target'] = df_w1['Close'].shift(-5)
            df_w1.dropna(inplace=True)
            m1 = RandomForestRegressor(n_estimators=100, random_state=42)
            m1.fit(df_w1[feats], df_w1['Target'])
            forecast_w1 = m1.predict(latest_data)[0]
            
            # Machine Learning 2: 10 Days
            df_w2 = data.copy()
            df_w2['Target'] = df_w2['Close'].shift(-10)
            df_w2.dropna(inplace=True)
            m2 = RandomForestRegressor(n_estimators=100, random_state=42)
            m2.fit(df_w2[feats], df_w2['Target'])
            forecast_w2 = m2.predict(latest_data)[0]
            
            # Machine Learning 3: 15 Days
            df_w3 = data.copy()
            df_w3['Target'] = df_w3['Close'].shift(-15)
            df_w3.dropna(inplace=True)
            m3 = RandomForestRegressor(n_estimators=100, random_state=42)
            m3.fit(df_w3[feats], df_w3['Target'])
            forecast_w3 = m3.predict(latest_data)[0]
            
            st.write("Scanning global sentiment...")
            news = get_resilient_news(stock, ticker)
            bull_s, bear_s = 0, 0
            
            if news:
                texts = [i.get('title', '') for i in news[:5] if 'title' in i]
                num_hl = len(texts)
                if num_hl > 0:
                    api_res = None
                    if hf_token:
                        api_res = query_finbert_api(texts, hf_token)
                        
                    is_valid_api = isinstance(api_res, list) and "error" not in api_res
                    if hf_token and is_valid_api:
                        for r_list in api_res:
                            if isinstance(r_list, list):
                                for item in r_list:
                                    lbl = item.get('label')
                                    scr = item.get('score', 0)
                                    if lbl == 'positive': bull_s += scr
                                    elif lbl == 'negative': bear_s += scr
                    else:
                        for txt in texts:
                            pol = TextBlob(txt).sentiment.polarity
                            if pol > 0.1: bull_s += pol
                            elif pol < -0.1: bear_s += abs(pol)
            else:
                num_hl = 0
            
            status.update(label="Complete!", state="complete", expanded=False)

        # --- Dashboard Rendering ---
        if not data.empty:
            tab1, tab2, tab3 = st.tabs(["🏦 Forecast", "⚡ Intraday", "👥 AI Debate"])
            
            with tab1:
                st.markdown(f"### 📊 Forecast Path: {ticker}")
                
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Current", f"${current_price:.2f}")
                
                val1 = f"${forecast_w1:.2f}"
                del1 = f"{forecast_w1-current_price:+.2f}"
                mc2.metric("Wk 1", val1, del1)
                
                val2 = f"${forecast_w2:.2f}"
                del2 = f"{forecast_w2-forecast_w1:+.2f}"
                mc3.metric("Wk 2", val2, del2)
                
                val3 = f"${forecast_w3:.2f}"
                del3 = f"{forecast_w3-forecast_w2:+.2f}"
                mc4.metric("Wk 3", val3, del3)
                st.markdown("---")
                
                # Math for standard deviation tunnels
                h_std = data['Close'].pct_change().std()
                u_w1 = current_price * h_std * (7 ** 0.5)
                u_w2 = current_price * h_std * (14 ** 0.5)
                u_w3 = current_price * h_std * (21 ** 0.5)
                
                fig = go.Figure()
                
                c_trace = go.Candlestick(
                    x=recent.index, open=recent['Open'], 
                    high=recent['High'], low=recent['Low'], 
                    close=recent['Close'], name='Price'
                )
                fig.add_trace(c_trace)
                
                s20_trace = go.Scatter(x=recent.index, y=recent['SMA_20'], name='20 SMA', line=dict(color='#29B6F6', width=1))
                fig.add_trace(s20_trace)
                
                s50_trace = go.Scatter(x=recent.index, y=recent['SMA_50'], name='50 SMA', line=dict(color='#FFA726', width=1))
                fig.add_trace(s50_trace)
                
                l_date = recent.index[-1]
                t_line = [l_date, l_date+timedelta(7), l_date+timedelta(14), l_date+timedelta(21)]
                p_vals = [current_price, forecast_w1, forecast_w2, forecast_w3]
                
                up_y = [current_price, forecast_w1+u_w1, forecast_w2+u_w2, forecast_w3+u_w3]
                up_trace = go.Scatter(x=t_line, y=up_y, mode='lines', showlegend=False, line=dict(width=0))
                fig.add
                
