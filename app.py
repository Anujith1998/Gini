import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from textblob import TextBlob
import requests
import plotly.graph_objects as go
from datetime import timedelta

# --- App Configuration ---
st.set_page_config(page_title="ProQuant AI", layout="centered", page_icon="📈")
st.title("ProQuant AI 📈")
st.write("Advanced Market Analysis & Day Trading Engine")

# --- Automated Token Detection ---
hf_token = st.secrets.get("HF_TOKEN", None)
st.sidebar.header("⚙️ Dashboard Controls")

if not hf_token:
    hf_token = st.sidebar.text_input("Hugging Face Token (Optional)", type="password", 
                                     help="Set up HF_TOKEN in your App Secrets to hide this box.")

max_price_filter = st.sidebar.slider(
    "Filter Watchlist by Max Price ($):", 
    min_value=10, 
    max_value=1000, 
    value=500, 
    step=10
)

# Helper function
def query_finbert_api(text_list, token):
    api_url = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.post(api_url, headers=headers, json={"inputs": text_list}, timeout=10)
        return response.json()
    except:
        return None

# Scanner with Smart Cache
@st.cache_data(ttl=900, show_spinner=False)
def scan_market_leaders_fast():
    watchlist = ["AAPL", "NVDA", "TSLA", "AMD", "MSFT", "AMZN", "META", "GOOGL"]
    scanned_data = []
    try:
        df = yf.download(watchlist, period="5d", progress=False) 
        if not df.empty:
            close_prices = df['Close'] if isinstance(df.columns, pd.MultiIndex) else df
            for t in watchlist:
                if t in close_prices.columns:
                    price_series = close_prices[t].dropna()
                    if len(price_series) >= 2:
                        close_today = price_series.iloc[-1]
                        close_prev = price_series.iloc[-2]
                        pct_change = ((close_today - close_prev) / close_prev) * 100
                        scanned_data.append({"ticker": t, "price": close_today, "change": pct_change})
    except: pass
    return pd.DataFrame(scanned_data)

# --- UI Render ---
st.markdown("### 🔥 Live Momentum Watchlist")
with st.spinner("Scanning..."):
    scanner_df = scan_market_leaders_fast()

if not scanner_df.empty:
    filtered_df = scanner_df[scanner_df['price'] <= max_price_filter].sort_values(by="change", ascending=False)
    if not filtered_df.empty:
        display_count = min(4, len(filtered_df))
        cols = st.columns(display_count)
        for i, row in enumerate(filtered_df.head(display_count).itertuples()):
            cols[i].metric(label=row.ticker, value=f"${row.price:.2f}", delta=f"{row.change:+.2f}%")

st.markdown("---")
st.markdown("### 🔍 Run Deep AI Analysis")
ticker = st.text_input("Enter Ticker Symbol:", "AAPL").upper()

if st.button("Run Master Analysis"):
    with st.status("Analyzing...", expanded=True) as status:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1y")
        intraday = stock.history(period="5d", interval="5m")
        
        if data.empty:
            st.error("Invalid Ticker")
        else:
            # Model Logic
            df = data.copy()
            df['Target'] = df['Close'].shift(-5)
            df.dropna(inplace=True)
            model = RandomForestRegressor(n_estimators=100).fit(df[['Open', 'High', 'Low', 'Close', 'Volume']], df['Target'])
            forecast = model.predict(data[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[-1:])[0]
            current_price = data['Close'].iloc[-1]
            status.update(label="Analysis Complete!", state="complete", expanded=False)

            # --- TABS ---
            tab1, tab2 = st.tabs(["🏦 Swing & AI Forecast", "⚡ Day Trading Engine"])
            
            with tab1:
                st.markdown(f"### 📊 AI Forecast: {ticker}")
                col1, col2 = st.columns(2)
                col1.metric("Current Price", f"${current_price:.2f}")
                col2.metric("5-Day AI Target", f"${forecast:.2f}", delta=f"${forecast - current_price:.2f}")

                # --- NEW: RISK MANAGEMENT CALCULATOR ---
                st.markdown("---")
                st.subheader("🛡️ Risk Management Calculator")
                r1, r2, r3 = st.columns(3)
                entry_input = r1.number_input("Entry Price", value=float(current_price))
                sl_input = r2.number_input("Stop Loss", value=float(current_price * 0.95))
                tp_input = r3.number_input("Target Price", value=float(forecast))

                risk = entry_input - sl_input
                reward = tp_input - entry_input
                
                if risk > 0:
                    ratio = reward / risk
                    st.metric("Risk/Reward Ratio", f"{ratio:.2f} : 1")
                    if ratio >= 2.0:
                        st.success(f"✅ GOOD TRADE: Potential reward is {ratio:.1f}x your risk.")
                    elif ratio > 1.0:
                        st.warning("⚠️ CAUTION: Reward is less than 2x your risk.")
                    else:
                        st.error("❌ AVOID: You are risking more than you stand to gain.")
                else:
                    st.error("Check your numbers: Stop Loss must be lower than Entry for a Buy trade.")

                # Charting
                fig = go.Figure(data=[go.Candlestick(x=data.index[-45:], open=data['Open'][-45:], high=data['High'][-45:], low=data['Low'][-45:], close=data['Close'][-45:])])
                fig.add_trace(go.Scatter(x=[data.index[-1], data.index[-1] + timedelta(days=5)], y=[current_price, forecast], mode='lines+markers', line=dict(color='#00E676', width=3, dash='dash')))
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                st.markdown(f"### ⚡ Intraday Momentum")
                if not intraday.empty:
                    intra_curr = intraday['Close'].iloc[-1]
                    st.metric("Live Price", f"${intra_curr:.2f}")
                    fig_intra = go.Figure(data=[go.Candlestick(x=intraday.index, open=intraday['Open'], high=intraday['High'], low=intraday['Low'], close=intraday['Close'])])
                    st.plotly_chart(fig_intra, use_container_width=True)
                    
