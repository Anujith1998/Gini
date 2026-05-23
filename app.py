import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from textblob import TextBlob
import requests
import plotly.graph_objects as go
from datetime import timedelta

# --- App Configuration ---
st.set_page_config(page_title="ProQuant AI", layout="wide", page_icon="📈")
st.title("ProQuant AI 📈")
st.write("Advanced Market Analysis & Day Trading Engine")

# --- Automated Token Detection ---
hf_token = st.secrets.get("HF_TOKEN", None)

# --- SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ Dashboard Controls")

if not hf_token:
    hf_token = st.sidebar.text_input("Hugging Face Token (Optional)", type="password", 
                                     help="Set up HF_TOKEN in your App Secrets.")

st.sidebar.markdown("---")
st.sidebar.header("📝 Personal Watchlist")
user_watchlist_input = st.sidebar.text_input(
    "Enter tickers (comma-separated):", 
    value="AAPL, NVDA, TSLA, AMD, MSFT, AMZN, META, GOOGL"
)

parsed_watchlist = [ticker.strip().upper() for ticker in user_watchlist_input.split(",") if ticker.strip()]
watchlist_tuple = tuple(parsed_watchlist)

st.sidebar.markdown("---")
st.sidebar.header("🚨 Alert Settings")
alert_threshold = st.sidebar.number_input("Alert (%) threshold:", min_value=1.0, max_value=50.0, value=4.0, step=0.5)

st.sidebar.markdown("---")
max_price_filter = st.sidebar.slider("Filter Watchlist by Max Price ($):", min_value=10, max_value=1000, value=1000, step=10)

# --- API FUNCTION ---
def query_finbert_api(text_list, token):
    api_url = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.post(api_url, headers=headers, json={"inputs": text_list}, timeout=10)
        return response.json()
    except:
        return None

# --- WATCHLIST SCANNER ---
@st.cache_data(ttl=900, show_spinner=False)
def scan_market_leaders_fast(watchlist):
    scanned_data = []
    if not watchlist: return pd.DataFrame()
    try:
        df = yf.download(list(watchlist), period="5d", progress=False) 
        if df.empty: return pd.DataFrame()
        for t in watchlist:
            try:
                # Handle MultiIndex for multiple tickers
                col_ref = df['Close'][t] if isinstance(df.columns, pd.MultiIndex) else df['Close']
                close_series = col_ref.dropna()
                if len(close_series) >= 2:
                    change = ((float(close_series.iloc[-1]) - float(close_series.iloc[-2])) / float(close_series.iloc[-2])) * 100
                    scanned_data.append({"ticker": t, "price": float(close_series.iloc[-1]), "change": change})
            except: continue
    except: pass
    return pd.DataFrame(scanned_data)

# --- UI Render: Scanner ---
st.markdown("### 🔥 Live Momentum Watchlist")
if not parsed_watchlist:
    st.warning("Please enter at least one ticker in the sidebar.")
else:
    scanner_df = scan_market_leaders_fast(watchlist_tuple)
    if not scanner_df.empty:
        filtered_df = scanner_df[scanner_df['price'] <= max_price_filter].sort_values(by="change", ascending=False)
        for _, row in filtered_df.iterrows():
            if abs(row['change']) >= alert_threshold:
                st.toast(f"{row['ticker']} moved {row['change']:+.2f}%", icon="🚀" if row['change'] > 0 else "🚨")
        
        cols = st.columns(min(4, len(filtered_df)))
        for i, row in enumerate(filtered_df.head(4).itertuples()):
            cols[i].metric(label=row.ticker, value=f"${row.price:.2f}", delta=f"{row.change:+.2f}%")

st.markdown("---")
st.markdown("### 🔍 Run Deep AI Analysis")

col_a, col_b = st.columns([2, 1])
ticker = col_a.text_input("Enter Ticker:", "AAPL").upper()
horizon = col_b.selectbox("Forecast Horizon:", ["1 Week (5 Days)", "2 Weeks (10 Days)"])
shift_days = 5 if "1 Week" in horizon else 10
calendar_days = 7 if "1 Week" in horizon else 14

if st.button("Run Master Analysis"):
    try:
        with st.status("Initializing Engine...", expanded=True) as status:
            stock = yf.Ticker(ticker)
            data = stock.history(period="1y")      
            intraday = stock.history(period="5d", interval="5m")  
            
            if data.empty:
                st.error("No data found.")
            else:
                # Prediction Model
                df = data.copy()
                df['Target'] = df['Close'].shift(-shift_days)
                df.dropna(inplace=True)
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(df[['Open', 'High', 'Low', 'Close', 'Volume']], df['Target'])
                
                forecast = model.predict(data[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[-1:])[0]
                current_price = data['Close'].iloc[-1]
                
                # Sentiment
                bullish, bearish, engine = 0, 0, "TextBlob"
                news = stock.news
                if news:
                    news_texts = [item['title'] for item in news[:5] if 'title' in item]
                    if hf_token:
                        res = query_finbert_api(news_texts, hf_token)
                        if isinstance(res, list):
                            engine = "FinBERT AI"
                            for r in res:
                                for item in r:
                                    if item['label'] == 'positive': bullish += item['score']
                                    elif item['label'] == 'negative': bearish += item['score']
                    else:
                        for text in news_texts:
                            pol = TextBlob(text).sentiment.polarity
                            if pol > 0.1: bullish += pol
                            elif pol < -0.1: bearish += abs(pol)
                status.update(label="Complete!", state="complete", expanded=False)

        tab1, tab2 = st.tabs(["🏦 Swing & AI Forecast", "⚡ Day Trading Engine"])
        
        with tab1:
            col1, col2 = st.columns(2)
            col1.metric("Current", f"${current_price:.2f}")
            col2.metric("AI Target", f"${forecast:.2f}", delta=f"${forecast-current_price:.2f}")

            # Technical Indicators
            data['SMA_20'] = data['Close'].rolling(20).mean()
            data['SMA_50'] = data['Close'].rolling(50).mean()
            hist_std = data['Close'].pct_change().std()
            uncertainty = current_price * hist_std * (calendar_days ** 0.5)
            
            # Chart
            recent = data.iloc[-90:]
            fig = go.Figure(data=[go.Candlestick(x=recent.index, open=recent['Open'], high=recent['High'], low=recent['Low'], close=recent['Close'], name='Price')])
            fig.add_trace(go.Scatter(x=recent.index, y=recent['SMA_20'], name='20-SMA', line=dict(color='#29B6F6')))
            fig.add_trace(go.Scatter(x=recent.index, y=recent['SMA_50'], name='50-SMA', line=dict(color='#FFA726')))
            
            # Forecast Path
            last_date = recent.index[-1]
            future_date = last_date + timedelta(days=calendar_days)
            fig.add_trace(go.Scatter(x=[last_date, future_date], y=[current_price, forecast], mode='lines+markers', name='AI Path', line=dict(color='#00E676', width=3, dash='dash')))
            st.plotly_chart(fig, use_container_width=True)
            
            st.table(pd.DataFrame({
                "Metric": ["Target", "Volatility", "Sentiment"],
                "Value": [f"${forecast:.2f}", f"±${uncertainty:.2f}", "Bullish" if bullish > bearish else "Bearish"]
            }))

        with tab2:
            st.write("Intraday Momentum Engine Active.")
            if not intraday.empty:
                st.metric("Live Price", f"${intraday['Close'].iloc[-1]:.2f}")
                st.progress((intraday['Close'].iloc[-1] - intraday['Low'].min()) / (intraday['High'].max() - intraday['Low'].min()))
    except Exception as e:
        st.error(f"Analysis Error: {e}")
        
