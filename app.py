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

# --- SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ Dashboard Controls")

if not hf_token:
    hf_token = st.sidebar.text_input("Hugging Face Token (Optional)", type="password", 
                                     help="Set up HF_TOKEN in your App Secrets to hide this box.")

st.sidebar.markdown("---")
st.sidebar.header("📝 Personal Watchlist")
# Let the user type their own tickers
user_watchlist_input = st.sidebar.text_input(
    "Enter tickers (comma-separated):", 
    value="AAPL, NVDA, TSLA, AMD, MSFT, AMZN, META, GOOGL"
)

# Parse the user input into a clean list of uppercase tickers
parsed_watchlist = [ticker.strip().upper() for ticker in user_watchlist_input.split(",") if ticker.strip()]
# We use a tuple for caching because Streamlit caches require immutable (unchangeable) variables
watchlist_tuple = tuple(parsed_watchlist)

st.sidebar.markdown("---")
st.sidebar.header("🚨 Alert Settings")
# Let the user define what a "Big Movement" is
alert_threshold = st.sidebar.number_input(
    "Alert me if a stock moves more than (%):", 
    min_value=1.0, 
    max_value=50.0, 
    value=4.0, 
    step=0.5,
    help="Triggers an on-screen popup notification if a stock exceeds this daily change."
)

st.sidebar.markdown("---")
max_price_filter = st.sidebar.slider(
    "Filter Watchlist by Max Price ($):", 
    min_value=10, 
    max_value=1000, 
    value=1000, 
    step=10,
    help="Slide left to only show cheaper stocks in your upfront watchlist."
)

# Helper function to talk to Hugging Face's serverless AI
def query_finbert_api(text_list, token):
    api_url = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.post(api_url, headers=headers, json={"inputs": text_list}, timeout=10)
        return response.json()
    except:
        return None

# --- UPGRADED BULLETPROOF BATCH SCANNER ---
@st.cache_data(ttl=900, show_spinner=False)  # Caches for 15 mins, ignores slider movements
def scan_market_leaders_fast(watchlist):
    scanned_data = []
    # Convert tuple back to list for yfinance
    query_list = list(watchlist)
    
    if not query_list:
        return pd.DataFrame()
        
    try:
        # Pull everything at once in a single lightweight web request
        df = yf.download(query_list, period="5d", progress=False) 
        
        if df.empty:
            return pd.DataFrame()

        # Handle yfinance MultiIndex structure
        if isinstance(df.columns, pd.MultiIndex):
            close_df = df['Close']
        else:
            close_df = df
            
        for t in query_list:
            if t in close_df.columns or (len(query_list) == 1):
                # Handle single stock vs multi stock series differences safely
                price_series = close_df[t].dropna() if len(query_list) > 1 else close_df.dropna()
                
                if len(price_series) >= 2:
                    close_today = price_series.iloc[-1]
                    close_prev = price_series.iloc[-2]
                    pct_change = ((close_today - close_prev) / close_prev) * 100
                    scanned_data.append({"ticker": t, "price": float(close_today), "change": float(pct_change)})
    except Exception as e:
        pass # Fallback gracefully
        
    return pd.DataFrame(scanned_data)

# --- UI Render: Upfront Scanner & Alerts ---
st.markdown("### 🔥 Live Momentum Watchlist")

if not parsed_watchlist:
    st.warning("Please enter at least one ticker in the sidebar.")
else:
    with st.spinner("Scanning your custom watchlist..."):
        scanner_df = scan_market_leaders_fast(watchlist_tuple)

    if not scanner_df.empty:
        # Filter the results in real-time based on your slider position
        filtered_df = scanner_df[scanner_df['price'] <= max_price_filter].sort_values(by="change", ascending=False)
        
        # --- TRIGGER POPUP ALERTS ---
        for _, row in filtered_df.iterrows():
            if abs(row['change']) >= alert_threshold:
                if row['change'] > 0:
                    st.toast(f"🚀 SURGE ALERT: {row['ticker']} is UP {row['change']:+.2f}% today!", icon="🚀")
                else:
                    st.toast(f"🩸 DUMP ALERT: {row['ticker']} is DOWN {row['change']:+.2f}% today!", icon="🚨")
        
        if not filtered_df.empty:
            # Dynamically create columns based on how many stocks match your filter
            display_count = min(4, len(filtered_df))
            cols = st.columns(display_count)
            for i, row in enumerate(filtered_df.head(display_count).itertuples()):
                cols[i].metric(
                    label=row.ticker, 
                    value=f"${row.price:.2f}", 
                    delta=f"{row.change:+.2f}%"
                )
        else:
            st.warning(f"⚠️ No watchlist stocks found under ${max_price_filter}. Adjust the sidebar slider to a higher value.")
    else:
        st.info("💡 Live feed temporarily unavailable or invalid tickers. Enter a ticker directly below to begin.")

st.markdown("---")

# --- User Input Deep Dive ---
st.markdown("### 🔍 Run Deep AI Analysis")

# --- TIMEFRAME SELECTOR ---
input_col1, input_col2 = st.columns([2, 1])
with input_col1:
    ticker = st.text_input("Enter Ticker Symbol to analyze:", "AAPL").upper()
with input_col2:
    horizon_choice = st.selectbox("AI Forecast Horizon:", ["1 Week (5 Days)", "2 Weeks (10 Days)"])

# Determine the math shift based on user selection
if "1 Week" in horizon_choice:
    shift_days = 5
    calendar_days = 7
else:
    shift_days = 10
    calendar_days = 14

if st.button("Run Master Analysis"):
    try:
        with st.status("Initializing ProQuant AI Engine...", expanded=True) as status:
            
            st.write("📡 Fetching multi-timeframe market data...")
            stock = yf.Ticker(ticker)
            data = stock.history(period="1y")      
            intraday = stock.history(period="5d", interval="5m")  
            
            if data.empty:
                status.update(label="Data Error", state="error", expanded=True)
                st.error("No data found. Please check the ticker symbol.")
            else:
                st.write(f"🧠 Running predictive mathematical model for {horizon_choice}...")
                df = data.copy()
                
                # --- DYNAMIC TARGET SHIFT based on user selection ---
                df['Target'] = df['Close'].shift(-shift_days)
                df.dropna(inplace=True)
                
                X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                y = df['Target']
                
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X, y)
                
                latest_data = data[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[-1:]
                forecast = model.predict(latest_data)[0]
                current_price = data['Close'].iloc[-1]
                
                st.write("📰 Scanning live market sentiment & news...")
                news = stock.news
                bullish_score = 0
                bearish_score = 0
                engine_used = "TextBlob (Basic)"
                
                if news:
                    news_texts = [item.get('title', '') for item in news[:5] if 'title' in item] 
                    if news_texts:
                        if hf_token:
                            api_result = query_finbert_api(news_texts, hf_token)
                            if api_result and isinstance(api_result, list) and "error" not in api_result:
                                engine_used = "FinBERT Cloud AI (Elite)"
                                for res_list in api_result:
                                    if isinstance(res_list, list):
                                        for item in res_list:
                                            if item.get('label') == 'positive':
                                                bullish_score += item.get('score', 0)
                                            elif item.get('label') == 'negative':
                                                bearish_score += item.get('score', 0)
                            else:
                                engine_used = "TextBlob (API Fallback)"
                                for text in news_texts:
                                    analysis = TextBlob(text)
                                    if analysis.sentiment.polarity > 0.1:
                                        bullish_score += analysis.sentiment.polarity
                                    elif analysis.sentiment.polarity < -0.1:
                                        bearish_score += abs(analysis.sentiment.polarity)
                        else:
                            for text in news_texts:
                                analysis = TextBlob(text)
                                if analysis.sentiment.polarity > 0.1:
                                    bullish_score += analysis.sentiment.polarity
                                elif analysis.sentiment.polarity < -0.1:
                                    bearish_score += abs(analysis.sentiment.polarity)
                
                status.update(label="Analysis Complete!", state="complete", expanded=False)

        if not data.empty:
            tab1, tab2 = st.tabs(["🏦 Swing & AI Forecast", "⚡ Day Trading Engine"])
            
            # --- TAB 1: LONG TERM FORECAST & RISK MANAGEMENT ---
            with tab1:
                st.markdown(f"### 📊 AI Forecast Outlook: {ticker}")
                col1, col2 = st.columns(2)
                col1.metric(label="Current Price", value=f"${current_price:.2f}")
                
                forecast_diff = forecast - current_price
                if forecast_diff >= 0:
                    delta_display = f"+${forecast_diff:.2f}"
                else:
                    delta_display = f"-${abs(forecast_diff):.2f}"
                    
                col2.metric(label=f"AI Target ({horizon_choice})", value=f"${forecast:.2f}", delta=delta_display)

                # --- RISK MANAGEMENT CALCULATOR ---
                st.markdown("---")
                st.markdown("#### 🛡️ Risk Management Calculator")
                r1, r2, r3 = st.columns(3)
                
                entry_input = r1.number_input("Entry Price ($)", value=float(current_price))
                sl_input = r2.number_input("Stop Loss ($)", value=float(current_price * 0.95))
                tp_input = r3.number_input("Target Price ($)", value=float(forecast))

                risk = entry_input - sl_input
                reward = tp_input - entry
                
