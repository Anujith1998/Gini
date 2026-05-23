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
user_watchlist_input = st.sidebar.text_input(
    "Enter tickers (comma-separated):", 
    value="AAPL, NVDA, TSLA, AMD, MSFT, AMZN, META, GOOGL"
)

parsed_watchlist = [ticker.strip().upper() for ticker in user_watchlist_input.split(",") if ticker.strip()]
watchlist_tuple = tuple(parsed_watchlist)

st.sidebar.markdown("---")
st.sidebar.header("🚨 Alert Settings")
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
    query_list = list(watchlist)
    
    if not query_list:
        return pd.DataFrame()
        
    try:
        df = yf.download(query_list, period="5d", progress=False) 
        if df.empty:
            return pd.DataFrame()

        for t in query_list:
            try:
                if isinstance(df.columns, pd.MultiIndex):
                    if 'Close' in df.columns.get_level_values(0):
                        close_df = df['Close']
                        if t in close_df.columns:
                            close_series = close_df[t].dropna()
                        else:
                            continue
                    else:
                        continue
                else:
                    if 'Close' in df.columns:
                        close_series = df['Close'].dropna()
                    else:
                        continue
                        
                if len(close_series) >= 2:
                    close_today = float(close_series.iloc[-1])
                    close_prev = float(close_series.iloc[-2])
                    pct_change = ((close_today - close_prev) / close_prev) * 100
                    scanned_data.append({"ticker": t, "price": close_today, "change": pct_change})
            except Exception:
                continue
    except Exception:
        pass
        
    return pd.DataFrame(scanned_data)

# --- UI Render: Upfront Scanner ---
st.markdown("### 🔥 Live Momentum Watchlist")

if not parsed_watchlist:
    st.warning("Please enter at least one ticker in the sidebar.")
else:
    with st.spinner("Scanning your custom watchlist..."):
        scanner_df = scan_market_leaders_fast(watchlist_tuple)

    if not scanner_df.empty:
        filtered_df = scanner_df[scanner_df['price'] <= max_price_filter].sort_values(by="change", ascending=False)
        
        for _, row in filtered_df.iterrows():
            if abs(row['change']) >= alert_threshold:
                if row['change'] > 0:
                    st.toast(f"🚀 SURGE ALERT: {row['ticker']} is UP {row['change']:+.2f}% today!", icon="🚀")
                else:
                    st.toast(f"🩸 DUMP ALERT: {row['ticker']} is DOWN {row['change']:+.2f}% today!", icon="🚨")
        
        if not filtered_df.empty:
            display_count = min(4, len(filtered_df))
            cols = st.columns(display_count)
            for i, row in enumerate(filtered_df.head(display_count).itertuples()):
                cols[i].metric(label=row.ticker, value=f"${row.price:.2f}", delta=f"{row.change:+.2f}%")
        else:
            st.warning(f"⚠️ No watchlist stocks found under ${max_price_filter}.")
    else:
        st.info("💡 Live feed temporarily unavailable or invalid tickers. Enter a ticker directly below to begin.")

st.markdown("---")
st.markdown("### 🔍 Run Deep AI Analysis")

input_col1, input_col2 = st.columns([2, 1])
with input_col1:
    ticker = st.text_input("Enter Ticker Symbol to analyze:", "AAPL").upper()
with input_col2:
    horizon_choice = st.selectbox("AI Forecast Horizon:", ["1 Week (5 Days)", "2 Weeks (10 Days)"])

if "1 Week" in horizon_choice:
    shift_days = 5
    calendar_days = 7
else:
    shift_days = 10
    calendar_days = 14

# --- MASTER ANALYSIS BLOCK ---
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
                num_headlines = 0
                
                if news:
                    news_texts = [item.get('title', '') for item in news[:5] if 'title' in item] 
                    num_headlines = len(news_texts)
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
            
            with tab1:
                st.markdown(f"### 📊 AI Forecast Outlook: {ticker}")
                col1, col2 = st.columns(2)
                col1.metric(label="Current Price", value=f"${current_price:.2f}")
                
                forecast_diff = forecast - current_price
                delta_display = f"+${forecast_diff:.2f}" if forecast_diff >= 0 else f"-${abs(forecast_diff):.2f}"
                col2.metric(label=f"AI Target ({horizon_choice})", value=f"${forecast:.2f}", delta=delta_display)

                st.markdown("---")
                st.markdown("#### 🛡️ Risk Management Calculator")
                r1, r2, r3 = st.columns(3)
                
                entry_input = r1.number_input("Entry Price ($)", value=float(current_price))
                sl_input = r2.number_input("Stop Loss ($)", value=float(current_price * 0.95))
                tp_input = r3.number_input("Target Price ($)", value=float(forecast))

                risk = entry_input - sl_input
                reward = tp_input - entry_input
                
                if risk > 0:
                    ratio = reward / risk
                    st.metric("Risk/Reward Ratio", f"{ratio:.2f} : 1")
                    if ratio >= 2.0:
                        st.success(f"✅ **GOOD TRADE:** Potential reward is {ratio:.1f}x your risk.")
                    elif ratio > 1.0:
                        st.warning(f"⚠️ **CAUTION:** Reward is less than 2x your risk ({ratio:.1f}x).")
                    else:
                        st.error("❌ **AVOID:** You are risking more than you stand to gain.")
                elif risk == 0:
                    st.error("Stop Loss cannot equal your Entry Price.")
                else:
                    st.error("Check your numbers: Stop Loss must be lower than Entry for a standard Long trade.")
                
                st.markdown(f"#### Price History & {horizon_choice} Forecast Path")
                
                # ---> NEW CHART DATA ADDED HERE <---
                # Calculate Moving Averages on the full dataset before slicing
                data['SMA_20'] = data['Close'].rolling(window=20).mean()
                data['SMA_50'] = data['Close'].rolling(window=50).mean()
                
                # Expand the view from 45 days to 90 days for better context
                recent_data = data.iloc[-90:]
                
                fig_daily = go.Figure(data=[go.Candlestick(
                    x=recent_data.index, open=recent_data['Open'], high=recent_data['High'],
                    low=recent_data['Low'], close=recent_data['Close'], name='Historical Price'
                )])
                
                # Add the 20-Day and 50-Day SMA lines to the chart
                fig_daily.add_trace(go.Scatter(
                    x=recent_data.index, y=recent_data['SMA_20'], 
                    mode='lines', name='20-Day SMA', line=dict(color='#29B6F6', width=1.5)
                ))
                fig_daily.add_trace(go.Scatter(
                    x=recent_data.index, y=recent_data['SMA_50'], 
                    mode='lines', name='50-Day SMA', line=dict(color='#FFA726', width=1.5)
                ))
                
                last_date = recent_data.index[-1]
                future_date = last_date + timedelta(days=calendar_days)
                line_color = '#00E676' if forecast_diff >= 0 else '#FF1744'
                
                fig_daily.add_trace(go.Scatter(
                    x=[last_date, future_date], y=[current_price, forecast],
                    mode='lines+markers', name='AI Forecast Path',
                    line=dict(color=line_color, width=3, dash='dash'),
                    marker=dict(size=8, symbol='circle')
                ))
                
                # Make the chart slightly taller to accommodate the extra data lines
                fig_daily.update_layout(
                    xaxis_rangeslider_visible=False, 
                    height=450, 
                    margin=dict(l=10, r=10, t=10, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_daily, use_container_width=True)
                
                st.markdown("#### AI Diagnostics & Reasoning")
                
                if forecast > current_price:
                    st.success("🤖 Mathematical Model: BULLISH")
                    st.write(f"**Specifics:** A Random Forest algorithm analyzed 1 year of daily technical data (Open, High, Low, Close, and Volume). Based on current momentum patterns, it projects the current price of **\${current_price:.2f}** will rise to **\${forecast:.2f}**. This is a mathematically predicted gain of **\${forecast_diff:.2f}** over the next {horizon_choice}.")
                else:
                    st.error("🤖 Mathematical Model: BEARISH")
                    st.write(f"**Specifics:** A Random Forest algorithm analyzed 1 year of daily technical data (Open, High, Low, Close, and Volume). Based on current weakness patterns, it projects the current price of **\${current_price:.2f}** will fall to **\${forecast:.2f}**. This is a mathematically predicted drop of **\${abs(forecast_diff):.2f}** over the next {horizon_choice}.")
                    
                if num_headlines > 0:
                    if bullish_score > bearish_score:
                        st.success("📰 Market Psychology: BULLISH")
                        st.write(f"**Specifics:** The **{engine_used}** Natural Language Processing model scanned the **{num_headlines} most recent news headlines** for {ticker}. It calculated a positive sentiment score of **{bullish_score:.2f}** compared to a negative score of **{bearish_score:.2f}**, indicating strong media optimism.")
                    elif bearish_score > bullish_score:
                        st.error("📰 Market Psychology: BEARISH")
                        st.write(f"**Specifics:** The **{engine_used}** Natural Language Processing model scanned the **{num_headlines} most recent news headlines** for {ticker}. It calculated a negative sentiment score of **{bearish_score:.2f}** compared to a positive score of **{bullish_score:.2f}**, indicating fear or media pessimism.")
                    else:
                        st.info("⚖️ Market Psychology: NEUTRAL")
                        st.write(f"**Specifics:** The **{engine_used}** Natural Language Processing model scanned the **{num_headlines} most recent news headlines** for {ticker}. The positive score (**{bullish_score:.2f}**) and negative score (**{bearish_score:.2f}**) perfectly offset each other, or no strong sentiment keywords were detected.")
                else:
                    st.info("⚖️ Market Psychology: UNKNOWN")
                    st.write("**Specifics:** No recent news headlines were found for this ticker to analyze.")

            with tab2:
                st.markdown(f"### ⚡ Intraday Momentum: {ticker}")
                
                if intraday.empty:
                    st.warning("Intraday data unavailable right now.")
                else:
                    intra_current = intraday['Close'].iloc[-1]
                    intra_high = intraday['High'].iloc[-len(intraday.loc[intraday.index.date == intraday.index.date[-1]]):].max()
                    intra_low = intraday['Low'].iloc[-len(intraday.loc[intraday.index.date == intraday.index.date[-1]]):].min()
                    intra_volume = intraday['Volume'].iloc[-1]
                    avg_volume = intraday['Volume'].mean()
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric(label="Live Price", value=f"${intra_current:.2f}")
                    c2.metric(label="Today's High", value=f"${intra_high:.2f}")
                    c3.metric(label="Today's Low", value=f"${intra_low:.2f}")
                    
                    st.markdown("#### Today's Session Position")
                    total_range = (intra_high - intra_low) if (intra_high - intra_low) != 0 else 1
                    position_pct = int(((intra_current - intra_low) / total_range) * 100)
                    position_pct = max(0, min(100, position_pct)) 
                    
                    st.progress(position_pct / 100)
                    st.caption(f"Price is sitting at {position_pct}% of today's total high-low bracket.")
                    
                    st.markdown("#### 5-Minute Live Momentum Chart")
                    fig_intra = go.Figure(data=[go.Candlestick(
                        x=intraday.index, open=intraday['Open'], high=intraday['High'],
                        low=intraday['Low'], close=intraday['Close'], name='5m Bars'
                    )])
                    fig_intra.update_layout(xaxis_rangeslider_visible=False, height=300, margin=dict(l=10,r=10,t=10,b=10))
                    st.plotly_chart(fig_intra, use_container_width=True)
                    
                    st.markdown("#### Trading Activity Alert")
                    if intra_volume > (avg_volume * 1.5):
                        st.success(f"🔥 VOLUME SURGE: {intra_volume:,} shares traded.")
                    elif intra_volume < (avg_volume * 0.5):
                        st.error(f"💤 SLEEPY VOLUME: {intra_volume:,} shares.")
                    else:
                        st.info(f"⚖️ NORMAL VOLUME: {intra_volume:,} shares.")

    except Exception as e:
        st.error("An error occurred during analysis.")
        st.error(f"System Log: {e}")
        
