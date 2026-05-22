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

if not hf_token:
    st.sidebar.header("🔑 AI Engine Settings")
    hf_token = st.sidebar.text_input("Hugging Face Token (Optional)", type="password", 
                                     help="Set up HF_TOKEN in your App Secrets to hide this box.")

# --- User Input ---
ticker = st.text_input("Enter Ticker Symbol (e.g. AAPL, TSLA):", "AAPL").upper()

def query_finbert_api(text_list, token):
    api_url = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.post(api_url, headers=headers, json={"inputs": text_list}, timeout=10)
        return response.json()
    except:
        return None

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
                st.write("🧠 Running predictive mathematical model...")
                df = data.copy()
                df['Target'] = df['Close'].shift(-5)
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
            
            # --- TAB 1: LONG TERM FORECAST ---
            with tab1:
                st.markdown(f"### 📊 AI Forecast Outlook: {ticker}")
                col1, col2 = st.columns(2)
                col1.metric(label="Current Price", value=f"${current_price:.2f}")
                
                forecast_diff = forecast - current_price
                if forecast_diff >= 0:
                    delta_display = f"+${forecast_diff:.2f}"
                else:
                    delta_display = f"-${abs(forecast_diff):.2f}"
                    
                col2.metric(label="5-Day AI Target", value=f"${forecast:.2f}", delta=delta_display)
                
                st.markdown("#### Price History & 5-Day Forecast Path")
                
                # Filter down to the last 45 days of history so the chart isn't too cramped to see the forecast
                recent_data = data.iloc[-45:]
                
                # Build base candlestick chart
                fig_daily = go.Figure(data=[go.Candlestick(
                    x=recent_data.index,
                    open=recent_data['Open'], high=recent_data['High'],
                    low=recent_data['Low'], close=recent_data['Close'], 
                    name='Historical Price'
                )])
                
                # Generate future dates for the timeline (skipping weekends roughly)
                last_date = recent_data.index[-1]
                future_date = last_date + timedelta(days=5)
                
                # Create the prediction line data arrays
                forecast_dates = [last_date, future_date]
                forecast_prices = [current_price, forecast]
                
                # Choose line color based on direction
                line_color = '#00E676' if forecast_diff >= 0 else '#FF1744'
                
                # Overlay the dotted forecast projection line
                fig_daily.add_trace(go.Scatter(
                    x=forecast_dates, 
                    y=forecast_prices,
                    mode='lines+markers',
                    name='AI Forecast Path',
                    line=dict(color=line_color, width=3, dash='dash'),
                    marker=dict(size=8, symbol='circle')
                ))
                
                fig_daily.update_layout(
                    xaxis_rangeslider_visible=False, 
                    height=350, 
                    margin=dict(l=10, r=10, t=10, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_daily, use_container_width=True)
                
                st.markdown("#### AI Diagnostics")
                st.caption(f"Sentiment Analysis active via: **{engine_used}**")
                
                if forecast > current_price:
                    st.success("🤖 Mathematical Model: BULLISH")
                else:
                    st.error("🤖 Mathematical Model: BEARISH")
                    
                if bullish_score > bearish_score:
                    st.success("📰 Market Psychology: BULLISH")
                elif bearish_score > bullish_score:
                    st.error("📰 Market Psychology: BEARISH")
                else:
                    st.info("⚖️ Market Psychology: NEUTRAL")

            # --- TAB 2: DAY TRADING ENGINE ---
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
                    fig_intra = go.Figure(data=[go.Candlestick(x=intraday.index,
                                    open=intraday['Open'], high=intraday['High'],
                                    low=intraday['Low'], close=intraday['Close'], name='5m Bars')])
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
        
