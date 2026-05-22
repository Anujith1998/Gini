import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from transformers import pipeline
import torch
import plotly.graph_objects as go

# --- App Configuration ---
st.set_page_config(page_title="ProQuant AI", layout="centered", page_icon="📈")
st.title("ProQuant AI 📈")
st.write("Advanced Market Analysis & Day Trading Engine")

# --- User Input ---
ticker = st.text_input("Enter Ticker Symbol (e.g. AAPL, TSLA):", "AAPL").upper()

if st.button("Run Master Analysis"):
    try:
        # --- THE ANIMATED BOOT-UP SEQUENCE ---
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
                if news:
                    sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
                    news_texts = [item.get('title', '') for item in news[:5] if 'title' in item] 
                    if news_texts:
                        results = sentiment_pipeline(news_texts)
                        for res in results:
                            if res['label'] == 'positive':
                                bullish_score += res['score']
                            elif res['label'] == 'negative':
                                bearish_score += res['score']
                
                # Close the loading animation cleanly
                status.update(label="Analysis Complete!", state="complete", expanded=False)

        # ----------------------------------------------------
        # THE DUAL-TAB INTERFACE
        # ----------------------------------------------------
        if not data.empty:
            tab1, tab2 = st.tabs(["🏦 Swing & AI Forecast", "⚡ Day Trading Engine"])
            
            # --- TAB 1: LONG TERM FORECAST ---
            with tab1:
                st.markdown(f"### 📊 AI Forecast Outlook: {ticker}")
                col1, col2 = st.columns(2)
                col1.metric(label="Current Price", value=f"${current_price:.2f}")
                
                forecast_diff = forecast - current_price
                if forecast_diff >= 0:
                    delta_display = f"${forecast_diff:.2f}"
                else:
                    delta_display = f"-${abs(forecast_diff):.2f}"
                col2.metric(label="5-Day AI Target", value=f"${forecast:.2f}", delta=delta_display)
                
                st.markdown("#### Price History (1 Year)")
                fig_daily = go.Figure(data=[go.Candlestick(x=data.index,
                                open=data['Open'], high=data['High'],
                                low=data['Low'], close=data['Close'], name='Daily')])
                fig_daily.update_layout(xaxis_rangeslider_visible=False, height=300, margin=dict(l=10,r=10,t=10,b=10))
                st.plotly_chart(fig_daily, use_container_width=True)
                
                st.markdown("#### AI Diagnostics")
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
                    st.warning("Intraday data unavailable. (Note: Live 5m data is restricted outside market hours or requires a premium data feed for some tickers).")
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
                    st.caption(f"Price is sitting at {position_pct}% of today's total high-low bracket. (Near 100% = Breakout, Near 0% = Support Crash)")
                    
                    st.markdown("#### 5-Minute Live Momentum Chart")
                    fig_intra = go.Figure(data=[go.Candlestick(x=intraday.index,
                                    open=intraday['Open'], high=intraday['High'],
                                    low=intraday['Low'], close=intraday['Close'], name='5m Bars')])
                    fig_intra.update_layout(xaxis_rangeslider_visible=False, height=300, margin=dict(l=10,r=10,t=10,b=10))
                    st.plotly_chart(fig_intra, use_container_width=True)
                    
                    st.markdown("#### Trading Activity Alert")
                    if intra_volume > (avg_volume * 1.5):
                        st.success(f"🔥 VOLUME SURGE: {intra_volume:,} shares traded in the last 5 mins! High volatility imminent.")
                    elif intra_volume < (avg_volume * 0.5):
                        st.error(f"💤 SLEEPY VOLUME: {intra_volume:,} shares. Price action is stagnant right now.")
                    else:
                        st.info(f"⚖️ NORMAL VOLUME: {intra_volume:,} shares. Steady session behavior.")

    except Exception as e:
        st.error("An error occurred during analysis.")
        st.error(f"System Log: {e}")
        
