import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from transformers import pipeline
import torch
import plotly.graph_objects as go

# --- App Configuration ---
st.set_page_config(page_title="ProQuant AI", layout="centered")
st.title("ProQuant AI 📈")
st.write("Advanced Market Analysis & Sentiment Engine")

# --- User Input ---
ticker = st.text_input("Enter Ticker Symbol (e.g. AAPL):", "AAPL").upper()

if st.button("Run Master Analysis"):
    try:
        with st.spinner("Initializing AI Engine & Fetching Market Data..."):
            # 1. Fetch Market Data
            stock = yf.Ticker(ticker)
            data = stock.history(period="1y")
            
            if data.empty:
                st.error("No data found. Please check the ticker symbol.")
            else:
                # 2. Predictive Mathematical Model (Random Forest)
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
                
                # 3. Market Psychology (Sentiment Analysis)
                news = stock.news
                bullish_score = 0
                bearish_score = 0
                
                if news:
                    sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
                    # Grab titles of the top 5 recent news articles
                    news_texts = [item.get('title', '') for item in news[:5] if 'title' in item] 
                    
                    if news_texts:
                        results = sentiment_pipeline(news_texts)
                        for res in results:
                            if res['label'] == 'positive':
                                bullish_score += res['score']
                            elif res['label'] == 'negative':
                                bearish_score += res['score']
                
                # 4. Final Outlook Metrics
                st.markdown("---")
                st.markdown(f"### 📊 Final Outlook: {ticker}")

                col1, col2 = st.columns(2)
                col1.metric(label="Current Price", value=f"${current_price:.2f}")
                forecast_diff = forecast - current_price
                col2.metric(label="5-Day AI Target", value=f"${forecast:.2f}", delta=f"${forecast_diff:.2f}")
                
                # 5. Interactive Candlestick Chart
                st.markdown("#### Price History")
                fig = go.Figure(data=[go.Candlestick(x=data.index,
                                open=data['Open'],
                                high=data['High'],
                                low=data['Low'],
                                close=data['Close'],
                                name='Price')])
                
                fig.update_layout(
                    xaxis_rangeslider_visible=False,
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 6. Engine Diagnostics
                st.markdown("#### Engine Diagnostics")
                if forecast > current_price:
                    st.success("🤖 Mathematical Model: BULLISH (Expecting Upward Trend)")
                else:
                    st.error("🤖 Mathematical Model: BEARISH (Expecting Downward Trend)")
                    
                if bullish_score > bearish_score:
                    st.success("📰 Market Psychology: BULLISH (Positive News Cycle)")
                elif bearish_score > bullish_score:
                    st.error("📰 Market Psychology: BEARISH (Negative News Cycle)")
                else:
                    st.info("⚖️ Market Psychology: NEUTRAL (No Extreme Sentiment)")

    except Exception as e:
        st.error("An error occurred during analysis.")
        st.error(f"System Log: {e}")
        
