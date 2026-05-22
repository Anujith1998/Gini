import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from transformers import pipeline
import warnings
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

# --- APP UI SETUP ---
st.set_page_config(page_title="AI Quant Trader", page_icon="📈")
st.title("📈 ProQuant AI Trader")
st.write("Your personal machine learning and sentiment analysis engine.")

@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

sentiment_analyzer = load_sentiment_model()

# --- USER INPUT ---
st.markdown("### 1. Select Asset")
ticker = st.text_input("Enter a Stock Ticker (e.g., AAPL, NVDA, TSLA):", "AAPL").upper()

# --- THE ENGINE ---
if st.button("Run Master Analysis"):
    with st.spinner(f"Running neural networks for {ticker}..."):
        try:
            data = yf.download(ticker, period="4y", progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
                
            data['10_SMA'] = data['Close'].rolling(window=10).mean()
            data['Volatility'] = data['Close'].rolling(window=10).std()
            data['Daily_Return'] = data['Close'].pct_change()
            data['Target_Price'] = data['Close'].shift(-5)
            
            ml_data = data.dropna()
            features = ['Close', '10_SMA', 'Volatility', 'Daily_Return']
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(ml_data[features][:-100], ml_data['Target_Price'][:-100])
            
            latest_data = ml_data[features].iloc[-1].values.reshape(1, -1)
            forecast = model.predict(latest_data)[0]
            current_price = float(ml_data['Close'].iloc[-1])
            
            stock = yf.Ticker(ticker)
            news = stock.news
            bullish_score = 0
            bearish_score = 0
            
            if news:
                for article in news[:5]:
                    headline = article.get('title')
                    if headline:
                        result = sentiment_analyzer(headline)[0]
                        if result['label'] == 'positive':
                            bullish_score += 1
                        elif result['label'] == 'negative':
                            bearish_score += 1
                            
            st.markdown("---")
            st.markdown(f"### 📊 Final Outlook: {ticker}")
            
                    col1, col2 = st.columns(2)
        col1.metric(label="Current Price", value=f"${current_price:.2f}")
        forecast_diff = forecast - current_price
        col2.metric(label="5-Day AI Target", value=f"${forecast:.2f}", delta=f"${forecast_diff:.2f}")
        
        # --- CANDLESTICK CHART ---
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
        # -------------------------

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
        st.error("An error occurred. Check the ticker symbol and try again.")
    
