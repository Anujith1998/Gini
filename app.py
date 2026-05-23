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

def query_finbert_api(text_list, token):
    api_url = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
    headers = {"Authorization": f"Bearer {token
    
