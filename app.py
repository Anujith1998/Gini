import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from textblob import TextBlob
import requests
import plotly.graph_objects as go
from datetime import timedelta
import xml.etree.ElementTree as ET

# --- Configuration ---
st.set_page_config(page_title="ProQuant AI", layout="centered", page_icon="📈")
st.title("ProQuant AI 📈")

# [ ... Keep your existing imports, sidebar, and resolve_company_name functions here ... ]
# (Ensure your indentation matches exactly as shown below)

# --- Updated Tab 3 Section ---
# Replace your existing Tab 3 logic with this fully corrected version:

with tab3:
    st.markdown("### 🗳️ 6-Agent AI Consensus Scoreboard")
    
    # ... (Keep your existing vote logic here) ...
    
    # Ensure this section is correctly closed and indented:
    s_w1 = sum([
        forecast_w3 >= forecast_w1, forecast_w1 >= l_sma20, 
        v3, forecast_w1 >= l_sma50, forecast_w1 >= current_price, 
        forecast_w2 >= forecast_w1
    ])
    
    # Corrected Ternary Logic (must include else)
    fc_clr = '#00E676' if s_w3 >= s_w1 else '#FF1744'
    
    # Corrected Parentheses usage for go.Scatter
    fc_line = go.Scatter(
        x=f_dates, 
        y=f_scores, 
        mode='lines+markers', 
        name='AI Forecast Path',
        line=dict(color=fc_clr, width=2.5, dash='dash')
    )
    
    # ... (Rest of your plotly code) ...
    
