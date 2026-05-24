import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from textblob import TextBlob
import requests
import plotly.graph_objects as go
from datetime import timedelta

st.set_page_config(page_title="ProQuant AI", layout="centered", page_icon="📈")
st.title("ProQuant AI 📈")

st.sidebar.header("⚙️ Settings")
wl_input = st.sidebar.text_input("Watchlist:", "AAPL, NVDA, TSLA, AMD")
wl_list = [t.strip().upper() for t in wl_input.split(",") if t.strip()]

# --- 1. LIVE MOMENTUM DASHBOARD (Runs Automatically) ---
st.markdown("### 🔥 Live Momentum Watchlist")
if wl_list:
    cols = st.columns(min(4, len(wl_list)))
    for i, ticker in enumerate(wl_list[:4]):
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(period="5d")
            if len(hist) >= 2:
                p_now = float(hist['Close'].iloc[-1])
                p_prev = float(hist['Close'].iloc[-2])
                pct_change = ((p_now - p_prev) / p_prev) * 100
                cols[i].metric(ticker, f"${p_now:.2f}", f"{pct_change:+.2f}%")
        except:
            cols[i].metric(ticker, "N/A", "N/A")

st.markdown("---")

# --- 2. MASTER AI ANALYSIS ENGINE ---
st.markdown("### 🔍 Run Deep AI Analysis")
target_ticker = st.text_input("Enter Ticker Symbol:", "AAPL").upper().strip()

if st.button("Run Master Analysis"):
    with st.spinner(f"Running multi-agent analysis on {target_ticker}..."):
        stock = yf.Ticker(target_ticker)
        data = stock.history(period="1y")
        intraday = stock.history(period="5d", interval="5m")
        
        if len(data) < 50:
            st.error("Not enough historical data for this asset.")
            st.stop()
            
        # Core Math & Data Preparation
        current_price = float(data['Close'].iloc[-1])
        data['SMA_20'] = data['Close'].rolling(20).mean()
        data['SMA_50'] = data['Close'].rolling(50).mean()
        recent = data.iloc[-90:].copy()
        
        feats = ['Open', 'High', 'Low', 'Close', 'Volume']
        latest_data = data[feats].iloc[-1:]
        
        # ML Engine 1: Week 1
        df_1 = data.copy()
        df_1['T'] = df_1['Close'].shift(-5)
        df_1.dropna(inplace=True)
        m1 = RandomForestRegressor(n_estimators=50, random_state=42)
        m1.fit(df_1[feats], df_1['T'])
        w1_pred = float(m1.predict(latest_data)[0])
        
        # ML Engine 2: Week 2
        df_2 = data.copy()
        df_2['T'] = df_2['Close'].shift(-10)
        df_2.dropna(inplace=True)
        m2 = RandomForestRegressor(n_estimators=50, random_state=42)
        m2.fit(df_2[feats], df_2['T'])
        w2_pred = float(m2.predict(latest_data)[0])
        
        # ML Engine 3: Week 3
        df_3 = data.copy()
        df_3['T'] = df_3['Close'].shift(-15)
        df_3.dropna(inplace=True)
        m3 = RandomForestRegressor(n_estimators=50, random_state=42)
        m3.fit(df_3[feats], df_3['T'])
        w3_pred = float(m3.predict(latest_data)[0])
        
        # Sentiment Engine
        bull_s = 0
        bear_s = 0
        try:
            news_items = stock.news[:5]
            for item in news_items:
                title = item.get('title', '')
                pol = TextBlob(title).sentiment.polarity
                if pol > 0.05:
                    bull_s += 1
                elif pol < -0.05:
                    bear_s += 1
        except:
            pass

        # --- TABS RENDERING ---
        t1, t2, t3 = st.tabs(["🏦 Multi-Week Forecast", "⚡ Intraday", "👥 Debate Room"])
        
        with t1:
            st.markdown(f"### 📊 AI Forecast Path: {target_ticker}")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Current", f"${current_price:.2f}")
            c2.metric("Week 1", f"${w1_pred:.2f}", f"{w1_pred-current_price:+.2f}")
            c3.metric("Week 2", f"${w2_pred:.2f}", f"{w2_pred-w1_pred:+.2f}")
            c4.metric("Week 3", f"${w3_pred:.2f}", f"{w3_pred-w2_pred:+.2f}")
            
            # Master Graph
            fig1 = go.Figure()
            
            trace_price = go.Scatter(x=recent.index, y=recent['Close'], name='Price', line=dict(color='#29B6F6', width=2))
            fig1.add_trace(trace_price)
            
            trace_s20 = go.Scatter(x=recent.index, y=recent['SMA_20'], name='20-SMA', line=dict(color='blue', width=1))
            fig1.add_trace(trace_s20)
            
            last_date = recent.index[-1]
            future_dates = [last_date, last_date+timedelta(7), last_date+timedelta(14), last_date+timedelta(21)]
            future_vals = [current_price, w1_pred, w2_pred, w3_pred]
            
            path_color = '#00E676' if w3_pred >= current_price else '#FF1744'
            trace_future = go.Scatter(x=future_dates, y=future_vals, mode='lines+markers', name='AI Path', line=dict(color=path_color, dash='dash', width=2))
            fig1.add_trace(trace_future)
            
            fig1.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig1, use_container_width=True)
            
            if w3_pred >= current_price:
                st.success("🤖 Net 3-Week Status: BULLISH PATH DETECTED")
            else:
                st.error("🤖 Net 3-Week Status: BEARISH PATH DETECTED")

        with t2:
            st.markdown(f"### ⚡ Live Intraday Range: {target_ticker}")
            if not intraday.empty:
                i_cur = intraday['Close'].iloc[-1]
                i_hi = intraday['High'].max()
                i_lo = intraday['Low'].min()
                
                ic1, ic2, ic3 = st.columns(3)
                ic1.metric("Live Price", f"${i_cur:.2f}")
                ic2.metric("Session High", f"${i_hi:.2f}")
                ic3.metric("Session Low", f"${i_lo:.2f}")
                
                spread = i_hi - i_lo if i_hi != i_lo else 1
                percent_pos = int(((i_cur - i_lo) / spread) * 100)
                clamped_pos = max(0, min(100, percent_pos))
                st.progress(clamped_pos / 100)
                st.caption(f"Current price is at the {clamped_pos}th percentile of today's total range.")

        with t3:
            st.markdown("### 🗳️ 6-Agent AI Consensus Scoreboard")
            
            sma20_val = data['SMA_20'].iloc[-1]
            sma50_val = data['SMA_50'].iloc[-1]
            open_val = data['Open'].iloc[-1]
            
            # The 6 Agents Vote
            v1 = w3_pred >= current_price
            v2 = current_price >= sma20_val
            v3 = bull_s >= bear_s
            v4 = current_price >= sma50_val
            v5 = current_price >= open_val
            v6 = w1_pred >= current_price
            
            votes = [v1, v2, v3, v4, v5, v6]
            bull_count = sum(votes)
            stances = ["🟢 BULL" if v else "🔴 BEAR" for v in votes]
            
            df_votes = pd.DataFrame({
                "Agent Array": ["Agent 1: 3-Week ML", "Agent 2: 20-SMA", "Agent 3: Sentiment", "Agent 4: 50-SMA", "Agent 5: Intraday Pivot", "Agent 6: 1-Week ML"],
                "Stance": stances
            })
            st.table(df_votes)
            
            st.markdown("#### ⚖️ Final Arbitration")
            if bull_count >= 4:
                st.success(f"🎯 **BULLISH CONSENSUS** ({bull_count}/6 Agents)")
            elif bull_count == 3:
                st.warning(f"⚡ **NEUTRAL / HOLD** ({bull_count}/6 Agents)")
            else:
                st.error(f"🚨 **BEARISH CONSENSUS** ({bull_count}/6 Agents)")

            st.markdown("---")
            st.markdown("#### 📈 Forward AI Consensus Path")
            st.write("Projecting how the Consensus Score will shift over the next 3 weeks based on ML price targets:")
            
            # Projecting Future Votes based on ML Targets
            f_w1 = sum([w3_pred>=w1_pred, w1_pred>=sma20_val, v3, w1_pred>=sma50_val, w1_pred>=current_price, w2_pred>=w1_pred])
            f_w2 = sum([w3_pred>=w2_pred, w2_pred>=sma20_val, v3, w2_pred>=sma50_val, w2_pred>=w1_pred, w3_pred>=w2_pred])
            f_w3 = sum([True, w3_pred>=sma20_val, v3, w3_pred>=sma50_val, w3_pred>=w2_pred, True])
            
            f_scores = [bull_count, f_w1, f_w2, f_w3]
            
            fig_cons = go.Figure()
            cons_trace = go.Scatter(x=future_dates, y=f_scores, mode='lines+markers', name='Consensus Path', line=dict(color=path_color, width=3, dash='dash'))
            fig_cons.add_trace(cons_trace)
            
            # Add safe background shading
            fig_cons.add_hrect(y0=3.5, y1=6.5, fillcolor="rgba(0,230,118,0.1)", layer="below", line_width=0)
            fig_cons.add_hrect(y0=-0.5, y1=2.5, fillcolor="rgba(255,23,68,0.1)", layer="below", line_width=0)
            
            fig_cons.update_layout(yaxis=dict(range=[-0.5, 6.5], tickvals=[0,1,2,3,4,5,6], title="Bull Votes"), height=300, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_cons, use_container_width=True)
            
