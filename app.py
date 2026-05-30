import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from textblob import TextBlob
import requests
import plotly.graph_objects as go
from datetime import timedelta
import xml.etree.ElementTree as ET

# --- Configuration ---
st.set_page_config(
    page_title="ProQuant AI", 
    layout="wide", 
    page_icon="📈"
)
st.title("ProQuant AI 📈")
st.write("Advanced Quantitative Analysis Engine")

# --- Token Handling ---
hf_token = st.secrets.get("HF_TOKEN", None)

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Controls")

if not hf_token:
    hf_token = st.sidebar.text_input("HF Token (Optional)", type="password")

st.sidebar.markdown("---")
st.sidebar.header("📝 Watchlist")
wl_input = st.sidebar.text_input("Tickers:", value="AAPL, NVDA, TSLA, AMD, MSFT")

parsed_wl = [t.strip().upper() for t in wl_input.split(",") if t.strip()]
watchlist_tuple = tuple(parsed_wl)

st.sidebar.markdown("---")
st.sidebar.header("🚨 Alerts")
alert_threshold = st.sidebar.number_input("Alert Trigger (%)", min_value=1.0, max_value=50.0, value=4.0)
max_price_filter = st.sidebar.slider("Max Price ($)", min_value=10, max_value=1000, value=1000)

# --- Functions ---
def resolve_company_name(query):
    cleaned = query.strip()
    if len(cleaned) <= 5 and cleaned.isalpha() and cleaned.isupper():
        return cleaned
    url = "https://query1.finance.yahoo.com/v1/finance/search"
    headers = {"User-Agent": "Mozilla/5.0"}
    params = {"q": cleaned, "lang": "en-US", "region": "US"}
    try:
        res = requests.get(url, params=params, headers=headers, timeout=5)
        data = res.json()
        if data.get('quotes'):
            return data['quotes'][0]['symbol'].upper()
    except:
        pass
    return cleaned.upper()

def get_resilient_news(stock_obj, ticker_str):
    news_data = []
    try:
        news_data = stock_obj.news
    except:
        pass
    if not news_data:
        try:
            url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker_str}"
            headers = {"User-Agent": "Mozilla/5.0"}
            res = requests.get(url, headers=headers, timeout=5)
            if res.status_code == 200:
                root = ET.fromstring(res.content)
                for item in root.findall('.//item'):
                    title = item.find('title')
                    if title is not None and title.text:
                        news_data.append({"title": title.text})
        except:
            pass
    return news_data if news_data else []

def query_finbert_api(text_list, token):
    url = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        res = requests.post(url, headers=headers, json={"inputs": text_list}, timeout=10)
        return res.json()
    except:
        return None

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

@st.cache_data(ttl=900, show_spinner=False)
def scan_market_leaders_fast(watchlist):
    scanned = []
    q_list = list(watchlist)
    if not q_list: 
        return pd.DataFrame()
    try:
        df = yf.download(q_list, period="5d", progress=False) 
        if df.empty: 
            return pd.DataFrame()
        for t in q_list:
            try:
                if isinstance(df.columns, pd.MultiIndex):
                    col = df['Close'][t] if t in df['Close'].columns else None
                else:
                    col = df['Close'] if 'Close' in df.columns else None
                
                if col is not None:
                    series = col.dropna()
                    if len(series) >= 2:
                        today = float(series.iloc[-1])
                        prev = float(series.iloc[-2])
                        change = ((today - prev) / prev) * 100
                        scanned.append({"ticker": t, "price": today, "change": change})
            except: 
                continue
    except: 
        pass
    return pd.DataFrame(scanned)

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv().encode('utf-8')

# --- UI Momentum Watchlist ---
st.markdown("### 🔥 Live Momentum Watchlist")
if not parsed_wl:
    st.warning("Enter tickers in the sidebar controls.")
else:
    with st.spinner("Scanning..."):
        scanner_df = scan_market_leaders_fast(watchlist_tuple)
    if not scanner_df.empty:
        f_df = scanner_df[scanner_df['price'] <= max_price_filter]
        f_df = f_df.sort_values(by="change", ascending=False)
        
        for _, row in f_df.iterrows():
            if abs(row['change']) >= alert_threshold:
                st.toast(f"{row['ticker']} shifted {row['change']:+.2f}%")
        
        display_count = min(5, len(f_df))
        if display_count > 0:
            cols = st.columns(display_count)
            for i, row in enumerate(f_df.head(display_count).itertuples()):
                cols[i].metric(label=row.ticker, value=f"${row.price:.2f}", delta=f"{row.change:+.2f}%")

st.markdown("---")
st.markdown("### 🔍 Run Deep AI Multi-Week Analysis")
user_input = st.text_input("Company Name or Ticker:", "Apple")

if st.button("Run Master Multi-Week Analysis"):
    try: 
        with st.status("Running Engine...", expanded=True) as status:
            st.write("Resolving asset identifier...")
            ticker = resolve_company_name(user_input)
            st.write(f"Target Confirmed: {ticker}")
            
            stock = yf.Ticker(ticker)
            data = stock.history(period="2y") # Pulled 2 years to ensure technical indicators and shifts have plenty of row runway
            intraday = stock.history(period="5d", interval="5m")  
            
            if data.empty:
                status.update(label="Error", state="error")
                st.error("No market data found for this asset.")
            else:
                st.write("Generating and Engineering Advanced Features...")
                
                # Tech Feature calculations
                data['SMA_20'] = data['Close'].rolling(window=20).mean()
                data['SMA_50'] = data['Close'].rolling(window=50).mean()
                data['RSI'] = calculate_rsi(data['Close'])
                
                exp1 = data['Close'].ewm(span=12, adjust=False).mean()
                exp2 = data['Close'].ewm(span=26, adjust=False).mean()
                data['MACD'] = exp1 - exp2
                data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
                
                data['BB_Mid'] = data['Close'].rolling(window=20).mean()
                data['BB_Std'] = data['Close'].rolling(window=20).std()
                data['BB_Upper'] = data['BB_Mid'] + (data['BB_Std'] * 2)
                data['BB_Lower'] = data['BB_Mid'] - (data['BB_Std'] * 2)
                
                # Strip out initial NaN values generated by technical indicators
                data.dropna(subset=['SMA_20', 'SMA_50', 'RSI', 'MACD', 'BB_Upper'], inplace=True)
                
                feats = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal_Line', 'BB_Upper', 'BB_Lower']
                latest_features = data[feats].iloc[-1:]
                current_price = float(data['Close'].iloc[-1])
                last_date = data.index[-1]
                
                # Format UI Target Dates
                d_w1 = (last_date + timedelta(days=7)).strftime('%b %d')
                d_w2 = (last_date + timedelta(days=14)).strftime('%b %d')
                d_w3 = (last_date + timedelta(days=21)).strftime('%b %d')
                d_w4 = (last_date + timedelta(days=28)).strftime('%b %d')
                
                recent = data.iloc[-90:].copy()
                
                st.write("Training Stationary Return Models (Anti-Overfit)...")
                
                # Model 1: 5 Days (Week 1 Returns)
                df_w1 = data.copy()
                df_w1['Target_Return'] = np.log(df_w1['Close'].shift(-5) / df_w1['Close'])
                df_w1.dropna(subset=['Target_Return'], inplace=True)
                m1 = RandomForestRegressor(n_estimators=150, max_depth=6, min_samples_leaf=4, random_state=42)
                m1.fit(df_w1[feats], df_w1['Target_Return'])
                pred_return_w1 = m1.predict(latest_features)[0]
                forecast_w1 = current_price * np.exp(pred_return_w1)
                
                # Model 2: 10 Days (Week 2 Returns)
                df_w2 = data.copy()
                df_w2['Target_Return'] = np.log(df_w2['Close'].shift(-10) / df_w2['Close'])
                df_w2.dropna(subset=['Target_Return'], inplace=True)
                m2 = RandomForestRegressor(n_estimators=150, max_depth=6, min_samples_leaf=4, random_state=42)
                m2.fit(df_w2[feats], df_w2['Target_Return'])
                pred_return_w2 = m2.predict(latest_features)[0]
                forecast_w2 = current_price * np.exp(pred_return_w2)
                
                # Model 3: 15 Days (Week 3 Returns)
                df_w3 = data.copy()
                df_w3['Target_Return'] = np.log(df_w3['Close'].shift(-15) / df_w3['Close'])
                df_w3.dropna(subset=['Target_Return'], inplace=True)
                m3 = RandomForestRegressor(n_estimators=150, max_depth=6, min_samples_leaf=4, random_state=42)
                m3.fit(df_w3[feats], df_w3['Target_Return'])
                pred_return_w3 = m3.predict(latest_features)[0]
                forecast_w3 = current_price * np.exp(pred_return_w3)
                
                # Model 4: 20 Days (Week 4 Returns)
                df_w4 = data.copy()
                df_w4['Target_Return'] = np.log(df_w4['Close'].shift(-20) / df_w4['Close'])
                df_w4.dropna(subset=['Target_Return'], inplace=True)
                m4 = RandomForestRegressor(n_estimators=150, max_depth=6, min_samples_leaf=4, random_state=42)
                m4.fit(df_w4[feats], df_w4['Target_Return'])
                pred_return_w4 = m4.predict(latest_features)[0]
                forecast_w4 = current_price * np.exp(pred_return_w4)
                
                st.write("Scanning global sentiment parameters...")
                news = get_resilient_news(stock, ticker)
                bull_s, bear_s, num_hl = 0, 0, 0
                
                if news:
                    texts = [i.get('title', '') for i in news[:5] if 'title' in i]
                    num_hl = len(texts)
                    if texts:
                        if hf_token:
                            api_res = query_finbert_api(texts, hf_token)
                            if api_res and isinstance(api_res, list) and "error" not in api_res:
                                for r_list in api_res:
                                    if isinstance(r_list, list):
                                        for item in r_list:
                                            lbl = item.get('label')
                                            scr = item.get('score', 0)
                                            if lbl == 'positive': bull_s += scr
                                            elif lbl == 'negative': bear_s += scr
                            else:
                                hf_token = None  
                        if not hf_token:
                            for txt in texts:
                                b_obj = TextBlob(txt)
                                pol = b_obj.sentiment.polarity
                                if pol > 0.1: bull_s += pol
                                elif pol < -0.1: bear_s += abs(pol)
                
                status.update(label="Analysis Complete!", state="complete", expanded=False)

        if not data.empty:
            csv_data = convert_df_to_csv(data)
            st.download_button(
                label=f"📥 Download {ticker} Full Dataset (CSV)",
                data=csv_data,
                file_name=f"{ticker}_ProQuant_Report.csv",
                mime="text/csv",
            )
            
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "🏦 Multi-Week Forecast", 
                "⚡ Intraday Engine",
                "👥 AI Debate Room",
                "📊 Financial Health",
                "🎯 Institutional Insights",
                "👔 Wall Street Consensus"
            ])
            
            with tab1:
                st.markdown(f"### 📊 4-Week Stationary Forecast Path: {ticker}")
                
                mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                mc1.metric(f"Current ({last_date.strftime('%b %d')})", f"${current_price:.2f}")
                mc2.metric(f"Wk 1 ({d_w1})", f"${forecast_w1:.2f}", f"{forecast_w1 - current_price:+.2f}")
                mc3.metric(f"Wk 2 ({d_w2})", f"${forecast_w2:.2f}", f"{forecast_w2 - forecast_w1:+.2f}")
                mc4.metric(f"Wk 3 ({d_w3})", f"${forecast_w3:.2f}", f"{forecast_w3 - forecast_w2:+.2f}")
                mc5.metric(f"Wk 4 ({d_w4})", f"${forecast_w4:.2f}", f"{forecast_w4 - forecast_w3:+.2f}")

                st.markdown("---")
                
                h_std = data['Close'].pct_change().std()
                u_w1 = current_price * h_std * (7 ** 0.5)
                u_w2 = current_price * h_std * (14 ** 0.5)
                u_w3 = current_price * h_std * (21 ** 0.5)
                u_w4 = current_price * h_std * (28 ** 0.5)
                
                fig = go.Figure(data=[go.Candlestick(
                    x=recent.index, open=recent['Open'], high=recent['High'],
                    low=recent['Low'], close=recent['Close'], name='Price'
                )])
                
                fig.add_trace(go.Scatter(x=recent.index, y=recent['BB_Upper'], name='Upper Bollinger', line=dict(color='rgba(158, 158, 158, 0.4)', width=1, dash='dot')))
                fig.add_trace(go.Scatter(x=recent.index, y=recent['BB_Lower'], name='Lower Bollinger', line=dict(color='rgba(158, 158, 158, 0.4)', width=1, dash='dot')))
                fig.add_trace(go.Scatter(x=recent.index, y=recent['SMA_20'], name='20 SMA', line=dict(color='#29B6F6', width=1)))
                fig.add_trace(go.Scatter(x=recent.index, y=recent['SMA_50'], name='50 SMA', line=dict(color='#FFA726', width=1)))
                
                t_line = [last_date, last_date+timedelta(7), last_date+timedelta(14), last_date+timedelta(21), last_date+timedelta(28)]
                p_vals = [current_price, forecast_w1, forecast_w2, forecast_w3, forecast_w4]
                
                fig.add_trace(go.Scatter(
                    x=t_line, y=[current_price, forecast_w1+u_w1, forecast_w2+u_w2, forecast_w3+u_w3, forecast_w4+u_w4], 
                    mode='lines', showlegend=False, line=dict(width=0)
                ))
                fig.add_trace(go.Scatter(
                    x=t_line, y=[current_price, forecast_w1-u_w1, forecast_w2-u_w2, forecast_w3-u_w3, forecast_w4-u_w4], 
                    mode='lines', fill='tonexty', fillcolor='rgba(0,230,118,0.03)', name='Tunnel'
                ))
                
                c_clr = '#00E676' if forecast_w4 >= current_price else '#FF1744'
                fig.add_trace(go.Scatter(x=t_line, y=p_vals, mode='lines+markers', name='AI Return Path', line=dict(color=c_clr, width=2, dash='dash')))
                fig.update_layout(xaxis_rangeslider_visible=False, height=450, margin=dict(l=5, r=5, t=5, b=5))
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                st.markdown(f"### ⚡ Intraday Diagnostics: {ticker}")
                if not intraday.empty:
                    latest_trading_day = intraday.index[-1].date()
                    day_data = intraday[intraday.index.date == latest_trading_day].copy()
                    
                    if not day_data.empty:
                        i_cur = day_data['Close'].iloc[-1]
                        i_hi = day_data['High'].max()
                        i_lo = day_data['Low'].min()
                        
                        day_data['Typical_Price'] = (day_data['High'] + day_data['Low'] + day_data['Close']) / 3
                        day_data['VWAP'] = (day_data['Typical_Price'] * day_data['Volume']).cumsum() / day_data['Volume'].cumsum()
                        vwap_val = day_data['VWAP'].iloc[-1]
                        
                        cc1, cc2, cc3, cc4 = st.columns(4)
                        cc1.metric("Live Execution", f"${i_cur:.2f}")
                        cc2.metric("Session High", f"${i_hi:.2f}")
                        cc3.metric("Session Low", f"${i_lo:.2f}")
                        vwap_delta = i_cur - vwap_val
                        cc4.metric("Session VWAP", f"${vwap_val:.2f}", f"{vwap_delta:+.2f} gap")
                        
                        denom = (i_hi - i_lo) if (i_hi - i_lo) != 0 else 1
                        pos = int(((i_cur - i_lo) / denom) * 100)
                        st.progress(max(0, min(100, pos)) / 100)
                        
                        fig_i = go.Figure()
                        fig_i.add_trace(go.Scatter(x=day_data.index, y=day_data['Close'], mode='lines', name='Price', line=dict(color='#29B6F6', width=2.5)))
                        fig_i.add_trace(go.Scatter(x=day_data.index, y=day_data['VWAP'], mode='lines', name='VWAP Tracker', line=dict(color='#FFA726', width=2, dash='dot')))
                        fig_i.update_layout(height=350, margin=dict(l=5, r=5, t=10, b=5), xaxis_rangeslider_visible=False)
                        st.plotly_chart(fig_i, use_container_width=True)

            with tab3:
                st.markdown(f"### 🗳️ 8-Agent AI Consensus Scoreboard ({ticker})")
                
                l_sma20 = float(data['SMA_20'].iloc[-1])
                l_sma50 = float(data['SMA_50'].iloc[-1])
                l_rsi = float(data['RSI'].iloc[-1]) if not data['RSI'].isna().all() else 50.0
                l_macd = float(data['MACD'].iloc[-1])
                l_signal = float(data['Signal_Line'].iloc[-1])
                
                v1 = (forecast_w4 >= current_price)
                v2 = (current_price >= l_sma20)
                v3 = (bull_s >= bear_s) if num_hl > 0 else True
                v4 = (current_price >= l_sma50)
                v5 = (l_rsi <= 65.0)  
                v6 = (forecast_w1 >= current_price)
                v7 = (l_macd >= l_signal) 
                v8 = (current_price <= data['BB_Upper'].iloc[-1]) 
                
                votes = [v1, v2, v3, v4, v5, v6, v7, v8]
                bull_votes = votes.count(True)
                
                v_df = pd.DataFrame({
                    "Algorithmic Voter Node": [
                        "Agent 1: Cascade ML Return Matrix (4-Week)",
                        "Agent 2: Short Trend Engine (20-Day SMA)",
                        "Agent 3: Media Sentiment Array",
                        "Agent 4: Macro Baseline (50-Day SMA)",
                        "Agent 5: Momentum Variance Check (RSI)",
                        "Agent 6: Immediate Vector (1-Week Return Path)",
                        "Agent 7: Trend Reversal Crossover (MACD)",
                        "Agent 8: Volatility Overbought Check (Bollinger)"
                    ],
                    "Stance": ["🟢 BULL" if v else "🔴 BEAR" for v in votes]
                })
                st.table(v_df)
                
                if bull_votes >= 6: st.success(f"🎯 **STRONG BUY** ({bull_votes}/8 Bulls)")
                elif bull_votes == 5: st.info(f"⚖️ **TACTICAL ACCUMULATE** ({bull_votes}/8 Bulls)")
                elif bull_votes == 4: st.warning(f"⚡ **EQUAL WEIGHT / HOLD** ({bull_votes}/8 Bulls)")
                elif bull_votes >= 2: st.error(f"🚨 **TACTICAL REDUCE / SHORT** ({bull_votes}/8 Bulls)")
                else: st.error(f"💀 **STRONG LIQUIDATE / SHORT** ({bull_votes}/8 Bulls)")

                st.markdown("---")
                st.markdown("#### 📉 Price Trajectory Overlay")
                fig_p = go.Figure()
                fig_p.add_trace(go.Scatter(x=recent.index, y=recent['Close'], mode='lines', name='Historical Price', line=dict(color='#29B6F6', width=2)))
                fig_p.add_trace(go.Scatter(x=t_line, y=p_vals, mode='lines+markers', name='AI Target Path', line=dict(color=c_clr, width=2.5, dash='dash')))
                fig_p.update_layout(yaxis=dict(title="Stock Price ($)"), height=250, margin=dict(l=5, r=5, t=10, b=5), xaxis_rangeslider_visible=False)
                st.plotly_chart(fig_p, use_container_width=True)

            with tab4:
                st.markdown(f"### 📊 Fundamental Corporate Financials: {ticker}")
                try:
                    financials = stock.financials
                    if not financials.empty:
                        idx = financials.index
                        rev_label = [i for i in idx if 'Total Revenue' in i or 'Revenue' in i]
                        net_label = [i for i in idx if 'Net Income' in i]
                        if rev_label and net_label:
                            f_plot = financials.loc[[rev_label[0], net_label[0]]].T
                            f_plot.index = pd.to_datetime(f_plot.index).strftime('%Y')
                            f_plot.columns = ['Total Revenue', 'Net Income']
                            st.bar_chart(f_plot)
                            st.dataframe(financials.dropna(how='all').head(10), use_container_width=True)
                        else:
                            st.info("Incomplete financial row mappings found for this asset.")
                except Exception as ex:
                    st.caption(f"Financial retrieval bypassed: {ex}")

            with tab5:
                st.markdown(f"### 🎯 Institutional Earnings Analysis: {ticker}")
                try:
                    earn_hist = stock.get_earnings_history()
                    if earn_hist is not None and not earn_hist.empty:
                        df_earn = earn_hist.head(8).copy()
                        df_earn.index = pd.to_datetime(df_earn.index)
                        df_earn = df_earn.sort_index()
                        
                        fig_e = go.Figure()
                        fig_e.add_trace(go.Bar(x=df_earn.index, y=df_earn['epsEstimate'], name='EPS Estimate', marker_color='#FFA726'))
                        fig_e.add_trace(go.Bar(x=df_earn.index, y=df_earn['epsActual'], name='EPS Actual', marker_color='#29B6F6'))
                        
                        fig_e.update_layout(barmode='group', height=300, margin=dict(l=5, r=5, t=10, b=5))
                        st.plotly_chart(fig_e, use_container_width=True)
                except:
                    st.info("Earnings matrix interface limits reached for this ticker configuration.")
                    
            with tab6:
                st.markdown(f"### 👔 Wall Street Consensus Ratings: {ticker}")
                try:
                    recs = stock.recommendations
                    if recs is not None and not recs.empty:
                        st.write("Recent institutional upgrades and downgrades:")
                        clean_recs = recs.tail(10).copy()
                        if 'period' in clean_recs.columns:
                            clean_recs = clean_recs.drop(columns=['period'])
                        st.dataframe(clean_recs, use_container_width=True)
                    else:
                        st.warning("No recent Wall Street recommendations found for this ticker.")
                except:
                    st.info("Analyst rating module unavailable for this asset.")

    except Exception as e:
        st.error(f"Terminal Exception Error: {e}")
