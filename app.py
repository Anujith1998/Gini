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
st.set_page_config(
    page_title="ProQuant AI", 
    layout="centered", 
    page_icon="📈"
)
st.title("ProQuant AI 📈")
st.write("Advanced Market Analysis Engine")

# --- Token Handling ---
hf_token = st.secrets.get("HF_TOKEN", None)

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Controls")

if not hf_token:
    hf_token = st.sidebar.text_input(
        "HF Token (Optional)", 
        type="password"
    )

st.sidebar.markdown("---")
st.sidebar.header("📝 Watchlist")
wl_input = st.sidebar.text_input(
    "Tickers:", 
    value="AAPL, NVDA, TSLA, AMD, MSFT"
)

parsed_wl = [
    t.strip().upper() 
    for t in wl_input.split(",") 
    if t.strip()
]
watchlist_tuple = tuple(parsed_wl)

st.sidebar.markdown("---")
st.sidebar.header("🚨 Alerts")
alert_threshold = st.sidebar.number_input(
    "Alert Trigger (%)", 
    min_value=1.0, 
    max_value=50.0, 
    value=4.0
)

max_price_filter = st.sidebar.slider(
    "Max Price ($)", 
    min_value=10, 
    max_value=1000, 
    value=1000
)

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
        
        display_count = min(4, len(f_df))
        if display_count > 0:
            cols = st.columns(display_count)
            for i, row in enumerate(f_df.head(display_count).itertuples()):
                cols[i].metric(
                    label=row.ticker, 
                    value=f"${row.price:.2f}", 
                    delta=f"{row.change:+.2f}%"
                )

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
            data = stock.history(period="1y")      
            intraday = stock.history(period="5d", interval="5m")  
            
            if data.empty:
                status.update(label="Error", state="error")
                st.error("No market data found for this asset.")
            else:
                st.write("Training predictive models...")
                feats = ['Open', 'High', 'Low', 'Close', 'Volume']
                latest_data = data[feats].iloc[-1:]
                current_price = data['Close'].iloc[-1]
                
                # Model 1: 5 Days
                df_w1 = data.copy()
                df_w1['Target'] = df_w1['Close'].shift(-5)
                df_w1.dropna(inplace=True)
                m1 = RandomForestRegressor(n_estimators=100, random_state=42)
                m1.fit(df_w1[feats], df_w1['Target'])
                forecast_w1 = m1.predict(latest_data)[0]
                
                # Model 2: 10 Days
                df_w2 = data.copy()
                df_w2['Target'] = df_w2['Close'].shift(-10)
                df_w2.dropna(inplace=True)
                m2 = RandomForestRegressor(n_estimators=100, random_state=42)
                m2.fit(df_w2[feats], df_w2['Target'])
                forecast_w2 = m2.predict(latest_data)[0]
                
                # Model 3: 15 Days
                df_w3 = data.copy()
                df_w3['Target'] = df_w3['Close'].shift(-15)
                df_w3.dropna(inplace=True)
                m3 = RandomForestRegressor(n_estimators=100, random_state=42)
                m3.fit(df_w3[feats], df_w3['Target'])
                forecast_w3 = m3.predict(latest_data)[0]
                
                st.write("Scanning global sentiment...")
                news = get_resilient_news(stock, ticker)
                bull_s, bear_s, num_hl = 0, 0, 0
                engine = "TextBlob Engine"
                
                if news:
                    texts = [i.get('title', '') for i in news[:5] if 'title' in i]
                    num_hl = len(texts)
                    if texts:
                        if hf_token:
                            api_res = query_finbert_api(texts, hf_token)
                            if api_res and isinstance(api_res, list) and "error" not in api_res:
                                engine = "FinBERT Engine"
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
                
                status.update(label="Complete!", state="complete", expanded=False)

        if not data.empty:
            tab1, tab2, tab3 = st.tabs([
                "🏦 Multi-Week Forecast", 
                "⚡ Intraday Engine",
                "👥 AI Debate Room"
            ])
            
            with tab1:
                st.markdown(f"### 📊 Forecast Path: {ticker}")
                
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Current", f"${current_price:.2f}")
                mc2.metric("Week 1", f"${forecast_w1:.2f}", f"{forecast_w1 - current_price:+.2f}")
                mc3.metric("Week 2", f"${forecast_w2:.2f}", f"{forecast_w2 - forecast_w1:+.2f}")
                mc4.metric("Week 3", f"${forecast_w3:.2f}", f"{forecast_w3 - forecast_w2:+.2f}")

                st.markdown("---")
                
                data['SMA_20'] = data['Close'].rolling(window=20).mean()
                data['SMA_50'] = data['Close'].rolling(window=50).mean()
                
                h_std = data['Close'].pct_change().std()
                u_w1 = current_price * h_std * (7 ** 0.5)
                u_w2 = current_price * h_std * (14 ** 0.5)
                u_w3 = current_price * h_std * (21 ** 0.5)
                
                recent = data.iloc[-90:]
                
                fig = go.Figure(data=[go.Candlestick(
                    x=recent.index, open=recent['Open'], high=recent['High'],
                    low=recent['Low'], close=recent['Close'], name='Price'
                )])
                
                fig.add_trace(go.Scatter(x=recent.index, y=recent['SMA_20'], name='20 SMA', line=dict(color='#29B6F6', width=1)))
                fig.add_trace(go.Scatter(x=recent.index, y=recent['SMA_50'], name='50 SMA', line=dict(color='#FFA726', width=1)))
                
                l_date = recent.index[-1]
                t_line = [l_date, l_date+timedelta(7), l_date+timedelta(14), l_date+timedelta(21)]
                p_vals = [current_price, forecast_w1, forecast_w2, forecast_w3]
                
                fig.add_trace(go.Scatter(x=t_line, y=[current_price, forecast_w1+u_w1, forecast_w2+u_w2, forecast_w3+u_w3], mode='lines', showlegend=False, line=dict(width=0)))
                fig.add_trace(go.Scatter(x=t_line, y=[current_price, forecast_w1-u_w1, forecast_w2-u_w2, forecast_w3-u_w3], mode='lines', fill='tonexty', fillcolor='rgba(0,230,118,0.03)', name='Tunnel'))
                
                c_clr = '#00E676' if forecast_w3 >= current_price else '#FF1744'
                fig.add_trace(go.Scatter(x=t_line, y=p_vals, mode='lines+markers', name='AI Path', line=dict(color=c_clr, width=2, dash='dash')))
                fig.update_layout(xaxis_rangeslider_visible=False, height=400, margin=dict(l=5, r=5, t=5, b=5))
                st.plotly_chart(fig, use_container_width=True)
                
                m_df = pd.DataFrame({
                    "Horizon": ["Week 1", "Week 2", "Week 3"],
                    "Target Price": [f"${forecast_w1:.2f}", f"${forecast_w2:.2f}", f"${forecast_w3:.2f}"],
                    "Net Move": [f"{((forecast_w1/current_price)-1)*100:+.2f}%", f"{((forecast_w2/current_price)-1)*100:+.2f}%", f"{((forecast_w3/current_price)-1)*100:+.2f}%"],
                    "Bounds": [f"±${u_w1:.2f}", f"±${u_w2:.2f}", f"±${u_w3:.2f}"]
                })
                st.table(m_df)
                
                st.markdown("#### AI Diagnostics & Analysis")
                if (forecast_w3 - current_price) >= 0:
                    st.success("🤖 Cascade Status: NET BULLISH TREND DETECTED")
                    st.write(f"Random Forest setups model asset expansion toward ${forecast_w3:.2f}.")
                else:
                    st.error("🤖 Cascade Status: NET BEARISH TREND DETECTED")
                    st.write(f"Random Forest setups model asset reduction toward ${forecast_w3:.2f}.")
                    
                if num_hl > 0:
                    if bull_s > bear_s:
                        st.success(f"📰 Sentiment Layer ({engine}): BULLISH ({num_hl} Headlines)")
                    elif bear_s > bull_s:
                        st.error(f"📰 Sentiment Layer ({engine}): BEARISH ({num_hl} Headlines)")
                    else:
                        st.info(f"📰 Sentiment Layer ({engine}): NEUTRAL ({num_hl} Headlines)")
                else:
                    st.warning("⚠️ Sentiment Alert: No active headlines discovered.")

            with tab2:
                st.markdown(f"### ⚡ Intraday Diagnostics: {ticker}")
                if not intraday.empty:
                    i_cur = intraday['Close'].iloc[-1]
                    i_hi = intraday['High'].max()
                    i_lo = intraday['Low'].min()
                    
                    cc1, cc2, cc3 = st.columns(3)
                    cc1.metric("Live Execution", f"${i_cur:.2f}")
                    cc2.metric("Session High", f"${i_hi:.2f}")
                    cc3.metric("Session Low", f"${i_lo:.2f}")
                    
                    denom = (i_hi - i_lo) if (i_hi - i_lo) != 0 else 1
                    pos = int(((i_cur - i_lo) / denom) * 100)
                    st.progress(max(0, min(100, pos)) / 100)
                    st.caption(f"Current price position is at {pos}% of today's bracket.")

            with tab3:
                st.markdown("### 👥 Institutional AI Debate Arena")
                st.write("Consensus negotiation between algorithmic agents.")
                st.markdown("---")
                
                l_sma20 = float(data['SMA_20'].iloc[-1])
                sophia_b = (forecast_w3 >= current_price)
                marcus_b = (current_price >= l_sma20)
                elena_b = (bull_s >= bear_s) if num_hl > 0 else None
                
                votes = [sophia_b, marcus_b]
                if elena_b is not None:
                    votes.append(elena_b)
                b_votes = votes.count(True)
                t_votes = len(votes)
                
                # Agent 1
                st.markdown("**🧠 Sophia Vance | Quant Modeling:**")
                if sophia_b:
                    st.info(f"Predictive matrices tracking target alignment toward ${forecast_w3:.2f}. Long position supported.")
                else:
                    st.error(f"Structural weakness modeling price degradation toward ${forecast_w3:.2f}. Short entry triggered.")
                    
                # Agent 2
                st.markdown("**📈 Marcus Brody | Technical Charting:**")
                if marcus_b:
                    st.info(f"Price matches parameters above 20 SMA baseline of ${l_sma20:.2f}. Upward trend intact.")
                else:
                    st.error(f"20-Day SMA ceiling at ${l_sma20:.2f} limits expansion. Resistance is dominant.")
                    
                # Agent 3
                st.markdown("**📰 Elena Rostova | Macro Sentiment:**")
                if elena_b is True:
                    st.info(f"Scan logs indicate constructive media flows across {num_hl} feeds. Bullish bias.")
                elif elena_b is False:
                    st.error(f"Scan logs show structural anxiety across macro outlets. Protecting risk assets.")
                else:
                    st.warning("Data feeds silent on this index. Declaring defensive neutral position.")
                    
                st.markdown("---")
                st.markdown("#### ⚖️ Risk Committee Arbitration Summary")
                
                if b_votes == t_votes:
                    st.success("🎯 VERDICT: STRONG BUY (UNANIMOUS)\n\nComplete model convergence achieved.")
                elif b_votes > (t_votes / 2):
                    st.info(f"⚖️ VERDICT: TACTICAL BUY (MAJORITY)\n\nUpside parameters lead room by {b_votes} votes.")
                elif b_votes == (t_votes / 2):
                    st.warning("⚡ VERDICT: HOLD\n\nDeadlock encountered. Risk systems mandate staying flat.")
                else:
                    st.error("🚨 VERDICT: REJECT/SHORT\n\nDownside parameters dominate. Capital preservation active.")

    except Exception as e:
        st.error(f"Terminal Exception Error: {e}")
        
