import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from textblob import TextBlob
import requests
import plotly.graph_objects as go
from datetime import timedelta
import xml.etree.ElementTree as ET

# --- App Configuration ---
st.set_page_config(page_title="ProQuant AI", layout="centered", page_icon="📈")
st.title("ProQuant AI 📈")
st.write("Advanced Market Analysis & Day Trading Engine")

# --- Automated Token Detection ---
hf_token = st.secrets.get("HF_TOKEN", None)

# --- SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ Dashboard Controls")

if not hf_token:
    hf_token = st.sidebar.text_input(
        "Hugging Face Token (Optional)", 
        type="password", 
        help="Set up HF_TOKEN in your App Secrets to hide this box."
    )

st.sidebar.markdown("---")
st.sidebar.header("📝 Personal Watchlist")
user_watchlist_input = st.sidebar.text_input(
    "Enter tickers (comma-separated):", 
    value="AAPL, NVDA, TSLA, AMD, MSFT, AMZN, META, GOOGL"
)

parsed_watchlist = [
    ticker.strip().upper() 
    for ticker in user_watchlist_input.split(",") 
    if ticker.strip()
]
watchlist_tuple = tuple(parsed_watchlist)

st.sidebar.markdown("---")
st.sidebar.header("🚨 Alert Settings")
alert_threshold = st.sidebar.number_input(
    "Alert me if a stock moves more than (%):", 
    min_value=1.0, 
    max_value=50.0, 
    value=4.0, 
    step=0.5
)

st.sidebar.markdown("---")
max_price_filter = st.sidebar.slider(
    "Filter Watchlist by Max Price ($):", 
    min_value=10, 
    max_value=1000, 
    value=1000, 
    step=10
)

# --- SMART RESOLVER FUNCTION ---
def resolve_company_name(input_query):
    """Translates company names into official trading tickers via Yahoo Finance."""
    cleaned_query = input_query.strip()
    
    if len(cleaned_query) <= 5 and cleaned_query.isalpha() and cleaned_query.isupper():
        return cleaned_query
        
    url = "https://query1.finance.yahoo.com/v1/finance/search"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    params = {"q": cleaned_query, "lang": "en-US", "region": "US"}
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=5)
        response_data = response.json()
        if response_data.get('quotes') and len(response_data['quotes']) > 0:
            resolved_ticker = response_data['quotes'][0]['symbol']
            return resolved_ticker.upper()
    except Exception:
        pass
        
    return cleaned_query.upper()

# --- RESILIENT NEWS HARVESTER ---
def get_resilient_news(stock_obj, ticker_str):
    """Gathers news headlines via yfinance or falls back to raw RSS streams."""
    news_data = []
    
    try:
        news_data = stock_obj.news
    except Exception:
        pass
        
    if not news_data:
        try:
            rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker_str}"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            response = requests.get(rss_url, headers=headers, timeout=5)
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                for item in root.findall('.//item'):
                    title_elem = item.find('title')
                    if title_elem is not None and title_elem.text:
                        news_data.append({"title": title_elem.text})
        except Exception:
            pass
            
    return news_data if news_data else []

# --- SENTIMENT API FUNCTION ---
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
    if not query_list: return pd.DataFrame()
    try:
        df = yf.download(query_list, period="5d", progress=False) 
        if df.empty: return pd.DataFrame()
        for t in query_list:
            try:
                if isinstance(df.columns, pd.MultiIndex):
                    col_ref = df['Close'][t] if t in df['Close'].columns else None
                else:
                    col_ref = df['Close'] if 'Close' in df.columns else None
                
                if col_ref is not None:
                    close_series = col_ref.dropna()
                    if len(close_series) >= 2:
                        close_today = float(close_series.iloc[-1])
                        close_prev = float(close_series.iloc[-2])
                        pct_change = ((close_today - close_prev) / close_prev) * 100
                        scanned_data.append({"ticker": t, "price": close_today, "change": pct_change})
            except: continue
    except: pass
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
                st.toast(f"{row['ticker']} moved {row['change']:+.2f}%", icon="🚀" if row['change'] > 0 else "🚨")
        
        display_count = min(4, len(filtered_df))
        if display_count > 0:
            cols = st.columns(display_count)
            for i, row in enumerate(filtered_df.head(display_count).itertuples()):
                cols[i].metric(label=row.ticker, value=f"${row.price:.2f}", delta=f"{row.change:+.2f}%")

st.markdown("---")
st.markdown("### 🔍 Run Deep AI Multi-Week Analysis")

user_search_input = st.text_input("Enter Company Name or Ticker Symbol:", "Apple")

# --- MASTER ANALYSIS BLOCK ---
if st.button("Run Master Multi-Week Analysis"):
    try: 
        with st.status("Initializing ProQuant Cascade Engine...", expanded=True) as status:
            st.write("🔍 Resolving company identity to exact index ticker...")
            ticker = resolve_company_name(user_search_input)
            st.write(f"🎯 Target confirmed: **{ticker}**")
            
            st.write("📡 Fetching historical timeframe data...")
            stock = yf.Ticker(ticker)
            data = stock.history(period="1y")      
            intraday = stock.history(period="5d", interval="5m")  
            
            if data.empty:
                status.update(label="Data Error", state="error", expanded=True)
                st.error(
                    f"Could not identify or pull market data for '{user_search_input}' "
                    f"(Resolved as: {ticker}). Please try typing a more specific name."
                )
            else:
                st.write("🧠 Training Cascade Models (Weeks 1, 2, and 3)...")
                features = ['Open', 'High', 'Low', 'Close', 'Volume']
                latest_data = data[features].iloc[-1:]
                current_price = data['Close'].iloc[-1]
                
                # Model 1: Week 1 (5 Trading Days)
                df_w1 = data.copy()
                df_w1['Target'] = df_w1['Close'].shift(-5)
                df_w1.dropna(inplace=True)
                m1 = RandomForestRegressor(n_estimators=100, random_state=42)
                m1.fit(df_w1[features], df_w1['Target'])
                forecast_w1 = m1.predict(latest_data)[0]
                
                # Model 2: Week 2 (10 Trading Days)
                df_w2 = data.copy()
                df_w2['Target'] = df_w2['Close'].shift(-10)
                df_w2.dropna(inplace=True)
                m2 = RandomForestRegressor(n_estimators=100, random_state=42)
                m2.fit(df_w2[features], df_w2['Target'])
                forecast_w2 = m2.predict(latest_data)[0]
                
                # Model 3: Week 3 (15 Trading Days)
                df_w3 = data.copy()
                df_w3['Target'] = df_w3['Close'].shift(-15)
                df_w3.dropna(inplace=True)
                m3 = RandomForestRegressor(n_estimators=100, random_state=42)
                m3.fit(df_w3[features], df_w3['Target'])
                forecast_w3 = m3.predict(latest_data)[0]
                
                st.write("📰 Scanning live market sentiment...")
                news = get_resilient_news(stock, ticker)
                bullish_score, bearish_score, num_headlines = 0, 0, 0
                engine_used = "TextBlob (Basic)"
                
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
                                            if item.get('label') == 'positive': bullish_score += item.get('score', 0)
                                            elif item.get('label') == 'negative': bearish_score += item.get('score', 0)
                            else:
                                hf_token = None  
                        if not hf_token:
                            for text in news_texts:
                                analysis = TextBlob(text)
                                if analysis.sentiment.polarity > 0.1: bullish_score += analysis.sentiment.polarity
                                elif analysis.sentiment.polarity < -0.1: bearish_score += abs(analysis.sentiment.polarity)
                
                status.update(label=f"Analysis Complete for {ticker}!", state="complete", expanded=False)

        if not data.empty:
            tab1, tab2, tab3 = st.tabs([
                "🏦 Multi-Week Forecast Path", 
                "⚡ Day Trading Engine",
                "👥 AI Trader Debate Room"
            ])
            
            with tab1:
                st.markdown(f"### 📊 AI Trajectory Outlook: {ticker}")
                
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                m_col1.metric(label="Current Price", value=f"${current_price:.2f}")
                m_col2.metric(label="Week 1 Target", value=f"${forecast_w1:.2f}", delta=f"{forecast_w1 - current_price:+.2f}")
                m_col3.metric(label="Week 2 Target", value=f"${forecast_w2:.2f}", delta=f"{forecast_w2 - forecast_w1:+.2f}")
                m_col4.metric(label="Week 3 Target", value=f"${forecast_w3:.2f}", delta=f"{forecast_w3 - forecast_w2:+.2f}")

                st.markdown("---")
                st.markdown(f"#### Price History & 3-Week AI Forecast Trajectory")
                
                data['SMA_20'] = data['Close'].rolling(window=20).mean()
                data['SMA_50'] = data['Close'].rolling(window=50).mean()
                
                hist_std = data['Close'].pct_change().std()
                unc_w1 = current_price * hist_std * (7 ** 0.5)
                unc_w2 = current_price * hist_std * (14 ** 0.5)
                unc_w3 = current_price * hist_std * (21 ** 0.5)
                
                recent_data = data.iloc[-90:]
                
                fig_daily = go.Figure(data=[go.Candlestick(
                    x=recent_data.index, open=recent_data['Open'], high=recent_data['High'],
                    low=recent_data['Low'], close=recent_data['Close'], name='Historical'
                )])
                
                fig_daily.add_trace(go.Scatter(x=recent_data.index, y=recent_data['SMA_20'], name='20-Day SMA', line=dict(color='#29B6F6', width=1.5)))
                fig_daily.add_trace(go.Scatter(x=recent_data.index, y=recent_data['SMA_50'], name='50-Day SMA', line=dict(color='#FFA726', width=1.5)))
                
                last_date = recent_data.index[-1]
                date_w1 = last_date + timedelta(days=7)
                date_w2 = last_date + timedelta(days=14)
                date_w3 = last_date + timedelta(days=21)
                
                timeline = [last_date, date_w1, date_w2, date_w3]
                path_values = [current_price, forecast_w1, forecast_w2, forecast_w3]
                
                fig_daily.add_trace(go.Scatter(
                    x=timeline, 
                    y=[current_price, forecast_w1 + unc_w1, forecast_w2 + unc_w2, forecast_w3 + unc_w3],
                    mode='lines', line=dict(width=0), showlegend=False
                ))
                fig_daily.add_trace(go.Scatter(
                    x=timeline, 
                    y=[current_price, forecast_w1 - unc_w1, forecast_w2 - unc_w2, forecast_w3 - unc_w3],
                    mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 230, 118, 0.04)', name='Variance Tunnel'
                ))
                
                fig_daily.add_trace(go.Scatter(
                    x=timeline, y=path_values,
                    mode='lines+markers', name='AI Forecast Path',
                    line=dict(color='#00E676' if forecast_w3 >= current_price else '#FF1744', width=3, dash='dash'),
                    marker=dict(size=8, symbol='circle')
                ))
                
                fig_daily.update_layout(xaxis_rangeslider_visible=False, height=450, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_daily, use_container_width=True)
                
                st.write("**Cascade Matrix Breakdown**")
                matrix_df = pd.DataFrame({
                    "Target Horizon": ["Week 1 Target (7 Days)", "Week 2 Target (14 Days)", "Week 3 Target (21 Days)"],
                    "Projected Price": [f"${forecast_w1:.2f}", f"${forecast_w2:.2f}", f"${forecast_w3:.2f}"],
                    "Net Change from Current": [f"{((forecast_w1/current_price)-1)*100:+.2f}%", f"{((forecast_w2/current_price)-1)*100:+.2f}%", f"{((forecast_w3/current_price)-1)*100:+.2f}%"],
                    "Statistical Bound": [f"±${unc_w1:.2f}", f"±${unc_w2:.2f}", f"±${unc_w3:.2f}"]
                })
                st.table(matrix_df)
                
                st.markdown("#### AI Diagnostics & Reasoning")
                net_move = forecast_w3 - current_price
                
                if net_move >= 0:
                    st.success("🤖 Mathematical Model Cascade: NET BULLISH OUTLOOK")
                    msg_bull = (
                        f"Specifics: Three distinct Random Forest setups calculated "
                        f"future intervals from a base price of ${current_price:.2f}. "
                        f"The math displays sequential shifts leading to a cumulative target "
                        f"of ${forecast_w3:.2f} over the next 3 weeks."
                    )
                    st.write(msg_bull)
                else:
                    st.error("🤖 Mathematical Model Cascade: NET BEARISH OUTLOOK")
                    msg_bear = (
                        f"Specifics: Three distinct Random Forest setups calculated "
                        f"future intervals from a base price of ${current_price:.2f}. "
                        f"The math displays sequential degradation leading to a cumulative target "
                        f"of ${forecast_w3:.2f} over the next 3 weeks."
                    )
                    st.write(msg_bear)
                    
                if num_headlines > 0:
                    if bullish_score > bearish_score:
                        lbl_bull = f"📰 Market Psychology ({engine_used}): BULLISH ({num_headlines} Headlines Scanned)"
                        st.success(lbl_bull)
                    elif bearish_score > bullish_score:
                        lbl_bear = f"📰 Market Psychology ({engine_used}): BEARISH ({num_headlines} Headlines Scanned)"
                        st.error(lbl_bear)
                    else:
                        lbl_neu = f"📰 Market Psychology ({engine_used}): NEUTRAL ({num_headlines} Headlines Scanned)"
                        st.info(lbl_neu)
                else:
                    st.warning("⚠️ Sentiment Warning: No active news headlines available for this asset currently.")

            with tab2:
                st.markdown(f"### ⚡ Intraday Momentum: {ticker}")
                if not intraday.empty:
                    intra_current = intraday['Close'].iloc[-1]
                    intra_high = intraday['High'].max()
                    intra_low = intraday['Low'].min()
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric(label="Live Price", value=f"${intra_current:.2f}")
                    c2.metric(label="Session High", value=f"${intra_high:.2f}")
                    c3.metric(label="Session Low", value=f"${intra_low:.2f}")
                    
                    total_range = (intra_high - intra_low) if (intra_high - intra_low) != 0 else 1
                    position_pct = int(((intra_current - intra_low) / total_range) * 100)
                    st.progress(max(0, min(100, position_pct)) / 100)
                    st.caption(f"Price is sitting at {position_pct}% of today's total bracket.")

            with tab3:
                st.markdown("### 👥 Institutional AI Debate Room")
                st.write("Dynamic consensus negotiation block between independent strategies.")
                st.markdown("---")
                
                latest_sma20 = float(data['SMA_20'].iloc[-1])
                latest_sma50 = float(data['SMA_50'].iloc[-1])
                
                sophia_bullish = (forecast_w3 >= current_price)
                marcus_bullish = (current_price >= latest_sma20)
                elena_bullish = (bullish_score >= bearish_score) if num_headlines > 0 else None
                
                votes = [sophia_bullish, marcus_bullish]
                if elena_bullish is not None:
                    votes.append(elena_bullish)
                bull_votes = votes.count(True)
                total_votes = len(votes)
                
                # --- AGENT 1 LOGIC ---
                st.markdown("**🧠 Sophia Vance | Chief Quant Modeler:**")
                if sophia_bullish:
                    sophia_txt = (
                        f"\"The machine learning model matrix is clear. My Random Forest "
                        f"cascade tracks non-linear momentum variables shifting toward a "
                        f"3-week target of ${forecast_w3:.2f}. This mathematical path is "
                        f"statistically sound. I am strictly buying this expansion.\""
                    )
                    st.info(sophia_txt)
                else:
                    sophia_txt = (
                        f"\"The data structure is breaking down. The predictive architecture "
                        f"is tracking a structural weakness pattern collapsing down toward "
                        f"${forecast_w3:.2f} by week 3. Buying here is playing chicken with "
                        f"a train. I am calling for an immediate short position.\""
                    )
                    st.error(s
