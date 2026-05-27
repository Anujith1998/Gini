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
                
                # Pre-calculate data structures
                data['SMA_20'] = data['Close'].rolling(window=20).mean()
                data['SMA_50'] = data['Close'].rolling(window=50).mean()
                data['RSI'] = calculate_rsi(data['Close'])
                recent = data.iloc[-90:].copy()
                
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
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "🏦 Multi-Week Forecast", 
                "⚡ Intraday Engine",
                "👥 AI Debate Room",
                "📊 Financial Health",
                "🎯 Institutional Insights"
            ])
            
            with tab1:
                st.markdown(f"### 📊 Forecast Path: {ticker}")
                
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Current", f"${current_price:.2f}")
                mc2.metric("Week 1", f"${forecast_w1:.2f}", f"{forecast_w1 - current_price:+.2f}")
                mc3.metric("Week 2", f"${forecast_w2:.2f}", f"{forecast_w2 - forecast_w1:+.2f}")
                mc4.metric("Week 3", f"${forecast_w3:.2f}", f"{forecast_w3 - forecast_w2:+.2f}")

                st.markdown("---")
                
                h_std = data['Close'].pct_change().std()
                u_w1 = current_price * h_std * (7 ** 0.5)
                u_w2 = current_price * h_std * (14 ** 0.5)
                u_w3 = current_price * h_std * (21 ** 0.5)
                
                fig = go.Figure(data=[go.Candlestick(
                    x=recent.index, open=recent['Open'], high=recent['High'],
                    low=recent['Low'], close=recent['Close'], name='Price'
                )])
                
                fig.add_trace(go.Scatter(
                    x=recent.index, y=recent['SMA_20'], 
                    name='20 SMA', line=dict(color='#29B6F6', width=1)
                ))
                fig.add_trace(go.Scatter(
                    x=recent.index, y=recent['SMA_50'], 
                    name='50 SMA', line=dict(color='#FFA726', width=1)
                ))
                
                l_date = recent.index[-1]
                t_line = [l_date, l_date+timedelta(7), l_date+timedelta(14), l_date+timedelta(21)]
                p_vals = [current_price, forecast_w1, forecast_w2, forecast_w3]
                
                fig.add_trace(go.Scatter(
                    x=t_line, 
                    y=[current_price, forecast_w1+u_w1, forecast_w2+u_w2, forecast_w3+u_w3], 
                    mode='lines', showlegend=False, line=dict(width=0)
                ))
                fig.add_trace(go.Scatter(
                    x=t_line, 
                    y=[current_price, forecast_w1-u_w1, forecast_w2-u_w2, forecast_w3-u_w3], 
                    mode='lines', fill='tonexty', fillcolor='rgba(0,230,118,0.03)', name='Tunnel'
                ))
                
                c_clr = '#00E676' if forecast_w3 >= current_price else '#FF1744'
                fig.add_trace(go.Scatter(
                    x=t_line, y=p_vals, mode='lines+markers', 
                    name='AI Path', line=dict(color=c_clr, width=2, dash='dash')
                ))
                fig.update_layout(
                    xaxis_rangeslider_visible=False, height=400, 
                    margin=dict(l=5, r=5, t=5, b=5)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                m_df = pd.DataFrame({
                    "Horizon": ["Week 1", "Week 2", "Week 3"],
                    "Target": [f"${forecast_w1:.2f}", f"${forecast_w2:.2f}", f"${forecast_w3:.2f}"],
                    "Move": [f"{((forecast_w1/current_price)-1)*100:+.2f}%", f"{((forecast_w2/current_price)-1)*100:+.2f}%", f"{((forecast_w3/current_price)-1)*100:+.2f}%"]
                })
                st.table(m_df)
                
                st.markdown("#### AI Diagnostics & Analysis")
                if (forecast_w3 - current_price) >= 0:
                    st.success("🤖 Cascade Status: NET BULLISH DETECTED")
                else:
                    st.error("🤖 Cascade Status: NET BEARISH DETECTED")
                    
                if num_hl > 0:
                    if bull_s > bear_s:
                        st.success(f"📰 Sentiment Layer: BULLISH ({num_hl} Alerts)")
                    elif bear_s > bull_s:
                        st.error(f"📰 Sentiment Layer: BEARISH ({num_hl} Alerts)")
                    else:
                        st.info(f"📰 Sentiment Layer: NEUTRAL ({num_hl} Alerts)")

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
                    st.caption(f"Price is at {pos}% of today's bracket.")

            with tab3:
                st.markdown(f"### 🗳️ 6-Agent AI Consensus Scoreboard ({ticker})")
                
                l_sma20 = float(data['SMA_20'].iloc[-1])
                l_sma50 = float(data['SMA_50'].iloc[-1])
                l_open = float(data['Open'].iloc[-1])
                l_rsi = float(data['RSI'].iloc[-1]) if not data['RSI'].isna().all() else 50.0
                
                # Dynamic calculated criteria specific to the loaded ticker
                v1 = (forecast_w3 >= current_price)
                v2 = (current_price >= l_sma20)
                v3 = (bull_s >= bear_s) if num_hl > 0 else True
                v4 = (current_price >= l_sma50)
                v5 = (l_rsi <= 65.0)  
                v6 = (forecast_w1 >= current_price)
                
                votes = [v1, v2, v3, v4, v5, v6]
                bull_votes = votes.count(True)
                
                v_df = pd.DataFrame({
                    "Algorithmic Voter Node": [
                        "Agent 1: Cascade ML Matrix (3-Week)",
                        "Agent 2: Short Trend Engine (20-Day SMA)",
                        "Agent 3: Media Sentiment Array",
                        "Agent 4: Macro Baseline (50-Day SMA)",
                        "Agent 5: Momentum Variance Check (RSI)",
                        "Agent 6: Immediate Vector (1-Week Path)"
                    ],
                    "Stance": ["🟢 BULL" if v else "🔴 BEAR" for v in votes]
                })
                st.table(v_df)
                
                st.markdown("#### ⚖️ Final Arbitration Verdict")
                if bull_votes >= 5:
                    st.success(f"🎯 **STRONG BUY** ({bull_votes} Bulls)")
                elif bull_votes == 4:
                    st.info(f"⚖️ **TACTICAL ACCUMULATE** ({bull_votes} Bulls)")
                elif bull_votes == 3:
                    st.warning(f"⚡ **EQUAL WEIGHT / HOLD** ({bull_votes} Bulls)")
                elif bull_votes == 2:
                    st.error(f"🚨 **TACTICAL REDUCE / SHORT** ({bull_votes} Bulls)")
                else:
                    st.error(f"💀 **STRONG LIQUIDATE / SHORT** ({bull_votes} Bulls)")

                st.markdown("---")
                st.markdown("#### 📈 History & 3-Week Forward Consensus Trend")
                
                # DYNAMIC BACKTESTING ROUTINE: Evaluates each historic day uniquely
                r_feats = recent[['Open', 'High', 'Low', 'Close', 'Volume']]
                
                # Generate matrix arrays across the last 90 index days dynamically
                h_v1 = m3.predict(r_feats) >= recent['Close']
                h_v2 = recent['Close'] >= recent['SMA_20']
                h_v3 = pd.Series([v3] * len(recent), index=recent.index) # bound to historical news segment
                h_v4 = recent['Close'] >= recent['SMA_50']
                h_v5 = recent['RSI'] <= 65.0
                h_v6 = m1.predict(r_feats) >= recent['Close']
                
                score_hist = (
                    h_v1.astype(int) + h_v2.astype(int) + 
                    h_v3.astype(int) + h_v4.astype(int) + 
                    h_v5.astype(int) + h_v6.astype(int)
                )

                # Future forward matrix progression projections
                s_w1 = sum([forecast_w3 >= forecast_w1, forecast_w1 >= l_sma20, v3, forecast_w1 >= l_sma50, l_rsi <= 65.0, forecast_w2 >= forecast_w1])
                s_w2 = sum([forecast_w3 >= forecast_w2, forecast_w2 >= l_sma20, v3, forecast_w2 >= l_sma50, l_rsi <= 65.0, forecast_w3 >= forecast_w2])
                s_w3 = sum([True, forecast_w3 >= l_sma20, v3, forecast_w3 >= l_sma50, l_rsi <= 65.0, True])
                
                f_dates = [recent.index[-1], recent.index[-1]+timedelta(7), recent.index[-1]+timedelta(14), recent.index[-1]+timedelta(21)]
                f_scores = [score_hist.iloc[-1], s_w1, s_w2, s_w3]
                
                fig_c = go.Figure()
                fig_c.add_trace(go.Scatter(x=recent.index, y=score_hist, mode='lines', name='Historic Consensus', line=dict(color='#29B6F6', width=2.5)))
                
                fc_clr = '#00E676' if s_w3 >= f_scores[0] else '#FF1744'
                fig_c.add_trace(go.Scatter(x=f_dates, y=f_scores, mode='lines+markers', name='AI Forecast Path', line=dict(color=fc_clr, width=2.5, dash='dash')))
                
                fig_c.add_hrect(y0=3.5, y1=6.5, fillcolor="rgba(0,230,118,0.1)", layer="below", line_width=0)
                fig_c.add_hrect(y0=-0.5, y1=2.5, fillcolor="rgba(255,23,68,0.1)", layer="below", line_width=0)
                
                fig_c.update_layout(
                    yaxis=dict(range=[-0.5, 6.5], tickvals=[0,1,2,3,4,5,6], title="Bull Votes (0-6)"),
                    height=250, margin=dict(l=5, r=5, t=10, b=5), xaxis_rangeslider_visible=False,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_c, use_container_width=True)

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
                    else:
                        st.info("No structured financial statements available.")
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
                        st.table(df_earn[['epsEstimate', 'epsActual', 'epsDifference', 'surprisePercent']].dropna().head(4))
                    else:
                        st.info("Historical tracking data is unavailable for this specific equity configuration.")
                except:
                    st.info("Earnings matrix interface limits reached for this ticker configuration.")

    except Exception as e:
        st.error(f"Terminal Exception Error: {e}")
