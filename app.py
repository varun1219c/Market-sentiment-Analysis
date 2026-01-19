# ------------------------------------------------------------
# app.py — Glass Sentiment Dashboard (Advanced, Fixed Version)
# ------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# optional autorefresh
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTREF = True
except:
    HAS_AUTREF = False

# optional OpenAI import
try:
    import openai
    HAS_OPENAI = True
except:
    HAS_OPENAI = False

# -------------------------------------------------
# Load CSS
# -------------------------------------------------
def load_css(path="styles.css"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except:
        st.error("CSS file missing — ensure styles.css is inside the same folder.")

load_css()

# -------------------------------------------------
# Streamlit page settings
# -------------------------------------------------
st.set_page_config(
    page_title="Glass Sentiment Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

if HAS_AUTREF:
    st_autorefresh(interval=20000, key="refresh")

# -------------------------------------------------
# Helper Functions
# -------------------------------------------------
analyzer = SentimentIntensityAnalyzer()

def sentiment_score(text):
    if not isinstance(text, str):
        text = str(text)
    return analyzer.polarity_scores(text)["compound"]

def fetch_google_news(keyword, max_items=80):
    url = f"https://news.google.com/rss/search?q={keyword}&hl=en-US&gl=US&ceid=US:en"
    try:
        rss = feedparser.parse(url)
        items = []
        for i, e in enumerate(rss.entries):
            if i >= max_items:
                break
            items.append({
                "title": getattr(e, "title", ""),
                "description": getattr(e, "summary", ""),
                "link": getattr(e, "link", ""),
                "published": getattr(e, "published", "")
            })
        return items
    except:
        return []

def build_news_df(term):
    data = fetch_google_news(term, max_items=100)
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["content"] = df["title"].fillna("") + " " + df["description"].fillna("")
    df["sentiment"] = df["content"].apply(sentiment_score)
    return df

def get_price(symbol, period="7d", interval="1h"):
    try:
        df = yf.Ticker(symbol).history(period=period, interval=interval)
        if df.empty:
            return None
        return df.reset_index()
    except:
        return None

def make_recommendation(score, price_df):
    if score > 0.25:
        rec = "BUY"; reason = "Sentiment positive."
    elif score < -0.25:
        rec = "SELL"; reason = "Sentiment negative."
    else:
        rec = "HOLD"; reason = "Sentiment mixed."

    # trend logic
    if price_df is not None and not price_df.empty:
        close = price_df["Close"].dropna()
        if len(close) >= 5:
            slope = close.iloc[-1] - close.iloc[-5]
            trend = "uptrend" if slope > 0 else "downtrend"
        else:
            trend = "flat"

        if rec == "BUY" and trend == "uptrend":
            conf = "High confidence — price supports sentiment."
        elif rec == "SELL" and trend == "downtrend":
            conf = "High confidence — price supports sentiment."
        else:
            conf = f"Trend: {trend}."
    else:
        conf = "Price unavailable."

    return rec, f"{reason} {conf}"

def ai_summary_from_news(df, top_n=3):
    if df.empty:
        return "No news found."
    o = df["sentiment"].mean()
    pos = df.sort_values("sentiment", ascending=False).head(top_n)["title"].tolist()
    neg = df.sort_values("sentiment", ascending=True).head(top_n)["title"].tolist()
    return f"Overall sentiment: {o:.3f}. Top positive: {', '.join(pos)}. Top negative: {', '.join(neg)}."

# -------------------------------------------------
# HERO TITLE
# -------------------------------------------------
st.markdown("""
<div class='hero-container animate-fade'>
<h1 class='hero-title'>
REAL-TIME <span class='hero-highlight'>MARKET</span> SENTIMENT<br>
<span class='hero-highlight'>DASHBOARD</span>
</h1>
<div class='hero-subtitle'>
⚡ Live News • AI Sentiment • Auto-Refreshing • Global Markets
</div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.header("Settings & Chat")
instrument = st.sidebar.text_input("Instrument / Company / Keyword", "AAPL")
st.sidebar.caption("Auto-refresh every 20 seconds.")

openai_key = st.sidebar.text_input("OpenAI API key (optional)", type="password")
if openai_key and HAS_OPENAI:
    openai.api_key = openai_key

# -------------------------------------------------
# Market Overview
# -------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### Market Overview (Live)")

overview_syms = {
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Crude Oil": "CL=F",
    "USD/INR": "INR=X",
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "NIFTY 50": "^NSEI",
}

for label, sym in overview_syms.items():
    try:
        df = yf.Ticker(sym).history(period="1d", interval="1h")
        if not df.empty:
            price = df["Close"].iloc[-1]
            prev = df["Close"].iloc[0]
            pct = ((price - prev) / prev) * 100
            sign = "▲" if pct >= 0 else "▼"
            st.sidebar.markdown(
                f"<div class='mini-row'><span class='mini-label'>{label}</span>"
                f"<span class='mini-value'>{price:.2f} {sign}{pct:+.2f}%</span></div>",
                unsafe_allow_html=True
            )
    except:
        pass

# -------------------------------------------------
# Fetch News
# -------------------------------------------------
news_df = build_news_df(instrument)

if news_df.empty:
    st.warning("No news found.")
    st.stop()

overall = news_df["sentiment"].mean()
rating = round(((overall + 1) / 2) * 4 + 1)

# -------------------------------------------------
# METRICS ROW
# -------------------------------------------------
c1, c2, c3, c4 = st.columns([2, 1, 1, 2])

with c1:
    st.markdown(f"<div class='card metric-large'><div class='metric-title'>Overall Sentiment</div><div class='metric-value'>{overall:.3f}</div></div>", unsafe_allow_html=True)

with c2:
    st.markdown(f"<div class='card metric'><div class='metric-title'>Rating</div><div class='metric-value'>{rating}/5</div></div>", unsafe_allow_html=True)

with c3:
    st.markdown(f"<div class='card metric'><div class='metric-title'>Articles</div><div class='metric-value'>{len(news_df)}</div></div>", unsafe_allow_html=True)

with c4:
    headline = news_df['title'].iloc[0]
    st.markdown(f"<div class='card metric'><div class='metric-title'>Top Headline</div><div class='top-head'>{headline}</div></div>", unsafe_allow_html=True)

st.markdown("<div class='glass-divider'></div>", unsafe_allow_html=True)

# -------------------------------------------------
# TREND + GAUGE COLUMNS
# -------------------------------------------------
left, right = st.columns([3, 1])

# ---------------- LEFT COLUMN ---------------------
with left:
    # TREND
    st.markdown("<div class='card'><h3 class='section'>Sentiment Trend</h3>", unsafe_allow_html=True)
    fig_trend = px.line(news_df.reset_index(), y="sentiment", markers=True)
    fig_trend.update_layout(template="plotly_dark", height=320, margin=dict(t=10, b=10))
    st.plotly_chart(fig_trend, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    # ----------------- Chatbot Logic -----------------
def rule_based_reply(prompt, news_df=None):
    prompt_lower = prompt.lower()

    # summary request
    if "summary" in prompt_lower:
        return ai_summary_from_news(news_df)

    # buy/sell/recommendation
    if "buy" in prompt_lower or "sell" in prompt_lower or "recommend" in prompt_lower:
        overall = news_df["sentiment"].mean() if (news_df is not None and not news_df.empty) else 0.0
        rec, summ = make_recommendation(overall, get_price(instrument) if instrument else None)
        return f"Recommendation: {rec}. {summ}"

    # headlines request
    if "headlines" in prompt_lower or "news" in prompt_lower:
        if news_df is None or news_df.empty:
            return "No recent headlines available."
        return "Top headlines: " + " | ".join(news_df['title'].head(5).tolist())

    # default
    return (
        "I can summarize news, infer basic sentiment, "
        "give a basic recommendation, or show top headlines. "
        "Try: 'summary', 'top headlines', or 'Should I buy AAPL?'"
    )

    # AI ASSISTANT
    st.markdown("<div class='card animate-pop'><h3 class='section'>🤖 AI Assistant</h3>", unsafe_allow_html=True)

    query = st.text_input("Ask anything about market, news or trends:", key="assistant_box")

    if query:
        if openai_key and HAS_OPENAI:
            try:
                with st.spinner("Thinking..."):
                    resp = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a market analysis assistant. Keep responses concise."},
                            {"role": "user", "content": query}
                        ],
                        max_tokens=200
                    )
                answer = resp["choices"][0]["message"]["content"]
                st.markdown(f"<div class='ai-output-box'>{ai_reply}</div>", unsafe_allow_html=True)

            except:
                local_reply = rule_based_reply(ai_query, news_df)
                st.markdown(f"<div class='ai-output-box'>{local_reply}</div>", unsafe_allow_html=True)

        else:
            local_reply = rule_based_reply(ai_query, news_df)
            st.markdown(f"<div class='ai-output-box'>{local_reply}</div>", unsafe_allow_html=True)


    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- RIGHT COLUMN ---------------------
with right:
    # GAUGE
    st.markdown("<div class='card'><h3 class='section'>Sentiment Gauge</h3>", unsafe_allow_html=True)
    gauge_val = (overall + 1) / 2
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=gauge_val * 100,
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': '#00eaff'},
            'steps': [
                {'range': [0, 33], 'color': '#ff6b6b'},
                {'range': [33, 66], 'color': '#ffd166'},
                {'range': [66, 100], 'color': '#00eaff'}
            ]
        },
        number={'suffix': '%'}
    ))
    fig_gauge.update_layout(template="plotly_dark", height=260)
    st.plotly_chart(fig_gauge, use_container_width=True)

    # RECOMMENDATION
    st.markdown("<h4 class='section'>Recommendation</h4>", unsafe_allow_html=True)
    price_df = get_price(instrument)
    rec, summary = make_recommendation(overall, price_df)
    if rec == "BUY":
        st.markdown(f"<div class='badge-buy'>BUY</div><div class='rec-text'>{summary}</div>", unsafe_allow_html=True)
    elif rec == "SELL":
        st.markdown(f"<div class='badge-sell'>SELL</div><div class='rec-text'>{summary}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='badge-hold'>HOLD</div><div class='rec-text'>{summary}</div>", unsafe_allow_html=True)

    # AI SUMMARY
    st.markdown("<h4 class='section'>AI Summary</h4>", unsafe_allow_html=True)
    st.write(ai_summary_from_news(news_df))


# -------------------------------------------------
# WORD CLOUD
# -------------------------------------------------
st.markdown("<div class='card'><h3 class='section'>Word Cloud</h3>", unsafe_allow_html=True)
text_blob = " ".join(news_df["content"].tolist())
wc = WordCloud(width=800, height=300, background_color="black", colormap="cool").generate(text_blob)
fig_wc, ax = plt.subplots(figsize=(10, 3))
ax.imshow(wc, interpolation="bilinear")
ax.axis("off")
st.pyplot(fig_wc)
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# NEWS FEED
# -------------------------------------------------
st.markdown("<div class='card'><h3 class='section'>Latest News</h3>", unsafe_allow_html=True)

for i, r in news_df.head(10).iterrows():
    st.markdown("<div class='news-item'>", unsafe_allow_html=True)
    st.markdown(f"<div class='news-title'>{r['title']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='news-desc'>{r['description']}</div>", unsafe_allow_html=True)
    st.markdown(f"<a href='{r['link']}' target='_blank' class='news-link'>Read Article</a>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("<div class='footer'>Data from Google News RSS and Yahoo Finance. Not financial advice.</div>", unsafe_allow_html=True)
