import streamlit as st
import pickle
import re
import string
import os
import requests
from difflib import SequenceMatcher

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 Fake News Detection")
st.caption("Live News API verification + TF-IDF/Logistic Regression backup model — IBM PBEL Project")

def load_model():
    try:
        with open("fake_news_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("tfidf_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        return None, None

model, vectorizer = load_model()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def search_news_api(query, api_key):
    url = "https://newsdata.io/api/1/latest"
    params = {
        "apikey": api_key,
        "q": query,
        "language": "en",
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        if data.get("status") == "success":
            raw_articles = data.get("results", [])
            
            normalized = []
            for a in raw_articles:
                normalized.append({
                    "title": a.get("title"),
                    "url": a.get("link"),
                    "source": {"name": a.get("source_id") or a.get("source_name") or "Unknown"},
                })
            return normalized
        return []
    except requests.exceptions.RequestException:
        return []


def get_live_verdict(headline, api_key):
    articles = search_news_api(headline[:150], api_key)
    if not articles:
        return "NO_MATCH", [], 0

    scored = []
    for a in articles:
        title = a.get("title") or ""
        score = similarity(headline, title)
        scored.append((score, a))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score = scored[0][0] if scored else 0
    top_matches = [a for score, a in scored if score > 0.35]

    if best_score >= 0.55 and top_matches:
        return "REAL", top_matches[:3], best_score
    elif top_matches:
        return "WEAK_MATCH", top_matches[:3], best_score
    else:
        return "NO_MATCH", [], best_score



st.sidebar.header("Settings")
st.sidebar.markdown()
api_key = st.sidebar.text_input("NewsData.io API Key", type="password",
                                 value=os.environ.get("NEWSDATA_KEY", ""))

if model is None:
    st.error()
    st.stop()


news_text = st.text_area(
    "Paste a news headline or article:",
    height=150,
    placeholder="e.g. Scientists discover new planet using powerful telescope",
)

col1, col2 = st.columns([1, 1])
check_button = col1.button("🔍 Check News", use_container_width=True)
clear_button = col2.button("Clear", use_container_width=True)

if clear_button:
    st.rerun()

if check_button:
    if not news_text.strip():
        st.warning("Please paste some text first.")
    else:
        live_status, live_articles, live_score = ("SKIPPED", [], 0)

        
        if api_key:
            with st.spinner("Checking live news sources..."):
                live_status, live_articles, live_score = get_live_verdict(news_text, api_key)

        
        cleaned = clean_text(news_text)
        vec = vectorizer.transform([cleaned])
        ml_prediction = model.predict(vec)[0]
        ml_confidence = model.predict_proba(vec).max() * 100
        ml_label = "REAL" if ml_prediction == 1 else "FAKE" 
       
        st.subheader("✅ Final Verdict")

        if live_status == "REAL":
            st.success(f"**REAL** — confirmed by live news sources (match strength: {live_score*100:.0f}%)")
            verdict_source = "live"
        elif live_status == "WEAK_MATCH":
            st.warning(
                f"**UNCERTAIN** — found loosely related articles online (match: {live_score*100:.0f}%), "
                f"not a strong confirmation. ML model backup says **{ml_label}** ({ml_confidence:.1f}% confidence)."
            )
            verdict_source = "mixed"
        elif live_status == "NO_MATCH":
            st.info(
                "No matching coverage found from live news sources — falling back to the trained ML model."
            )
            if ml_label == "REAL":
                st.success(f"**Backup Model says REAL** — confidence: {ml_confidence:.1f}%")
            else:
                st.error(f"**Backup Model says FAKE** — confidence: {ml_confidence:.1f}%")
            verdict_source = "ml_backup"
        else:  
            st.info("No NewsData.io key entered — using ML model only (no live check performed).")
            if ml_label == "REAL":
                st.success(f"**Model Prediction: REAL** — confidence: {ml_confidence:.1f}%")
            else:
                st.error(f"**Model Prediction: FAKE** — confidence: {ml_confidence:.1f}%")
            verdict_source = "ml_only"

        
        if live_articles:
            st.subheader("🌐 Matching Live Articles")
            for a in live_articles:
                with st.container(border=True):
                    st.markdown(f"**{a.get('title')}**")
                    st.caption(f"Source: {a.get('source', {}).get('name', 'Unknown')}")
                    if a.get("url"):
                        st.markdown(f"[Read more]({a.get('url')})")

        with st.expander("See ML model's independent opinion"):
            st.write(f"Style-based prediction: **{ml_label}** ({ml_confidence:.1f}% confidence)")
            st.caption()

st.divider()
st.caption()
