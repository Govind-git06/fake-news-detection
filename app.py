"""
app.py
-------
Streamlit web app for Fake News Detection — REAL-TIME VERSION.

This version treats live internet verification as the PRIMARY signal,
and the trained ML model as a BACKUP signal:

1. LIVE CHECK (primary)  -> Ask NewsData.io: "is any real news outlet
   currently reporting something matching this headline?"
   If yes, with high similarity -> we trust that as REAL.
   NewsData.io's free tier (unlike NewsAPI.org) explicitly allows
   requests from a deployed/public domain, not just localhost —
   so this version keeps working after you deploy it.

2. ML MODEL (backup)     -> If nothing is found online (too old, too new,
   or outside the free tier's coverage), fall back to the trained
   TF-IDF + Logistic Regression model, which judges based on writing
   STYLE learned from thousands of past real/fake articles.

3. FINAL VERDICT          -> Combines both into one clear answer, and
   always tells the user WHICH method the verdict came from, so it's
   honest about its own confidence.

The interface is organized into tabs: Check News, How It Works,
Model Stats (real numbers loaded from model_metrics.json, produced by
train_model.py — never hardcoded), and About.

RUN LOCALLY:
    streamlit run app.py

DEPLOY:
    Push to GitHub -> deploy on https://share.streamlit.io
    Add NEWSDATA_KEY under Secrets.
"""

import streamlit as st
import pickle
import re
import string
import os
import json
import requests
from difflib import SequenceMatcher

# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 Fake News Detection")
st.caption("Live News API verification + TF-IDF/Logistic Regression backup model — IBM PBEL Project")


# ---------------------------------------------------------
# Load the trained ML model (cached so it only loads once)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        with open("fake_news_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("tfidf_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        return None, None


@st.cache_data
def load_metrics():
    """Loads real training metrics saved by train_model.py.
    Returns None if the file isn't there (e.g. older model version)."""
    try:
        with open("model_metrics.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


model, vectorizer = load_model()
metrics = load_metrics()


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
    """Returns a 0-1 score of how similar two headlines are."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


# ---------------------------------------------------------
# Live News API check (NewsData.io — production/public-domain safe
# on its free tier, unlike NewsAPI.org which is localhost-only)
# ---------------------------------------------------------
def search_news_api(query, api_key):
    """Searches NewsData.io for real articles matching the headline."""
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
    """
    Checks live news sources and returns:
      ("REAL", best_match_articles, best_score)   -> strong match found
      ("NO_MATCH", [], 0)                         -> nothing relevant found
    """
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


# ---------------------------------------------------------
# Sidebar: API key (built-in via Secrets, with manual override)
# ---------------------------------------------------------
st.sidebar.header("Settings")


def get_builtin_api_key():
    """Checks Streamlit Secrets first (used when deployed), then a local
    environment variable (used when running locally with the key exported).
    Returns an empty string if neither is set."""
    try:
        return st.secrets["NEWSDATA_KEY"]
    except (KeyError, FileNotFoundError):
        return os.environ.get("NEWSDATA_KEY", "")


builtin_key = get_builtin_api_key()

if builtin_key:
    st.sidebar.success("✅ Live news verification is enabled.")
    api_key = builtin_key
else:
    st.sidebar.markdown(
        "No built-in API key found. Get a free key from "
        "[newsdata.io](https://newsdata.io/register) to enable **live** "
        "verification, or the app will use the trained ML model only."
    )
    api_key = st.sidebar.text_input("NewsData.io API Key (optional)", type="password")

if model is None:
    st.error(
        "ML backup model files not found. Please run `train_model.py` first to generate "
        "`fake_news_model.pkl`, `tfidf_vectorizer.pkl`, and `model_metrics.json`, then "
        "place them in this folder."
    )
    st.stop()

# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
tab_check, tab_how, tab_stats, tab_about = st.tabs(
    ["🔍 Check News", "⚙️ How It Works", "📊 Model Stats", "ℹ️ About"]
)

# ===========================================================
# TAB 1: Check News
# ===========================================================
with tab_check:
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

            # --- Step 1: Try live verification first ---
            if api_key:
                with st.spinner("Checking live news sources..."):
                    live_status, live_articles, live_score = get_live_verdict(news_text, api_key)

            # --- Step 2: ML model prediction (always computed, used as backup) ---
            cleaned = clean_text(news_text)
            vec = vectorizer.transform([cleaned])
            ml_prediction = model.predict(vec)[0]
            ml_confidence = model.predict_proba(vec).max() * 100
            ml_label = "REAL" if ml_prediction == 1 else "FAKE"

            # --- Step 3: Combine into a final verdict ---
            st.subheader("✅ Final Verdict")

            if live_status == "REAL":
                st.success(f"**REAL** — confirmed by live news sources (match strength: {live_score*100:.0f}%)")
            elif live_status == "WEAK_MATCH":
                st.warning(
                    f"**UNCERTAIN** — found loosely related articles online (match: {live_score*100:.0f}%), "
                    f"not a strong confirmation. ML model backup says **{ml_label}** ({ml_confidence:.1f}% confidence)."
                )
            elif live_status == "NO_MATCH":
                st.info(
                    "No matching coverage found from live news sources — falling back to the trained ML model."
                )
                if ml_label == "REAL":
                    st.success(f"**Backup Model says REAL** — confidence: {ml_confidence:.1f}%")
                else:
                    st.error(f"**Backup Model says FAKE** — confidence: {ml_confidence:.1f}%")
            else:  # SKIPPED - no API key entered
                st.info("No NewsData.io key entered — using ML model only (no live check performed).")
                if ml_label == "REAL":
                    st.success(f"**Model Prediction: REAL** — confidence: {ml_confidence:.1f}%")
                else:
                    st.error(f"**Model Prediction: FAKE** — confidence: {ml_confidence:.1f}%")

            # --- Step 4: Show supporting evidence ---
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
                st.caption(
                    "This is based purely on writing patterns learned from historical "
                    "labeled articles — it does not check if this is currently in the news."
                )

    st.divider()
    st.caption(
        "How it decides: live NewsData.io results are checked first (primary signal). "
        "If nothing relevant is found online, the app falls back to the trained "
        "TF-IDF + Logistic Regression model (secondary signal)."
    )

# ===========================================================
# TAB 2: How It Works
# ===========================================================
with tab_how:
    st.subheader("⚙️ How the Detection Works")

    st.markdown("""
This app makes a decision in **two layers**, and always tells you which one produced the final answer.
""")

    st.markdown("#### 1️⃣ Live Verification (checked first)")
    st.write(
        "Your headline is sent to **NewsData.io**, a live news search API, to check whether "
        "real news outlets are currently reporting something similar. If a strong match is "
        "found, the headline is marked **REAL** directly from live evidence."
    )

    st.markdown("#### 2️⃣ Machine Learning Model (backup)")
    st.write(
        "If nothing relevant is found online — the story might be old, very recent, or outside "
        "the API's free coverage — the app falls back to a model trained on thousands of "
        "labeled real and fake news articles."
    )
    st.markdown("""
- **TF-IDF (Term Frequency–Inverse Document Frequency)** converts the article's text into
  numbers, highlighting words that are distinctive to that article rather than common
  filler words.
- **Logistic Regression** then uses those numbers to predict REAL or FAKE, based on writing
  style patterns it learned from the training data.
""")

    st.markdown("#### 3️⃣ Final Verdict")
    st.write(
        "The app combines both signals into one clear answer and is transparent about which "
        "method produced it — live confirmation, ML backup, or a mix of both when the live "
        "match is weak."
    )

    st.info(
        "**Why two layers instead of one?** The live check is accurate but can only verify "
        "topics currently in the news. The ML model works on any text instantly, but only "
        "judges writing style — not real-world truth. Together they cover each other's blind spots."
    )

# ===========================================================
# TAB 3: Model Stats
# ===========================================================
with tab_stats:
    st.subheader("📊 Model Performance")

    if metrics:
        col1, col2 = st.columns(2)
        col1.metric("Accuracy", f"{metrics['accuracy']}%")
        col2.metric("F1 Score", f"{metrics['f1_score']}%")

        col3, col4 = st.columns(2)
        col3.metric("Precision", f"{metrics['precision']}%")
        col4.metric("Recall", f"{metrics['recall']}%")

        st.divider()
        st.markdown("#### Training Data")
        st.write(f"**Total articles used:** {metrics['total_articles']:,}")
        st.write(f"- Fake articles: {metrics['fake_articles']:,}")
        st.write(f"- Real articles: {metrics['real_articles']:,}")
        st.write(f"**Training set size:** {metrics['training_samples']:,} articles")
        st.write(f"**Testing set size:** {metrics['testing_samples']:,} articles (unseen by the model during training)")

        st.caption(
            "These are real numbers generated by train_model.py on the actual dataset, "
            "not hardcoded — retraining will update this automatically."
        )
    else:
        st.warning(
            "No `model_metrics.json` found. This file is created automatically the next time "
            "you run `train_model.py`. Older model versions won't have it — retrain to see stats here."
        )

# ===========================================================
# TAB 4: About
# ===========================================================
with tab_about:
    st.subheader("ℹ️ About This Project")
    st.markdown("""
**Fake News Detection using Machine Learning and Live News API**

A project built as part of the **IBM PBEL (Project-Based Experiential Learning)** program,
combining a classic NLP/ML pipeline with real-time API verification.

**Tech stack:**
- Python, scikit-learn (TF-IDF + Logistic Regression)
- Streamlit (web interface)
- NewsData.io (live news verification API)

**How it's different from a typical fake news classifier:**
Most beginner projects stop at "train a model, show accuracy." This app goes a step further
by checking live news sources first, and only relying on the trained model when the internet
has nothing to confirm or deny — making the tool genuinely useful rather than a style-guessing
demo.
""")
    st.caption("Built by Govind · GNIOT (AKTU) · IBM PBEL Project")
