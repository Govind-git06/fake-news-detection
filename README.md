# 📰 Fake News Detection — Web App (Streamlit + News API)

An interactive web app that classifies news as **REAL** or **FAKE** by checking **live news sources first**, and falling back to a trained Machine Learning model when nothing current is found.

## 📌 Project Description

This project detects fake news using a two-layer approach. The **primary check** queries the **NewsData.io REST API** in real time to see whether credible news outlets are currently reporting something matching the headline — if a strong match is found, the headline is verified as REAL directly from live data. When no relevant live coverage exists (older stories, very recent events not yet indexed, or no API key provided), the app **falls back** to a TF-IDF + Logistic Regression model trained on ~40,000 labeled real and fake news articles, which judges the text based on writing style and language patterns instead. The entire experience is wrapped in a **Streamlit** web interface, and the app always tells the user which method produced the final verdict, so the result stays transparent rather than a black box.

## 🧠 Why Two Layers?

- **Live API check** — accurate for verifying *"is this actually being reported right now,"* but only works for topics recent/indexed news sources cover.
- **ML model** — works on any text instantly (even offline), but only judges *writing style*, not real-world truth — useful as a fallback, and also catches satire/fabricated stories that no outlet would ever report.

## 🛠️ Tech Stack

- **Python 3**
- **Scikit-learn** — TF-IDF vectorization + Logistic Regression model
- **Streamlit** — interactive web app / UI
- **NewsData.io** — live news search API for cross-referencing
- **Requests** — API calls

## 📁 Project Structure

```
fake-news-detection/
├── train_model.py          # Trains the ML model, saves .pkl files
├── app.py                  # Streamlit web app (the main deliverable)
├── requirements.txt        # Dependencies
├── fake_news_model.pkl     # (generated) trained model
├── tfidf_vectorizer.pkl    # (generated) trained vectorizer
└── README.md
```

## ⚙️ How It Works

1. **Train the model** — `train_model.py` loads the Kaggle "Fake and Real News" dataset, cleans the text, converts it to TF-IDF vectors, and trains a Logistic Regression classifier
2. **Save the model** — trained model + vectorizer are saved as `.pkl` files so the app doesn't retrain every time
3. **Web app** — `app.py` loads the saved model, takes user input, predicts REAL/FAKE with a confidence score
4. **Live API check** — the app queries NewsData.io for articles matching the headline, showing real outlet results side-by-side with the model's prediction

## 🚀 How to Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download Fake.csv and True.csv from Kaggle
#    ("Fake and Real News Dataset") and place them in this folder

# 3. Train the model (creates the .pkl files)
python train_model.py

# 4. Launch the web app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## 🔑 Getting a NewsData.io Key (free, production-safe)

Unlike NewsAPI.org (localhost-only on its free tier), NewsData.io's free tier explicitly allows requests from a deployed/public domain — so this key works both locally and after you deploy.

1. Go to [newsdata.io/register](https://newsdata.io/register)
2. Sign up for a free developer account
3. Copy your API key
4. Paste it into the sidebar of the running app (or set it as an environment variable `NEWSDATA_KEY`)

## 🌐 Deploying for a Public Demo Link

1. Push this whole folder (including the `.pkl` files) to a GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app**, select your repo and `app.py`
4. Under **Advanced settings → Secrets**, add:
   ```
   NEWSDATA_KEY = "your_key_here"
   ```
5. Click **Deploy** — you'll get a public URL like `https://your-app.streamlit.app` to show during your presentation

## 🎓 About

Built as part of the **IBM PBEL (Project-Based Experiential Learning)** program.
