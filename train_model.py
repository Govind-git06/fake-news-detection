"""
train_model.py
----------------
Trains the Fake News Detection model and saves it to disk so the
Streamlit app (app.py) can load it instantly without retraining
every time someone opens the app.

HOW TO RUN:
1. Download "Fake.csv" and "True.csv" from the Kaggle dataset:
   "Fake and Real News Dataset"
2. Put both files in this same folder
3. Run:  python train_model.py
4. This creates two files: fake_news_model.pkl and tfidf_vectorizer.pkl
   -> Upload BOTH of these to your GitHub repo along with app.py
"""

import pandas as pd
import re
import string
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def clean_text(text):
    """Removes punctuation, links, numbers, and noise from text."""
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text


def main():
    print("Loading data...")
    fake_df = pd.read_csv("Fake.csv")
    real_df = pd.read_csv("True.csv")

    fake_df["label"] = 0  # FAKE
    real_df["label"] = 1  # REAL

    df = pd.concat([fake_df, real_df], axis=0)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df["text"] = df["title"] + " " + df["text"]
    df["text"] = df["text"].apply(clean_text)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42
    )

    print("Vectorizing text (TF-IDF)...")
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training model (Logistic Regression)...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))

    print("Saving model files...")
    with open("fake_news_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("\nDone! Created fake_news_model.pkl and tfidf_vectorizer.pkl")
    print("Now run: streamlit run app.py")


if __name__ == "__main__":
    main()
