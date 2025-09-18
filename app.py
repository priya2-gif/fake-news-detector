import streamlit as st
import pickle
import re
import nltk
import string
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Set up NLTK data directory (safe for Streamlit Cloud)
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
nltk.download('stopwords', download_dir=nltk_data_dir)
nltk.download('wordnet', download_dir=nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

# Load model and vectorizer
model = pickle.load(open('fake_news_model.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Text preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    words = text.split()  # ✅ replaced nltk.word_tokenize with split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# Streamlit app UI
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detection App")
st.markdown("Enter a news article to check if it's **Fake** or **Real**.")

user_input = st.text_area("Paste news content here...", height=200)

if st.button("Check News"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
        if prediction == 1:
            st.error("❌ This news is **FAKE**.")
        else:
            st.success("✅ This news is **REAL**.")
