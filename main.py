# chatbot_faq.py

import nltk
import string
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Download required NLTK data (only first run) ---
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")

# --- Sample FAQ dataset ---
faqs = {
    "What are your business hours?": "We are open from 9 AM to 6 PM, Monday to Saturday.",
    "Where are you located?": "We are located in Islamabad, Pakistan.",
    "How can I contact customer support?": "You can contact us at support@example.com or call +92-300-1234567.",
    "Do you offer international shipping?": "Yes, we ship to most countries worldwide.",
    "What payment methods do you accept?": "We accept Visa, MasterCard, PayPal, and Cash on Delivery."
}

# --- Preprocessing function ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess(text):
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

# --- Prepare FAQ data ---
questions = list(faqs.keys())
answers = list(faqs.values())
preprocessed_questions = [preprocess(q) for q in questions]

# --- TF-IDF Vectorizer ---
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(preprocessed_questions)

def chatbot_response(user_input):
    user_input_processed = preprocess(user_input)
    user_vector = vectorizer.transform([user_input_processed])
    similarity = cosine_similarity(user_vector, faq_vectors)
    idx = similarity.argmax()
    return answers[idx]

# --- Streamlit UI ---
st.title("💬 FAQ Chatbot")
st.write("Ask me anything about our services!")

# Chat input
user_input = st.text_input("You:", "")

if user_input:
    response = chatbot_response(user_input)
    st.text_area("Chatbot:", value=response, height=100)

st.markdown("---")
st.caption("Built with 🧠 NLP + Streamlit")
