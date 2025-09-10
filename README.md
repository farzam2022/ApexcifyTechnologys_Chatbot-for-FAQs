
# 💬 FAQ Chatbot  

A simple **FAQ-based Chatbot** built with **Python, NLP, and Streamlit**.  
It matches user questions with pre-defined FAQs using **TF-IDF + Cosine Similarity** and returns the most relevant answer.  

---

## 📖 Description  
This chatbot is designed to handle **Frequently Asked Questions (FAQs)** about a topic or product.  
It uses **NLP preprocessing** (tokenization, stopword removal, lemmatization) and **similarity matching** to understand user queries.  

---

## 🎯 Features  
- Predefined FAQ dataset (customizable)  
- Text preprocessing with **NLTK**  
- TF-IDF + Cosine Similarity for matching  
- Simple **Streamlit UI** for chatting  
- Lightweight and easy to extend  

---

## 🚀 How to Run  

1. Clone this repository or download the files.  
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  
3. Run the app:  
   ```bash
   streamlit run chatbot.py
   ```  

---

## 📌 Example FAQs  
- **Q:** What is this chatbot?  
  **A:** It's an FAQ bot that answers predefined questions.  

- **Q:** How does it work?  
  **A:** It uses TF-IDF and cosine similarity to find the closest answer.  

---

## ✅ Future Enhancements  
- Add **embedding-based matching** (e.g., BERT, Sentence Transformers)  
- Store FAQs in a **database**  
- Add **user feedback system**  

---

## 🏷️ Tech Stack  
- Python 🐍  
- Streamlit 🎨  
- NLTK 📚  
- Scikit-learn 🤖  

---

👨‍💻 Built as part of an internship project.  
