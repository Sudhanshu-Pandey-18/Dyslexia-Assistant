# dyslexia_assistant_app.py

import streamlit as st
import pandas as pd
import joblib
import textstat
import spacy
from io import BytesIO
from xhtml2pdf import pisa

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load ML model
model = joblib.load("/Users/sudhanshu_pandey/Library/CloudStorage/OneDrive-Personal/Data Science/Dyslexia-assistant/model/readibility_model.pkl")

# ========== Feature extraction ==========
def extract_features(text):
    return {
        "flesch_score": textstat.flesch_reading_ease(text),
        "sentence_count": textstat.sentence_count(text),
        "words_per_sentence": textstat.words_per_sentence(text),
        "syllable_count": textstat.syllable_count(text),
        "difficult_words": textstat.difficult_words(text),
    }

# ========== Highlight difficulty ==========
def highlight_difficult_words(text):
    doc = nlp(text)
    highlighted = []
    for token in doc:
        word = token.text
        if not word.isalpha():
            highlighted.append(word)
            continue
        syllables = textstat.syllable_count(word)
        length = len(word)
        if syllables >= 4 or length >= 12:
            highlighted.append(f"<span style='color:red; font-weight:bold'>{word}</span>")
        elif syllables == 3 or 10 <= length < 12:
            highlighted.append(f"<span style='color:orange'>{word}</span>")
        else:
            highlighted.append(word)
    return " ".join(highlighted)

# ========== Predict readability ==========
def predict_readability(text):
    feats = extract_features(text)
    df = pd.DataFrame([feats])
    prediction = model.predict(df)[0]
    prob = model.predict_proba(df)[0][prediction]
    return prediction, prob

# ========== Download helper ==========
def generate_pdf_download(html_content):
    pdf_buffer = BytesIO()
    pisa.CreatePDF(f"<html><body>{html_content}</body></html>", dest=pdf_buffer)
    pdf_buffer.seek(0)
    return pdf_buffer

# ========== Streamlit App ==========
st.set_page_config(page_title="AI Dyslexia Assistant", layout="wide")
st.title("🧠 AI-Powered Dyslexia Assistant")
st.markdown("""Highlight, simplify, predict readability, and download dyslexia-friendly text.""")

user_input = st.text_area("📥 Enter your paragraph here:", height=200)

if st.button("🔍 Analyze Text") and user_input:
    # Prediction
    pred, prob = predict_readability(user_input)
    label_text = "✅ Easy to Read" if pred == 1 else "⚠️ Hard to Read"
    st.subheader(f"Readability Prediction: {label_text} ({prob*100:.1f}% confidence)")

    # Highlighted Output
    highlighted_text = highlight_difficult_words(user_input)
    st.markdown("### 📖 Reformatted Output")
    st.markdown(highlighted_text, unsafe_allow_html=True)

    # Download Option
    st.markdown("### 📥 Download Reformatted Text")
    st.download_button(
        "⬇️ Download .pdf",
        generate_pdf_download(highlighted_text),
        file_name="reformatted.pdf",
        mime="application/pdf"
    )
