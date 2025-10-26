```markdown
# AI Dyslexia Assistant

A Streamlit app that predicts readability, highlights difficult words, and lets you download a cleaned-up PDF.

## Features
- Readability label with confidence (easy/hard).
- Highlights difficult words based on syllables/length (spaCy tokenization + textstat).
- One-click PDF download of the reformatted output.

## Quickstart
- Python 3.x
- Install:
  ```
  pip install streamlit pandas joblib textstat spacy xhtml2pdf
  python -m spacy download en_core_web_sm
  ```
- Run:
  ```
  streamlit run dyslexia_assistant_app.py
  ```

## Configure model path
Update the hardcoded absolute path to a relative path in your repo:
```
# dyslexia_assistant_app.py
model = joblib.load("model/readibility_model.pkl")
```
Place your trained `readibility_model.pkl` in the `model/` folder.

## How it works
- Extracts features: Flesch score, sentence count, words per sentence, syllable count, difficult word count.
- Predicts a class and confidence; maps to “Easy to Read” or “Hard to Read”.
- Highlights tokens with high syllables or long length while preserving punctuation.

## Suggested structure
```
.
├─ dyslexia_assistant_app.py
├─ model/
│  └─ readibility_model.pkl
├─ requirements.txt
└─ README.md
```

## Notes
- Ensure `en_core_web_sm` is installed before running.
- If you change feature engineering, retrain or align the saved model to the same feature schema.
```
