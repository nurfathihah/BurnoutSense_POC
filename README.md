# BurnoutSense

Multilingual work-burnout expression classifier (English, Bahasa Malaysia, Manglish).
MSc Data Science, UiTM.

Classifies text into three MBI dimensions: Emotional Exhaustion,
Depersonalisation, and Reduced Personal Accomplishment.

Model: TF-IDF (5,000 features, unigram+bigram) + MLP (256 hidden units).
Test weighted F1 0.9626, macro F1 0.9540.

Research prototype. Not a diagnostic tool.

## Run locally
pip install -r requirements.txt
streamlit run app.py