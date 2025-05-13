# Fake News Detection System 🔍

A machine learning model to classify news articles as real or fake using NLP techniques.
[Screenshot 2025-05-14 040727](https://github.com/user-attachments/assets/d2e08aeb-3da1-4037-951d-d9921df067be)
[Screenshot 2025-05-14 040702](https://github.com/user-attachments/assets/a5faa372-4b8a-49c4-b20e-1d4bb0720716)


## 📥 Download Links
**Pre-trained Models:**
- [Model (Pickle)]((https://drive.google.com/file/d/1ZW2MI_4XIZe8bMqO5q_SB4X-LpxsUWDr/view?usp=sharing)) -  (*Name it as fake_news_model.pkl*)
- [TF-IDF Vectorizer]((https://drive.google.com/file/d/10GcHo64vqYjtNXGT3fSOLwiwwGxT_CoT/view?usp=sharing)) -  (*Name it as tfidf_vectorizer.pkl*)

**Dataset:**
https://www.kaggle.com/datasets/csmalarkodi/isot-fake-news-dataset?select=Fake.csv
(*True and Fake both datasets*)

## 🛠️ Setup

### Conda Environment Setup!


```bash
conda env create -f environment.yml
conda activate fake_news_detection
python -m nltk.downloader punkt stopwords wordnet!
```

## Other Resources
(https://drive.google.com/file/d/10GcHo64vqYjtNXGT3fSOLwiwwGxT_CoT/view?usp=sharing)
