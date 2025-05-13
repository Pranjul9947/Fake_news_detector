import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd

# Load pre-trained model
@st.cache_resource
def load_model():
    model = joblib.load('fake_news_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

# Initialize NLTK
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Streamlit UI
st.title('Fake News Detector')
st.write("Upload trained model files to analyze news articles")

# File upload
uploaded_model = st.file_uploader("Upload Model (.pkl)", type="pkl")
uploaded_vec = st.file_uploader("Upload Vectorizer (.pkl)", type="pkl")

if uploaded_model and uploaded_vec:
    model = joblib.load(uploaded_model)
    vectorizer = joblib.load(uploaded_vec)
    
    # Input text
    news_text = st.text_area("Paste news article here:", height=300)
    
    if st.button('Analyze'):
        if news_text:
            # Preprocess and predict
            processed_text = preprocess_text(news_text)
            text_vec = vectorizer.transform([processed_text])
            prediction = model.predict(text_vec)[0]
            proba = model.predict_proba(text_vec)[0]
            
            # Display results
            if prediction == 1:
                st.error(f"Fake News (confidence: {proba[1]*100:.1f}%)")
            else:
                st.success(f"Real News (confidence: {proba[0]*100:.1f}%)")
            
            # Probability chart
            prob_df = pd.DataFrame({
                'Label': ['Real', 'Fake'],
                'Probability': proba
            })
            st.bar_chart(prob_df.set_index('Label'))
        else:
            st.warning("Please enter some text to analyze")
else:
    st.info("Please upload both model files to begin")