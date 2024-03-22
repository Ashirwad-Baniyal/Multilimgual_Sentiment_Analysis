import streamlit as st
import numpy as np
import re
from nltk.stem import PorterStemmer
import pickle
import nltk
from googletrans import Translator

# Download NLTK stopwords
nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('english'))

# Loading the saved files
lg = pickle.load(open('logistic_regresion.pkl','rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl','rb'))
lb = pickle.load(open('label_encoder.pkl','rb'))

# Repeating the same functions
def clean_text(text):
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    return " ".join(text)

def predict_emotion(input_text):
    cleaned_text = clean_text(input_text)
    input_vectorized = tfidf_vectorizer.transform([cleaned_text])

    # Predict emotion
    predicted_label = lg.predict(input_vectorized)[0]
    predicted_emotion = lb.inverse_transform([predicted_label])[0]
    label =  np.max(lg.predict(input_vectorized))

    return predicted_emotion, label

# Translate text to English
def translate_to_english(text):
    translator = Translator()
    translated = translator.translate(text, dest='en').text
    return translated

# Creating the app
st.title("Six Human Emotions Detection App")
st.write("=================================================")
st.write("['Joy,'Fear','Anger','Love','Sadness','Surprise']")
st.write("=================================================")

# Taking input from user
user_input = st.text_input("Enter your text here:")

if st.button("Predict"):
    # Translate input to English
    english_text = translate_to_english(user_input)
    predicted_emotion, label = predict_emotion(english_text)
    st.write("Predicted Emotion:", predicted_emotion)
    st.write("Probability:", label)