import numpy as np
import streamlit as st

 
import keras
from keras.datasets import imdb
from keras.utils import pad_sequences

word_index = imdb.get_word_index()
reverse_word_index = {value: key for (key, value) in word_index.items()}

@st.cache_resource
def load_my_model():
    return keras.models.load_model("simple_rnn_imdb.h5")

model = load_my_model()

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = pad_sequences([encoded_review], maxlen=500)
    return padded_review

def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')
user_input = st.text_area('Movie Review')

if st.button('Classify'):
    if user_input.strip():
        sentiment, score = predict_sentiment(user_input)
        st.write(f'Sentiment: {sentiment}')
        st.write(f'Prediction Score: {score:.4f}')
    else:
        st.warning('Please enter a review before classifying.')
else:
    st.write('Please enter a movie review.')


 
