import re
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the tokenizer
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Load the trained model
model = load_model("sentiment_model.h5")

def preprocess_text(text):
    """Clean and preprocess the text."""
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())  # Clean the text
    sequences = tokenizer.texts_to_sequences([text])  # Tokenize
    padded_sequence = pad_sequences(sequences, maxlen=200)  # Pad the sequence
    return padded_sequence

def predict_sentiment(text):
    """Predict the sentiment of the given text."""
    processed_text = preprocess_text(text)
    prediction = model.predict(processed_text)  # Predict sentiment
    sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
    return sentiment
