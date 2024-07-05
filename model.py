import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM,Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

data = pd.read_csv("twitter_tweets.csv",encoding = 'latin1')
data = data[data['sentiment'] != 'neutral']


data = data.drop(columns="textID",axis = 1)
data = data.drop(columns="Time of Tweet",axis = 1)
data = data.drop(columns="Age of User",axis = 1)
data = data.drop(columns="Country",axis = 1)
data = data.drop(columns="Population -2020",axis = 1)
data = data.drop(columns="Land Area (Km²)",axis = 1)
data = data.drop(columns="Density (P/Km²)",axis = 1)
data = data.dropna()
data["sentiment"].value_counts()

data.replace({"sentiment": {"positive": 1, "negative": 0}}, inplace=True)


train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
print("Train data shape:", train_data.shape)


tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(train_data["text"])
X_train = pad_sequences(tokenizer.texts_to_sequences(train_data["text"]), maxlen=200)
X_test = pad_sequences(tokenizer.texts_to_sequences(test_data["text"]), maxlen=200)
Y_train = train_data["sentiment"].values
Y_test = test_data["sentiment"].values


print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)


model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128))  # No input_length here
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, input_shape=(200, 128)))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
print("model compiled")

model.fit(X_train,Y_train,epochs = 6,batch_size = 8,validation_split=0.2)
print("model_trained")
def predict_sentiment(text):
    
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower()) 
    sequences = tokenizer.texts_to_sequences([text]) 
    padded_sequence = pad_sequences(sequences, maxlen=200)  
    prediction = model.predict(padded_sequence)  # Predict sentiment
    sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
    return sentiment

model.save("sentiment_model.h5")
print(".h5 file created")
import pickle
with open("tokenizer.pkl", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(".pkl file created")