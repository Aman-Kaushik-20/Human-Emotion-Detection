import uvicorn
from fastapi import FastAPI
import pickle
import pandas as pd
import numpy as np
import nltk
from nltk.stem import PorterStemmer
import re

## Load Models
model = pickle.load(open('logistic_regresion.pkl', 'rb'))
lb = pickle.load(open('label_encoder.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))


## Custom Function
def predict_emotion(input_text):
    cleaned_text = clean_text(input_text)
    input_vectorized = tfidf_vectorizer.transform([cleaned_text])

    # Predict emotion
    predicted_label = model.predict(input_vectorized)[0]
    predicted_emotion = lb.inverse_transform([predicted_label])[0]
    label =  np.max(model.predict(input_vectorized))

    return predicted_emotion,label


nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('english'))
def clean_text(text):
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    return " ".join(text)


##app making
app = FastAPI()

@app.get("/")
async def index():
    return {"Hello": "World"}

@app.get("/Welcome")
async def read_name(name:str):
    return {"Welcome to Human Emotion Predictor Application": f'{name}'}


@app.put("/Predict Emotion{str}")
async def read_name(text:str):
    prediction, label = predict_emotion(text)
    return {"Predicted Emotion " : prediction }
     
##if -__name__ == '__main__':
  ##  uvicorn.run(app, host= '127.0.0.1', port=8000)