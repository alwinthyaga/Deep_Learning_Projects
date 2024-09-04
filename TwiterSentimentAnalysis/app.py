from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import numpy as np
import os

app = Flask(__name__)

# Check if model and tokenizer files exist
model_path = 'sentiment_analysis_model.h5'
tokenizer_path = 'tokenizer.joblib'

print(f"Looking for model at: {os.path.abspath(model_path)}")
print(f"Looking for tokenizer at: {os.path.abspath(tokenizer_path)}")

# Load the tokenizer
try:
    tokenizer = joblib.load(tokenizer_path)
    print("Tokenizer loaded successfully")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    tokenizer = None

# Load the model
try:
    model = load_model(model_path)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Function to predict sentiment
def predict_sentiment(text):
    if not model or not tokenizer:
        return "Model or Tokenizer not available"
    
    try:
        # Tokenize and pad the input text
        sequences = tokenizer.texts_to_sequences([text])
        padded_sequences = pad_sequences(sequences, maxlen=100)
        
        # Predict the sentiment
        prediction = model.predict(padded_sequences)
        
        # Decode the prediction
        sentiment = np.argmax(prediction, axis=1)[0]
        labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        return labels.get(sentiment, "Unknown sentiment")
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error in prediction"

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = ''
    if request.method == 'POST':
        try:
            user_input = request.form['text']
            sentiment = predict_sentiment(user_input)
        except Exception as e:
            print(f"Error processing request: {e}")
            sentiment = "Error processing request"
    return render_template('index.html', sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)