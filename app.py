from flask import Flask, request, render_template
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import requests
import os

app = Flask(__name__)

# Dropbox file URL
dropbox_url = "https://www.dropbox.com/scl/fi/ay9hr2f3lng4ot6jvilx9/TinyBERT_model.pt?rlkey=ftszbz8tq0zunmdti5twciqex&st=m89asrzl&dl=1"
output_file = "TinyBERT_model.pt"

# Check if the model file exists, otherwise download it
if not os.path.exists(output_file):
    print("Downloading model from Dropbox...")
    response = requests.get(dropbox_url, stream=True)
    with open(output_file, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Download completed.")
else:
    print("Model file already exists locally.")

# Load the saved model
model = torch.load(output_file)
model.eval()

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")

# Preprocessing function for reviews
def preprocess_review(review):
    # Tokenize the input review
    encoding = tokenizer(
        review,
        padding=True,
        truncation=True,
        max_length=100,  # Match the max_length used during training
        return_tensors="pt"
    )
    return encoding

# Prediction function
def predict_sentiment(review):
    # Preprocess the review
    inputs = preprocess_review(review)

    # Pass through the model
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probabilities, axis=1).item()

    # Convert to sentiment label
    return "Positive" if prediction == 1 else "Negative"

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']  # Get review input from the form
    sentiment = predict_sentiment(review)  # Predict sentiment
    return render_template('result.html', result=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
