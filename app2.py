from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load tokenizer and model for Persian sentiment analysis (use BertTokenizer and BertForSequenceClassification)
tokenizer = BertTokenizer.from_pretrained("HooshvareLab/bert-fa-base-uncased-sentiment-snappfood")
model = BertForSequenceClassification.from_pretrained("HooshvareLab/bert-fa-base-uncased-sentiment-snappfood")

@app.route('/analyze-sentiment', methods=['POST'])
def analyze_sentiment():
    try:
        # Get text from the request body
        data = request.json
        text = data.get('text', '')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Tokenize input text
        inputs = tokenizer(text, return_tensors="pt")

        # Run the model to get logits (without gradient calculation)
        with torch.no_grad():
            logits = model(**inputs).logits

        # Get predicted class (0 = negative, 1 = neutral, 2 = positive)
        predicted_class_id = logits.argmax().item()
        sentiment = model.config.id2label[predicted_class_id]

        return jsonify({'sentiment': sentiment})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
