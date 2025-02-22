from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import pickle

# Load the trained model
model_filename = 'best_phishing_model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)

# Home route with input form
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get the email text input from form
        email_text = request.form['email_text']
        
        # Preprocess the input (This should match the training data preprocessing)
        # Example: Convert text to features
        input_features = np.array([len(email_text), email_text.count('!'), email_text.count('$')]).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_features)
        result = 'Phishing' if prediction[0] == 1 else 'Not Phishing'
        
        return render_template('index.html', result=result, email_text=email_text)
    
    return render_template('index.html')

# API route for prediction (optional for integration with other systems)
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    email_text = data['email_text']
    
    # Preprocess the input
    input_features = np.array([len(email_text), email_text.count('!'), email_text.count('$')]).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(input_features)
    result = 'Phishing' if prediction[0] == 1 else 'Not Phishing'
    
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
