from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle

# Load your trained model
with open('Notebooks/logistic_regression_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load feature dataset
feature_data = pd.read_csv('Notebook/features_binned.csv')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # HTML form for customer input

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the customer ID from the form
        customer_id = request.form.get('customer_id')
        
        # Find the corresponding features for that Customer ID
        customer_features = feature_data[feature_data['CustomerId'] == customer_id]
        
        if customer_features.empty:
            return jsonify({'error': 'Customer ID not found'})
        
        # Extract the relevant features
        features = customer_features[['Recency_WoE', 'Transaction_Frequency', 'Total_Transaction_Volume']]
        
        # Make prediction
        prediction = model.predict(features)
        
        # Return prediction result
        risk_label = 'High Risk' if prediction[0] == 1 else 'Low Risk'
        return jsonify({'Customer ID': customer_id, 'Risk Label': risk_label})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
