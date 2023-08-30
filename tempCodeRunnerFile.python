from flask import Flask, request, jsonify
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler  # Import the scaler class

app = Flask(__name__)

# Load the pickled MLP model
loaded_mlp_model = joblib.load('mlp_model.pkl')

# Instantiate the scaler
scaler = StandardScaler()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    
    # Fit the scaler on your training data and then transform the features
    scaled_features = scaler.transform(features)  # Apply the same scaling as during training
    prediction = loaded_mlp_model.predict(scaled_features)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
