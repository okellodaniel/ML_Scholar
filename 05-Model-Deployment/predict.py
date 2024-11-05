import pickle
import logging

from flask import Flask
from flask import request
from flask import jsonify

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

model_file = 'model_C=1.0.bin'

app = Flask('Ping')

# Load the model
logging.info(f"Loading model from {model_file}")
with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)


@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()
    logging.info(f"Predicting probabilities for customer {customer}")
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[:, 1]
    churn = y_pred >= 0.5

    return jsonify({
        'prediction': float(y_pred),
        'churn': bool(churn)
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
    logging.info(f'App running')
