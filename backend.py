import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

# App Initialization
app = Flask(__name__)

# Load The Models
with open('preprocess.pkl', 'rb') as file_1:
    preprocess = joblib.load(file_1)

model_dnn = load_model('TC_model.h5')

# Route : Homepage
@app.route('/')
def home():
    return '<h1> It Works! </h1>'

@app.route('/predict', methods=['POST'])

def titanic_predict():
    args = request.json
    data_inf = {
        'customerID': args.get('customerID'),               # 'nama di csv': nama variable penerima di streamlit
        'gender': args.get('gender'),
        'SeniorCitizen': args.get('SeniorCitizen'),
        'Partner': args.get('Partner'),
        'Dependents': args.get('Dependents'),
        'tenure': args.get('tenure'),
        'PhoneService': args.get('PhoneService'),
        'MultipleLines': args.get('MultipleLines'),
        'InternetService': args.get('InternetService'),
        'OnlineSecurity': args.get('OnlineSecurity'),
        'OnlineBackup': args.get('OnlineBackup'),
        'DeviceProtection': args.get('DeviceProtection'),
        'TechSupport': args.get('TechSupport'),
        'StreamingTV': args.get('StreamingTV'),
        'StreamingMovies': args.get('StreamingMovies'),
        'Contract': args.get('Contract'),
        'PaperlessBilling': args.get('PaperlessBilling'),
        'PaymentMethod': args.get('PaymentMethod'),
        'MonthlyCharges': args.get('MonthlyCharges'),
        'TotalCharges': args.get('TotalCharges')
    }

    print('[DEBUG] Data Inference : ', data_inf)

    # Transform Inference-Set
    data_inf = pd.DataFrame([data_inf])
    data_inf_transform = preprocess.transform(data_inf)
    y_pred_inf = model_dnn.predict(data_inf_transform)
    y_pred_inf = np.where(y_pred_inf >= 0.5, 1, 0)

    if y_pred_inf == 0:
        label = 'Not Churn'
    else:
        label = 'Churn'

    print('[DEBUG] Result : ', y_pred_inf, label)
    print('')

    response = jsonify(
        result = str(y_pred_inf),
        label_names = label
    )

    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0')      # range port http 500 s/d 8000 tapi bukan port yang default http