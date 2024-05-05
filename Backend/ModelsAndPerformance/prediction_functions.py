import joblib
from statsmodels.tsa.arima.model import ARIMAResults
import torch

def predict_arima(steps=5):
    model_path = './Backend/SavedModels/inflation_arima_model.gz'
    model = ARIMAResults.load(model_path)
    forecast = model.forecast(steps=steps)
    return forecast.tolist()

def predict_nlp_nine_inputs_inflation(input_data):
    scaler = joblib.load('./Backend/SavedModels/nlp_scaler.gz')
    input_data_normalized = scaler.transform([input_data])
    input_tensor = torch.tensor(input_data_normalized, dtype=torch.float32)
    model = torch.load('./Backend/SavedModels/nlp')
    model.eval()
    with torch.no_grad():
        prediction = model(input_tensor).item()
    return prediction

def predict_lstm(input_data):
    scaler = joblib.load('./Backend/SavedModels/lstm_scaler.gz')
    model = torch.load('./Backend/SavedModels/lstm_model_state_dict.pth')
    model.eval()
    input_data_normalized = scaler.transform([input_data])
    input_tensor = torch.tensor(input_data_normalized, dtype=torch.float32)
    with torch.no_grad():
        prediction = model(input_tensor.unsqueeze(0)).numpy().flatten() 
    return prediction.tolist()

def predict_inflation_rnn(input_data):
    model = torch.load('./Backend/SavedModels/rnn.pth')
    model.eval()
    scaler = joblib.load('./Backend/SavedModels/rnn_scaler.gz')
    input_data_normalized = scaler.transform([input_data])
    input_tensor = torch.tensor(input_data_normalized, dtype=torch.float32)
    with torch.no_grad():
        prediction = model(input_tensor.unsqueeze(0)).numpy().flatten() 
    return prediction.tolist()

def predict_inflation_three_inputs(input_data):
    model = torch.load('./Backend/SavedModels/inflation_predictor_threeinputs_model.pth')
    scaler = joblib.load('./Backend/SavedModels/three_inputs_nlp_scaler.gz') 
    input_data_normalized = scaler.transform([input_data])
    input_tensor = torch.tensor(input_data_normalized, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        prediction = model(input_tensor).item()
    return prediction