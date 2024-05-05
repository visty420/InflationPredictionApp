import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from lstm import LSTMModel

def load_model_and_scaler(model_path, scaler_path):
    model = LSTMModel(input_dim=9, hidden_dim=31, num_layers=2, output_dim=1, dropout_rate=0.16652487982507866)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    scaler = joblib.load(scaler_path)
    return model, scaler

df = pd.read_csv('./Backend/Data/complete_data.csv')
feature_columns = df.columns[1:]

def predict_future_inflation(model, scaler, initial_data, months_to_predict):
    predictions = []

    for _ in range(months_to_predict):
        # Get the last sequence from initial data for prediction
        current_sequence = initial_data[-1].reshape(1, -1)

        # Scale the current sequence
        current_sequence_scaled = scaler.transform(current_sequence)

        # Convert to tensor
        sequence_tensor = torch.tensor(current_sequence_scaled, dtype=torch.float32).unsqueeze(0)

        # Predict
        with torch.no_grad():
            prediction = model(sequence_tensor).cpu().numpy().flatten()[0]

        # Inverse transform the prediction by creating a full feature array
        full_feature_array = np.zeros((1, len(scaler.scale_)))
        full_feature_array[0, 0] = prediction  # We only have a prediction for the first feature
        predicted_inflation = scaler.inverse_transform(full_feature_array)[0, 0]

        predictions.append(predicted_inflation)
        new_data_point = np.hstack((predicted_inflation, initial_data[-1, 1:]))
        initial_data = np.vstack((initial_data[1:], new_data_point))  # Drop the oldest, append the new

    return predictions

model_path = './Backend/SavedModels/lstm_model_state_dict.pth'
scaler_path = './Backend/SavedModels/lstm_scaler.gz'
model, scaler = load_model_and_scaler(model_path, scaler_path)

months_to_predict = 12

initial_data = df.iloc[-12:, 1:-1].values

model.eval()
predictions = predict_future_inflation(model, scaler, initial_data, months_to_predict)
print(predictions)
