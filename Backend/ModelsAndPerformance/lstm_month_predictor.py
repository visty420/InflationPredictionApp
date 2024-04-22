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


# initial_data = np.array([[300.356,260.227,18229.6,4.33,3.4,26813.601,21209,64.9,5.9]
# , [301.509,258.669,18296.5,4.57,3.6,26896.738,21086.1,66.9,6.1]
# , [301.744,257.062,18282.6,4.65,3.5,26979.875,20861.3,62,6.5]
# , [303.032,256.908,18363.8,4.83,3.4,27063.012,20689.3,63.7,6.4]
# , [303.365,253.67,18407.8,5.06,3.7,27245.384,20803.2,59,6.4]
# , [304.003,253.86,18485.4,5.08,3.6,27427.756,20835.7,64.2,6]
# , [304.628,253.835,18595.4,5.12,3.5,27610.128,20841.8,71.5,5.7]
# , [306.187,257.68,18651.6,5.33,3.8,27721.62767,20798.4,69.4,5.2]
# , [307.288,258.934,18791.5,5.33,3.8,27833.12733,20723.7,67.9,5.2]
# , [307.531,255.121,18794.7,5.33,3.8,27944.627,20690.5,63.8,5.2]
# , [308.024,253.063,18867.8,5.33,3.7,28091.563,20730,61.3,5.2]
# , [308.742,249.767,19001.7,5.33,3.7,28238.499,20827.2,69.7,5.2]
# ])
months_to_predict = 12

initial_data = df.iloc[-12:, 1:-1].values

predictions = predict_future_inflation(model, scaler, initial_data, months_to_predict)
print(predictions)
