import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import torch.nn as nn

df = pd.read_csv('./Backend/Data/complete_data.csv')

class InflationPredictor(nn.Module):
    def __init__(self, input_size, num_layers, num_neurons):
        super(InflationPredictor, self).__init__()
        layers = [nn.Linear(input_size, num_neurons), nn.ReLU()]
        for _ in range(1, num_layers):
            layers += [nn.Linear(num_neurons, num_neurons), nn.ReLU()]
        layers += [nn.Linear(num_neurons, 1)]
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

model = InflationPredictor(input_size=9, num_layers=3, num_neurons=98)
model.load_state_dict(torch.load('./Backend/SavedModels/model_sate_dict'))
model.eval()


features = df[['CPIAUCSL', 'PPIACO', 'PCE', 'FEDFUNDS', 'UNRATE', 'GDP', 'M2SL', 'UMCSENT', 'Overall Wage Growth']].values

scaler = StandardScaler()
scaler.fit(features)
features_normalized = scaler.transform(features)

features_tensor = torch.tensor(features_normalized, dtype=torch.float32)

with torch.no_grad():
    predictions = model(features_tensor).numpy()


df['Predicted INFLRATE'] = predictions

print(df)

df.to_csv('./Auxiliaries/predictions.csv', index=False)
