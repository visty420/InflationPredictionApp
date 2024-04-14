import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import torch.nn as nn

# Assuming 'df' is your loaded DataFrame with the consolidated data
# Load the data
df = pd.read_csv('./Backend/Data/complete_data.csv')

# Define the model class as you provided
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

# Initialize the model, assuming state_dict has been loaded
model = InflationPredictor(input_size=9, num_layers=3, num_neurons=98)
# model.load_state_dict(torch.load('path_to_your_model_state_dict.pt'))
model.eval()

# Define features
features = df[['CPIAUCSL', 'PPIACO', 'PCE', 'FEDFUNDS', 'UNRATE', 'GDP', 'M2SL', 'UMCSENT', 'Overall Wage Growth']].values

# Normalize the features
scaler = StandardScaler()
scaler.fit(features)
features_normalized = scaler.transform(features)

# Convert normalized features to tensor
features_tensor = torch.tensor(features_normalized, dtype=torch.float32)

# Predict INFLRATE for each row and store the results
with torch.no_grad():
    predictions = model(features_tensor).numpy()

# Add the predictions to the dataframe
df['Predicted INFLRATE'] = predictions

# Show the dataframe with the predictions
print(df)

# Save the dataframe to a new CSV if needed
df.to_csv('./predictions.csv', index=False)
