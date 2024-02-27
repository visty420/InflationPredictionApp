import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import optuna

# Load the dataset
df = pd.read_csv('economic_data.csv')  # Update the path to match where your file is located

features = df[['CPIAUCSL', 'PPIACO', 'PCE']].values
target = df['INFLRATE'].values

# Normalize the features
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(features_normalized, target, test_size=0.2, random_state=42)

class DynamicInflationPredictor(nn.Module):
    def __init__(self, input_size, num_layers, num_neurons):
        super(DynamicInflationPredictor, self).__init__()
        layers = [nn.Linear(input_size, num_neurons), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(num_neurons, num_neurons), nn.ReLU()]
        layers.append(nn.Linear(num_neurons, 1))
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)

# Define the objective function for Optuna
def objective(trial):
    # Hyperparameters to tune
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    num_layers = trial.suggest_int('num_layers', 1, 5)
    num_neurons = trial.suggest_int('num_neurons', 10, 100)

    model = DynamicInflationPredictor(input_size=3, num_layers=num_layers, num_neurons=num_neurons)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Training loop
    for epoch in range(100):  
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.squeeze(), target)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float)
        predictions = model(X_test_tensor)
        test_loss = criterion(predictions.squeeze(), y_test_tensor)

    return test_loss.item()

# Run the Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)  # Adjust the number of trials if needed

print("Best hyperparameters:", study.best_trial.params)

# Use the best hyperparameters to define your final model
best_params = study.best_trial.params
model = DynamicInflationPredictor(input_size=3, num_layers=best_params['num_layers'], num_neurons=best_params['num_neurons'])

