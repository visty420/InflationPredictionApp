import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from neural_network import InflationPredictor
import pandas as pd
import numpy as np

# Load and preprocess the dataset
df = pd.read_csv('economic_data.csv')  # Adjust path as needed
features = df[['CPIAUCSL', 'PPIACO', 'PCE']].values
target = df['INFLRATE'].values

# Normalize the features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(features)

# Convert numpy arrays to tensors
tensor_X = torch.tensor(X_normalized, dtype=torch.float32)
tensor_y = torch.tensor(target, dtype=torch.float32).unsqueeze(1) # Ensure target tensor is the correct shape

# Model definition remains the same

# Initialize the model with the best hyperparameters
best_params ={ 
    'lr': 0.007066923822087133,
    'num_layers': 2,
    'num_neurons': 56
}
model = InflationPredictor(input_size=3, num_layers=best_params['num_layers'], num_neurons=best_params['num_neurons'])

# k-Fold Cross Validation setup
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
fold_performances = []

for fold, (train_idx, val_idx) in enumerate(kf.split(tensor_X)):
    print(f"Fold {fold + 1}/{k_folds}")
    train_features, val_features = tensor_X[train_idx], tensor_X[val_idx]
    train_targets, val_targets = tensor_y[train_idx], tensor_y[val_idx]
    train_loader = DataLoader(TensorDataset(train_features, train_targets), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_features, val_targets), batch_size=64, shuffle=False)
    
    # Re-initialize the model for each fold
    model = InflationPredictor(input_size=3, num_layers=best_params['num_layers'], num_neurons=best_params['num_neurons'])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])
    
    # Training loop for the fold
    for epoch in range(100):  # Adjust number of epochs as necessary
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
    
    # Evaluation on the validation set for the fold
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            predictions = model(batch_X)
            val_loss += criterion(predictions, batch_y).item()
    val_loss /= len(val_loader)
    fold_performances.append(val_loss)
    print(f"Validation Loss for Fold {fold + 1}: {val_loss}")

# Average performance across folds
average_performance = sum(fold_performances) / len(fold_performances)
print(f"Average Validation Loss Across {k_folds} Folds: {average_performance}")

# Further code for predictions remains unchanged
