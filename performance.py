import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from neural_network import InflationPredictor
import pandas as pd
import numpy as np

def create_dataloader(X, y, batch_size=64):
    tensor_X = torch.tensor(X, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(tensor_X, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Load and preprocess the dataset
df = pd.read_csv('./economic_data.csv')
features = df[['CPIAUCSL', 'PPIACO', 'PCE']].values
target = df['INFLRATE'].values
scaler = StandardScaler()
X_normalized = scaler.fit_transform(features)

# Setup for 10-fold Cross-Validation
k_folds = 10
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

best_params = {
    'lr': 0.007066923822087133,
    'num_layers': 2,
    'num_neurons': 56
}

epochs = 600
fold_performances = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_normalized, target)):
    print(f'Fold {fold+1}/{k_folds}')
    
    # Prepare data loaders
    train_loader = create_dataloader(X_normalized[train_idx], target[train_idx])
    val_loader = create_dataloader(X_normalized[val_idx], target[val_idx])

    # Initialize model, criterion, and optimizer
    model = InflationPredictor(input_size=3, num_layers=best_params['num_layers'], num_neurons=best_params['num_neurons'])
    optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions.squeeze(), batch_y)
            loss.backward()
            optimizer.step()

    # Evaluation loop
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            output = model(batch_X)
            predictions.extend(output.view(-1).cpu().numpy())
            actuals.extend(batch_y.cpu().numpy())

    # Compute R-squared score
    r2 = r2_score(actuals, predictions)
    fold_performances.append(r2)
    print(f'R-squared Score for Fold {fold+1}: {r2 * 100:.2f}%')

# Calculate average and standard deviation of R-squared scores
average_r2 = np.mean(fold_performances)
std_dev_r2 = np.std(fold_performances)
print(f'Average R-squared Score: {average_r2 * 100:.2f}%')
print(f'Standard Deviation of R-squared Scores: {std_dev_r2 * 100:.2f}%')
