import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error  # Import mean_squared_error
from neural_network import SimpleInflationPredictor
import pandas as pd
import numpy as np

def create_dataloader(X, y, batch_size=64):
    tensor_X = torch.tensor(X, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(tensor_X, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

df = pd.read_csv('./Backend/Data/economic_data.csv')
features = df[['CPIAUCSL', 'PPIACO', 'PCE']].values
target = df['INFLRATE'].values
scaler = StandardScaler()
X_normalized = scaler.fit_transform(features)

k_folds = 10
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

best_params = {
    'lr': 0.007066923822087133,
    'num_layers': 2,
    'num_neurons': 56
}

epochs = 600
fold_performances_r2 = []
fold_performances_mse = []  # List to store MSE values for each fold

for fold, (train_idx, val_idx) in enumerate(kf.split(X_normalized, target)):
    print(f'Fold {fold+1}/{k_folds}')
    
    train_loader = create_dataloader(X_normalized[train_idx], target[train_idx])
    val_loader = create_dataloader(X_normalized[val_idx], target[val_idx])

    model = SimpleInflationPredictor(input_size=3, num_layers=best_params['num_layers'], num_neurons=best_params['num_neurons'])
    optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions.squeeze(), batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            output = model(batch_X)
            predictions.extend(output.view(-1).cpu().numpy())
            actuals.extend(batch_y.cpu().numpy())

    r2 = r2_score(actuals, predictions)
    mse = mean_squared_error(actuals, predictions)  # Calculate MSE
    fold_performances_r2.append(r2)
    fold_performances_mse.append(mse)  # Store MSE value
    print(f'R-squared Score for Fold {fold+1}: {r2 * 100:.2f}%')
    print(f'MSE for Fold {fold+1}: {mse:.4f}')

average_r2 = np.mean(fold_performances_r2)
std_dev_r2 = np.std(fold_performances_r2)
average_mse = np.mean(fold_performances_mse)  # Calculate average MSE
std_dev_mse = np.std(fold_performances_mse)  # Calculate standard deviation of MSE

print(f'Average R-squared Score: {average_r2 * 100:.2f}%')
print(f'Standard Deviation of R-squared Scores: {std_dev_r2 * 100:.2f}%')
print(f'Average MSE: {average_mse:.4f}')
print(f'Standard Deviation of MSE: {std_dev_mse:.4f}')
