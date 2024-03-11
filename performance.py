import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from neural_network import InflationPredictor
import pandas as pd
import numpy as np



def create_dataloader(X, y, batch_size=64):
    tensor_X = torch.tensor(X, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(tensor_X, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


df = pd.read_csv('./economic_data.csv')  


features = df[['CPIAUCSL', 'PPIACO', 'PCE']].values
target = df['INFLRATE'].values


scaler = StandardScaler()
X_normalized = scaler.fit_transform(features)

# k-Fold Cross Validation setup
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)


best_params = { 
    'lr': 0.007066923822087133,
    'num_layers': 2,
    'num_neurons': 56
}


epochs = 250

# Store the fold performances
fold_performances = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_normalized, target)):
    print(f'Fold {fold+1}/{k_folds}')

   
    train_loader = create_dataloader(X_normalized[train_idx], target[train_idx])
    val_loader = create_dataloader(X_normalized[val_idx], target[val_idx])

    model = InflationPredictor(input_size=3, num_layers=best_params['num_layers'], num_neurons=best_params['num_neurons'])
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
    val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            predictions = model(batch_X)
            val_loss += criterion(predictions.squeeze(), batch_y).item()
    val_loss /= len(val_loader)
    fold_performances.append(val_loss)
    print(f'Validation Loss for Fold {fold+1}: {val_loss}')


average_loss = np.mean(fold_performances)
std_dev_loss = np.std(fold_performances)
print(f'Average Validation Loss: {average_loss}')
print(f'Standard Deviation of Validation Loss: {std_dev_loss}')

