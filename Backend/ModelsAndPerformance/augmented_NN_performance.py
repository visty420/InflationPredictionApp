
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import optuna
from augmented_neural_network import InflationPredictor

def create_dataloader(X, y, batch_size=64):
    tensor_X = torch.tensor(X, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.float32).view(-1, 1)  
    dataset = TensorDataset(tensor_X, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_and_evaluate_model(train_loader, val_loader, input_size, num_layers, num_neurons, lr, epochs):
    model = InflationPredictor(input_size=input_size, num_layers=num_layers, num_neurons=num_neurons)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            output = model(batch_X)
            predictions.extend(output.view(-1).cpu().numpy())
            actuals.extend(batch_y.view(-1).cpu().numpy())

    r2 = r2_score(actuals, predictions)
    mse = mean_squared_error(actuals, predictions)  # Calculate MSE
    return r2, mse  # Return both R² and MSE

if __name__ == '__main__':
    df = pd.read_csv('./Backend/Data/complete_data.csv')
    features = df[['CPIAUCSL', 'PPIACO', 'PCE', 'FEDFUNDS', 'UNRATE', 'GDP', 'M2SL', 'UMCSENT', 'Overall Wage Growth']].values
    target = df['INFLRATE'].values
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(features)
    k_folds = 10
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    best_params = {
        'lr': 0.000952163930520129,
        'num_layers': 3,
        'num_neurons': 98,
        'epochs': 663
    }

    r2_scores = []
    mse_scores = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_normalized, target)):
        print(f'Fold {fold+1}/{k_folds}')

        train_loader = create_dataloader(X_normalized[train_idx], target[train_idx])
        val_loader = create_dataloader(X_normalized[val_idx], target[val_idx])

        r2, mse = train_and_evaluate_model(train_loader, val_loader, input_size=9, num_layers=best_params['num_layers'], num_neurons=best_params['num_neurons'], lr=best_params['lr'], epochs=best_params['epochs'])
        r2_scores.append(r2)
        mse_scores.append(mse)
        print(f'R-squared Score for Fold {fold+1}: {r2:.4f}')
        print(f'MSE for Fold {fold+1}: {mse:.4f}')

    average_r2 = np.mean(r2_scores)
    std_dev_r2 = np.std(r2_scores)
    average_mse = np.mean(mse_scores)
    std_dev_mse = np.std(mse_scores)
    print(f'Average R-squared Score: {average_r2 * 100:.2f}%')
    print(f'Standard Deviation of R-squared Scores: {std_dev_r2 * 100:.2f}%')
    print(f'Average MSE: {average_mse:.4f}')
    print(f'Standard Deviation of MSE: {std_dev_mse:.4f}')
