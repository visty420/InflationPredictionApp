
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
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

    # Training loop
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

    # Evaluation loop
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            output = model(batch_X)
            predictions.extend(output.view(-1).cpu().numpy())
            actuals.extend(batch_y.view(-1).cpu().numpy())

    r2 = r2_score(actuals, predictions)
    return r2

if __name__ == '__main__':
    # Load and preprocess the dataset
    df = pd.read_csv('./Backend/Data/augmented_economic_data.csv')
    features = df[['CPIAUCSL', 'PPIACO', 'PCE', 'FEDFUNDS', 'UNRATE', 'GDP', 'M2SL', 'UMCSENT', 'Overall Wage Growth']].values
    target = df['INFLRATE'].values
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(features)
    k_folds = 10
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Best hyperparameters from the Optuna study
    best_params = {
        'lr': 0.000952163930520129,
        'num_layers': 3,
        'num_neurons': 98,
        'epochs': 663
    }

    r2_scores = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_normalized, target)):
        print(f'Fold {fold+1}/{k_folds}')
        
        # Prepare data loaders
        train_loader = create_dataloader(X_normalized[train_idx], target[train_idx])
        val_loader = create_dataloader(X_normalized[val_idx], target[val_idx])

        # Initialize model, criterion, and optimizer with the best hyperparameters
        model = InflationPredictor(input_size=9, num_layers=best_params['num_layers'], num_neurons=best_params['num_neurons'])
        optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])
        criterion = nn.MSELoss()

        # Train and evaluate the model
        r2 = train_and_evaluate_model(train_loader, val_loader, input_size=9, num_layers=best_params['num_layers'], num_neurons=best_params['num_neurons'], lr=best_params['lr'], epochs=best_params['epochs'])
        r2_scores.append(r2)
        print(f'R-squared Score for Fold {fold+1}: {r2:.4f}')

    # Calculate average and standard deviation of R-squared scores
    average_r2 = np.mean(r2_scores)
    std_dev_r2 = np.std(r2_scores)
    print(f'Average R-squared Score: {average_r2 * 100:.2f}%')
    print(f'Standard Deviation of R-squared Scores: {std_dev_r2 * 100:.2f}%')

    # # Optuna optimization
    # study = optuna.create_study(direction='maximize')
    # study.optimize(objective, n_trials=100)  # Adjust the number of trials as needed
    
    # print(f'Best hyperparameters: {study.best_params}')
    