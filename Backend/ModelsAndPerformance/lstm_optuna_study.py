import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

df = pd.read_csv('./Backend/Data/complete_data.csv')

class OptunaLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_rate):
        super(OptunaLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    hidden_dim = trial.suggest_int('hidden_dim', 20, 200)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    batch_size = trial.suggest_int('batch_size', 16, 128)

    model = OptunaLSTM(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=1, dropout_rate=dropout_rate)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(10):  
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y.view(-1, 1))  
            optimizer.step()

    
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            output = model(batch_X)
            predictions.extend(output.view(-1).cpu().numpy())
            actuals.extend(batch_y.view(-1).cpu().numpy())

 
    r2 = r2_score(actuals, predictions)
    return r2


features = df[['CPIAUCSL', 'PPIACO', 'PCE', 'FEDFUNDS', 'UNRATE', 'GDP', 'M2SL', 'UMCSENT', 'Overall Wage Growth']].values
target = df['INFLRATE'].values.reshape(-1, 1)

scaler = StandardScaler()
X_normalized = scaler.fit_transform(features)
input_dim = X_normalized.shape[1]


X_train, X_test, y_train, y_test = train_test_split(X_normalized, target, test_size=0.2, random_state=42)


X_train_tensor = torch.tensor(X_train, dtype=torch.float32).reshape(-1, 1, input_dim)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).reshape(-1, 1, input_dim)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)


train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=64, shuffle=False)


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=500)  

print('Best trial:', study.best_trial.params)
