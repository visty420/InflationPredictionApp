import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np


df = pd.read_csv('./Backend/Data/augmented_economic_data.csv')

features = df[['CPIAUCSL', 'PPIACO', 'PCE', 'FEDFUNDS', 'UNRATE', 'GDP', 'M2SL', 'UMCSENT', 'Overall Wage Growth']].values
target = df['INFLRATE'].values.reshape(-1, 1)


scaler = StandardScaler()
X_normalized = scaler.fit_transform(features)
input_dim = X_normalized.shape[1]

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_rate):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        self.lstm.flatten_parameters()  
        out, (hn, cn) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

optimal_params = {
    'lr': 0.09903400349769692,
    'num_layers':2,
    'hidden_dim': 32,
    'dropout_rate': 0.02061234092208354,
    'batch_size': 29
}

def create_dataloader(X, y, batch_size=64, shuffle=True):
    tensor_X = torch.tensor(X, dtype=torch.float32).reshape(-1, 1, X.shape[1])
    tensor_y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    dataset = TensorDataset(tensor_X, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

X_train, X_test, y_train, y_test = train_test_split(X_normalized, target, test_size=0.2, random_state=42)

model = LSTMModel(input_dim, optimal_params['hidden_dim'], optimal_params['num_layers'], output_dim=1, dropout_rate=optimal_params['dropout_rate'])

optimizer = optim.Adam(model.parameters(), lr=optimal_params['lr'])
criterion = nn.MSELoss()


train_loader = create_dataloader(X_train, y_train, batch_size=optimal_params['batch_size'])
test_loader = create_dataloader(X_test, y_test, batch_size=optimal_params['batch_size'], shuffle=False)


num_epochs = 300  
model.train()
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')


model.eval()
predictions, actuals = [], []
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        predictions.extend(outputs.view(-1).cpu().numpy())
        actuals.extend(targets.view(-1).cpu().numpy())

r2 = r2_score(actuals, predictions)
print(f'R-squared: {r2 * 100:.2f}%')
