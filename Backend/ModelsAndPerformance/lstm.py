import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np


def create_sequences(X, y, sequence_length):
    Xs, ys = [], []
    for i in range(len(X) - sequence_length):
        Xs.append(X[i:(i + sequence_length), :])
        ys.append(y[i + sequence_length])
    return np.array(Xs), np.array(ys)


df = pd.read_csv('./Backend/Data/complete_data.csv')
features = df[['CPIAUCSL', 'PPIACO', 'PCE', 'FEDFUNDS', 'UNRATE', 'GDP', 'M2SL', 'UMCSENT', 'Overall Wage Growth']].values
target = df['INFLRATE'].values.reshape(-1, 1)

scaler = StandardScaler()
X_normalized = scaler.fit_transform(features)


sequence_length = 12  
X_seq, y_seq = create_sequences(X_normalized, target.flatten(), sequence_length)


X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

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
    'lr': 0.05689607838852533,
    'num_layers': 2,
    'hidden_dim': 31,
    'dropout_rate': 0.16652487982507866,
    'batch_size': 33
}
model = LSTMModel(input_dim=X_train.shape[2], hidden_dim=optimal_params['hidden_dim'], num_layers=optimal_params['num_layers'], output_dim=1, dropout_rate=optimal_params['dropout_rate'])
optimizer = optim.Adam(model.parameters(), lr=optimal_params['lr'])
criterion = nn.MSELoss()

def create_dataloader(X, y, batch_size=64, shuffle=True):
    tensor_X = torch.tensor(X, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(tensor_X, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

train_loader = create_dataloader(X_train, y_train, batch_size=optimal_params['batch_size'])
test_loader = create_dataloader(X_test, y_test, batch_size=optimal_params['batch_size'], shuffle=False)

num_epochs = 163
model.train()
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

torch.save(model.state_dict(), './Backend/SavedModels/lstm_model_state_dict.pth')
joblib.dump(scaler, './Backend/SavedModels/lstm_scaler.gz')

model.eval()
predictions, actuals = [], []
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        predictions.extend(outputs.view(-1).cpu().numpy())
        actuals.extend(targets.view(-1).cpu().numpy())

r2 = r2_score(actuals, predictions)
print(f'R-squared: {r2 * 100:.2f}%')
