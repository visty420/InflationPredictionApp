import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('./augmented_economic_data.csv')

# Define features and target
features = df[['CPIAUCSL', 'PPIACO', 'PCE', 'FEDFUNDS', 'UNRATE', 'GDP', 'M2SL', 'UMCSENT', 'Overall Wage Growth']].values
target = df['INFLRATE'].values.reshape(-1, 1)

# Normalize the features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(features)
input_dim = X_normalized.shape[1]

# LSTM Model Definition
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_rate):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate)
        # Multiply hidden_dim by 2 if LSTM is bidirectional
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        self.lstm.flatten_parameters()  # This can sometimes solve issues related to GPU usage and multiprocessing
        out, (hn, cn) = self.lstm(x)
        # If your LSTM is bidirectional, concatenate the final forward and backward hidden states
        # out = self.fc(torch.cat((hn[-2,:,:], hn[-1,:,:]), dim = 1))
        out = self.fc(hn[-1])
        return out

# Hyperparameters from Optuna study
optimal_params = {
    'lr': 0.09903400349769692,
    'num_layers':3,
    'hidden_dim': 29,
    'dropout_rate': 0.02061234092208354,
    'batch_size': 29
}

# DataLoaders
def create_dataloader(X, y, batch_size=64, shuffle=True):
    tensor_X = torch.tensor(X, dtype=torch.float32).reshape(-1, 1, X.shape[1])
    tensor_y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    dataset = TensorDataset(tensor_X, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_normalized, target, test_size=0.2, random_state=42)

# Initialize the model
model = LSTMModel(input_dim, optimal_params['hidden_dim'], optimal_params['num_layers'], output_dim=1, dropout_rate=optimal_params['dropout_rate'])

# Optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=optimal_params['lr'])
criterion = nn.MSELoss()

# Create DataLoaders
train_loader = create_dataloader(X_train, y_train, batch_size=optimal_params['batch_size'])
test_loader = create_dataloader(X_test, y_test, batch_size=optimal_params['batch_size'], shuffle=False)

# Training Loop
num_epochs = 300  # Adjust this based on your needs
model.train()
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Evaluation
model.eval()
predictions, actuals = [], []
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        predictions.extend(outputs.view(-1).cpu().numpy())
        actuals.extend(targets.view(-1).cpu().numpy())

# Calculate R-squared
r2 = r2_score(actuals, predictions)
print(f'R-squared: {r2 * 100:.2f}%')
