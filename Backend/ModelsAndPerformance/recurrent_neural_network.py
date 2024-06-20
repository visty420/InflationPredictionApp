from datetime import datetime
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_squared_error  # Import mean_squared_error
from torch.utils.tensorboard import SummaryWriter

# Citirea datelor din CSV
file_path = './Backend/Data/complete_data.csv'
df = pd.read_csv(file_path)

# Selectarea caracteristicilor și a țintei
features = df[['CPIAUCSL', 'PPIACO', 'PCE', 'FEDFUNDS', 'UNRATE', 'GDP', 'M2SL', 'UMCSENT', 'Overall Wage Growth']]
target = df['INFLRATE']

# Scalarea caracteristicilor
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Salvarea scalerului
scaler_path = './Backend/SavedModels/rnn_scaler.gz'
joblib.dump(scaler, scaler_path)
print(f"Scaler saved to {scaler_path}")

# Crearea ferestrelor de date pentru RNN
window_size = 12

X, y = [], []
for i in range(len(features_scaled) - window_size):
    X.append(features_scaled[i:(i + window_size), :])
    y.append(target.iloc[i + window_size])

X = np.array(X)
y = np.array(y).reshape(-1, 1)

# Împărțirea setului de date în antrenament și testare
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definirea modelului RNN
class InflationRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(InflationRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])  
        return out

# Setările pentru log-uri
log_dir = "logs/architecture/" + datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(log_dir, exist_ok=True)

writer = SummaryWriter(log_dir)

# Inițializarea modelului, criteriului de pierdere și optimizerului
model = InflationRNN(input_size=9, hidden_size=100, num_layers=3, output_size=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Adăugarea modelului la SummaryWriter
sample_data = torch.tensor(X_train[:1], dtype=torch.float32)
writer.add_graph(model, sample_data)
writer.close()

# Crearea DataLoader-elor pentru antrenament și testare
train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)), batch_size=32, shuffle=False)

# Funcția de antrenament a modelului
def train_model(model, train_loader, criterion, optimizer, num_epochs=100):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# Antrenarea modelului
num_epochs = 400
train_model(model, train_loader, criterion, optimizer, num_epochs)

# # Salvarea modelului
# model_path = './Backend/SavedModels/rnn_model.pth'
# torch.save(model.state_dict(), model_path)
# print(f"Model state dictionary saved to {model_path}")

# Evaluarea modelului
model.eval()
with torch.no_grad():
    all_predictions = [model(inputs).view(-1).cpu().numpy() for inputs, targets in test_loader]
    all_targets = [targets.view(-1).cpu().numpy() for inputs, targets in test_loader]

all_predictions = np.concatenate(all_predictions)
all_targets = np.concatenate(all_targets)

# Calculează R-squared și MSE
r2 = r2_score(all_targets, all_predictions)
mse = mean_squared_error(all_targets, all_predictions)

print(f"R-squared Score: {r2 * 100:.2f}%")
print(f"MSE: {mse:.4f}")

# Salvarea întregului model
# torch.save(model, './Backend/SavedModels/rnn_model_complete.pth')
