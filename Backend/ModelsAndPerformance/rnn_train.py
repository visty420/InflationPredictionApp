import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score

# Încărcarea și pregătirea datelor
file_path = './Backend/Data/complete_data.csv'
df = pd.read_csv(file_path)

features = df[['CPIAUCSL', 'PPIACO', 'PCE', 'FEDFUNDS', 'UNRATE', 'GDP', 'M2SL', 'UMCSENT', 'Overall Wage Growth']]
target = df['INFLRATE']

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

window_size = 12
X, y = [], []
for i in range(len(features_scaled) - window_size):
    X.append(features_scaled[i:(i + window_size), :])
    y.append(target.iloc[i + window_size])

X = np.array(X)
y = np.array(y).reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definirea modelului RNN
class InflationRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(InflationRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Inițializează starea ascunsă la zero la începutul fiecărei epoci
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])  # Luăm doar ultimul output pentru predicție
        return out


# Inițializarea și antrenarea modelului
model = InflationRNN(input_size=9, hidden_size=50, num_layers=2, output_size=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)), batch_size=64, shuffle=False)

# Funcția de antrenare
def train_model(model, train_loader, criterion, optimizer, num_epochs=100):
    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

num_epochs = 400
train_model(model, train_loader, criterion, optimizer, num_epochs)

model.eval()
with torch.no_grad():
    all_predictions = [model(inputs).view(-1).cpu().numpy() for inputs, targets in test_loader]
    all_targets = [targets.view(-1).cpu().numpy() for inputs, targets in test_loader]

torch.save(model, './Backend/SavedModels/inflation_rnn_model_full.pth')

all_predictions = np.concatenate(all_predictions)
all_targets = np.concatenate(all_targets)
r2 = r2_score(all_targets, all_predictions)
print(f"R-squared Score: {r2 * 100:.2f}%")
