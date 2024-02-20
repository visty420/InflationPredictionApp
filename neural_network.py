import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('C:/Users/manea/Desktop/Licenta/InflationPredictionApp/economic_data.csv')


features = df[['CPIAUCSL', 'PPIACO', 'PCE']].values
target = df['INFLRATE'].values

# Normalize the features
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(features_normalized, target, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float)
X_test = torch.tensor(X_test, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.float)
y_test = torch.tensor(y_test, dtype=torch.float)

# Create dataloaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


class InflationPredictor(nn.Module):
    def __init__(self, input_size):
        super(InflationPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = InflationPredictor(input_size=3)
criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 1000
for epoch in range(epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output.squeeze(), labels)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')


current_month_features = np.array([309.685, 250.698, 19091])
current_month_features_normalized = scaler.transform([current_month_features])
current_month_tensor = torch.tensor(current_month_features_normalized, dtype=torch.float)

model.eval()  
with torch.no_grad():
    predicted_inflation_rate = model(current_month_tensor).item()

print(f"Predicted Inflation Rate: {predicted_inflation_rate}%")