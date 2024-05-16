from datetime import datetime
import os
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter


df = pd.read_csv('./Backend/Data/complete_data.csv')


features = df[['CPIAUCSL', 'PPIACO', 'PCE', 'FEDFUNDS', 'UNRATE', 'GDP', 'M2SL', 'UMCSENT', 'Overall Wage Growth']].values
target = df['INFLRATE'].values


scaler = StandardScaler()
X_normalized = scaler.fit_transform(features)

# scaler_path = './Backend/SavedModels/9in_model_scaler.gz'
# joblib.dump(scaler, scaler_path)
# print(f"Scaler saved to {scaler_path}")

X_train, X_test, y_train, y_test = train_test_split(X_normalized, target, test_size=0.2, random_state=42)

best_params = {
    'lr': 0.000952163930520129,
    'num_layers': 3,
    'num_neurons': 98,
    'epochs': 663  
}

class InflationPredictor(nn.Module):
    def __init__(self, input_size, num_layers, num_neurons):
        super(InflationPredictor, self).__init__()
        layers = [nn.Linear(input_size, num_neurons), nn.ReLU()]
        for _ in range(1, num_layers):
            layers += [nn.Linear(num_neurons, num_neurons), nn.ReLU()]
        layers += [nn.Linear(num_neurons, 1)]
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

log_dir = "logs/architecture/" + datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(log_dir, exist_ok=True)

def create_dataloader(X, y, batch_size=64):
    tensor_X = torch.tensor(X, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.float32).view(-1, 1)  
    dataset = TensorDataset(tensor_X, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


train_loader = create_dataloader(X_train, y_train)
test_loader = create_dataloader(X_test, y_test)

writer = SummaryWriter(log_dir)

model = InflationPredictor(input_size=9, num_layers=best_params['num_layers'], num_neurons=best_params['num_neurons'])
optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])
criterion = nn.MSELoss()

sample_data = torch.tensor(X_train[:1], dtype=torch.float32)
writer.add_graph(model, sample_data)

for epoch in range(best_params['epochs']):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# model_path = './Backend/SavedModels/9in_model.pth'
# torch.save(model.state_dict(), model_path)
# print(f"Model saved to {model_path}")
writer.close()

new_features = np.array([[29.11,31.7,318.2,4,5,524.2403333,294.1,95.05,3.918944099]])  

row_features = np.array([[29.15,31.7,317.8,4,5.1,525.034,295.2,94.8,3.918944]])


new_features_normalized = scaler.transform(new_features)
new_features_tensor = torch.tensor(new_features_normalized, dtype=torch.float32)

row_normalized = scaler.transform(row_features)
row_tensor = torch.tensor(row_normalized, dtype=torch.float32)


model.eval()
with torch.no_grad():
    predicted_inflation_rate2 = model(new_features_tensor).item()
    inflrate_row2 = model(row_tensor).item()

print(f"Predicted Inflation Rate: {predicted_inflation_rate2}%")

