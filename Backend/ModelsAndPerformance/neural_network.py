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
import optuna

from torch.utils.tensorboard import SummaryWriter

df = pd.read_csv('./Backend/Data/economic_data.csv')  

features = df[['CPIAUCSL', 'PPIACO', 'PCE']].values
target = df['INFLRATE'].values

scaler = StandardScaler()
X_normalized = scaler.fit_transform(features)

# scaler_path = './Backend/SavedModels/3in_model_scaler.gz'
# joblib.dump(scaler, scaler_path)
# print(f"Scaler saved to {scaler_path}")


X_train, X_test, y_train, y_test = train_test_split(X_normalized, target, test_size=0.2, random_state=42)



class SimpleInflationPredictor(nn.Module):
    def __init__(self, input_size, num_layers, num_neurons):
        super(SimpleInflationPredictor, self).__init__()
        layers = [nn.Linear(input_size, num_neurons), nn.ReLU()]
        for _ in range(1, num_layers):
            layers += [nn.Linear(num_neurons, num_neurons), nn.ReLU()]
        layers += [nn.Linear(num_neurons, 1)]
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

log_dir = "logs/architecture/" + datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(log_dir, exist_ok=True)

writer = SummaryWriter(log_dir)

def create_dataloader(X, y, batch_size=64):
    tensor_X = torch.tensor(X, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(tensor_X, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


train_loader = create_dataloader(X_train, y_train)
test_loader = create_dataloader(X_test, y_test)




def optimize_model(trial):
    
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    num_layers = trial.suggest_int('num_layers', 1, 5)
    num_neurons = trial.suggest_int('num_neurons', 10, 100)
    
    
    model = SimpleInflationPredictor(input_size=3, num_layers=num_layers, num_neurons=num_neurons)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
   
    for epoch in range(100):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
    
    
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            predictions = model(batch_X)
            test_loss += criterion(predictions.squeeze(), batch_y).item()
    test_loss /= len(test_loader)
    
    return test_loss


best_params ={ 
    'lr':0.007066923822087133,
    'num_layers':2,
    'num_neurons':56
}
model = SimpleInflationPredictor(input_size=3, num_layers=best_params['num_layers'], num_neurons=best_params['num_neurons'])
optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])

sample_data = torch.tensor(X_train[:1], dtype=torch.float32)
writer.add_graph(model, sample_data)


writer.close()

epochs = 400
criterion = nn.MSELoss()

for epoch in range(epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions.squeeze(), batch_y)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}: Loss = {loss.item()}')

# model_path = './Backend/SavedModels/3in_model.pth'
# torch.save(model.state_dict(), model_path)
# print(f"Model state dictionary saved to {model_path}")

current_month_features = np.array([309.685, 250.698, 19091])
current_month_features_normalized = scaler.transform([current_month_features])
current_month_tensor = torch.tensor(current_month_features_normalized, dtype=torch.float)

model.eval()  
with torch.no_grad():
    predicted_inflation_rate = model(current_month_tensor).item()

print(f"Predicted Inflation Rate: {predicted_inflation_rate}%")
