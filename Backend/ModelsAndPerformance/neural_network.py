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

df = pd.read_csv('./Backend/Data/economic_data.csv')  

# Define features and target
features = df[['CPIAUCSL', 'PPIACO', 'PCE']].values
target = df['INFLRATE'].values

# Normalize the features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(features)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_normalized, target, test_size=0.2, random_state=42)

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


def create_dataloader(X, y, batch_size=64):
    tensor_X = torch.tensor(X, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(tensor_X, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


train_loader = create_dataloader(X_train, y_train)
test_loader = create_dataloader(X_test, y_test)

# Define the Optuna optimization function
def optimize_model(trial):
    
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    num_layers = trial.suggest_int('num_layers', 1, 5)
    num_neurons = trial.suggest_int('num_neurons', 10, 100)
    
    
    model = InflationPredictor(input_size=3, num_layers=num_layers, num_neurons=num_neurons)
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


# Re-train the model with the best hyperparameters
best_params ={ 
    'lr':0.007066923822087133,
    'num_layers':2,
    'num_neurons':56
}
model = InflationPredictor(input_size=3, num_layers=best_params['num_layers'], num_neurons=best_params['num_neurons'])
optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])

epochs = 400
criterion = nn.MSELoss()
# Final training loop
for epoch in range(epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions.squeeze(), batch_y)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}: Loss = {loss.item()}')

current_month_features = np.array([309.685, 250.698, 19091])
current_month_features_normalized = scaler.transform([current_month_features])
current_month_tensor = torch.tensor(current_month_features_normalized, dtype=torch.float)

model.eval()  
with torch.no_grad():
    predicted_inflation_rate = model(current_month_tensor).item()

torch.save(model, './Backend/SavedModels/inflation_predictor_threeinputs_model.pth')
joblib.dump(scaler, './Backend/SavedModels/three_inputs_nlp_scaler.gz')

print(f"Predicted Inflation Rate: {predicted_inflation_rate}%")
