import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from recurrent_neural_network import InflationRNN, train_model, prepare_data  

file_path = './economic_data.csv'  # Update this path as necessary
df = pd.read_csv(file_path)

features = df[['CPIAUCSL', 'PPIACO', 'PCE']]
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

train_loader = prepare_data(X_train, y_train, batch_size=64)
test_loader = prepare_data(X_test, y_test, batch_size=64)  

model = InflationRNN(input_size=3, hidden_size=50, num_layers=2, output_size=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
train_model(model, train_loader, criterion, optimizer, num_epochs)

