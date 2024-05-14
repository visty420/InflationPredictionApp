# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset

# class InflationRNN(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size):
#         super(InflationRNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#         out, _ = self.rnn(x, h0)
#         out = self.fc(out[:, -1, :])
#         return out

# def train_model(model, train_loader, criterion, optimizer, num_epochs=100):
#     for epoch in range(num_epochs):
#         for inputs, targets in train_loader:
            
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
            
            
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# def prepare_data(X, y, batch_size=64, test_size=0.2):
    
#     tensor_X = torch.tensor(X, dtype=torch.float32)
#     tensor_y = torch.tensor(y, dtype=torch.float32).view(-1, 1) 
    
   
#     dataset = TensorDataset(tensor_X, tensor_y)
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#     return loader

