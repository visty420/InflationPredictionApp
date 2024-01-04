import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
data = pd.read_csv("C:/Users/manea/Desktop/Licenta/InflationPredictionApp/cpi_data.csv")

selected_columns = ['DATE', 'CPILFESL']
data = data[selected_columns]

# Convertirea coloanei 'DATE' la tipul de dată datetime
data['DATE'] = pd.to_datetime(data['DATE'])

# Setează coloana 'DATE' ca index
data.set_index('DATE', inplace=True)

# Normalizarea datelor
scaler = StandardScaler()
data['CPILFESL'] = scaler.fit_transform(data['CPILFESL'].values.reshape(-1, 1))

# Divizarea datelor în seturi de antrenare și testare
train_size = int(len(data) * 0.80)
train_data, test_data = data.iloc[:train_size], data.iloc[train_size:]

# Funcția pentru pregătirea datelor în loturi
def create_sequences(data, seq_length):
    sequences = []
    targets = []

    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        target = data.iloc[i + seq_length:i + seq_length + 1]['CPILFESL']
        sequences.append(seq.values)
        targets.append(target.values)

    return torch.tensor(sequences), torch.tensor(targets)

# Creează secvențe pentru setul de antrenare
seq_length = 10
X_train, y_train = create_sequences(train_data, seq_length)

# Creează secvențe pentru setul de testare
X_test, y_test = create_sequences(test_data, seq_length)