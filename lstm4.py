# Improved Autoregressive Version (based on your original multi-step model structure)

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import time
import io

# Constants
seq_length = 30
output_horizon = 10
hidden_dim = 500
layer_dim = 2
learning_rate = 0.00005
training_size = 0.7
stock = '^GSPC'
period = '10y'

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

# Load data
try:
    cont_data_frame = pd.read_csv(f'data/{stock}_{period}_data_cont.csv')
except FileNotFoundError:
    stock_data = yf.Ticker(stock).history(period=period, interval='1d')
    sti = StockTechnicalIndicators(stock_data)
    cont_data_frame, _ = sti.get_dataframes(days=30, interval=1)
    cont_data_frame.to_csv(f'data/{stock}_{period}_data_cont.csv', index=False)

features = len(cont_data_frame.columns)
data_frame = cont_data_frame.copy()

# Normalize data to [0, 1]
for i in range(features):
    col = data_frame.columns[i]
    min_val = data_frame[col].min()
    max_val = data_frame[col].max()
    data_frame[col] = (data_frame[col] - min_val) / (max_val - min_val)

# Train/val split
data = data_frame.to_numpy()
train_size = int(training_size * len(data))
data_train = data[:train_size]
data_val = data[train_size:]

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length]  # Next step only
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

X, y = create_sequences(data_train, seq_length)
X_val, y_val = create_sequences(data_val, seq_length)

trainX = torch.tensor(X, dtype=torch.float32)
trainY = torch.tensor(y, dtype=torch.float32)
valX = torch.tensor(X_val, dtype=torch.float32)
valY = torch.tensor(y_val, dtype=torch.float32)

# Model
def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(layer_dim, x.size(0), hidden_dim).to(x.device)
        c0 = torch.zeros(layer_dim, x.size(0), hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # last time step only
        return out.unsqueeze(1)

model = LSTMModel(features, hidden_dim, layer_dim, features).to(device)
model.apply(init_weights)

# Training
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(100):
    model.train()
    output = model(trainX)
    loss = criterion(output.squeeze(1), trainY)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        model.eval()
        val_output = model(valX)
        val_loss = criterion(val_output.squeeze(1), valY)
        print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

# Autoregressive inference
def autoregressive_forecast(model, initial_seq, steps):
    model.eval()
    current_seq = initial_seq.clone()
    preds = []
    with torch.no_grad():
        for _ in range(steps):
            next_step = model(current_seq)
            preds.append(next_step.squeeze(1).cpu().numpy())
            current_seq = torch.cat((current_seq[:, 1:, :], next_step), dim=1)
    return np.array(preds)

# Forecast future
initial_input = valX[-1:].to(device)
auto_preds = autoregressive_forecast(model, initial_input, output_horizon)

# Plot
import matplotlib.pyplot as plt

def plot_autoregressive_preds(preds, df):
    time_steps = np.arange(preds.shape[0])
    num_features = preds.shape[1]

    plt.figure(figsize=(15, num_features * 2))
    for i in range(num_features):
        plt.subplot(num_features, 1, i + 1)
        plt.plot(time_steps, preds[:, i], label=f"Predicted {df.columns[i]}")
        plt.title(f"Autoregressive Forecast - {df.columns[i]}")
        plt.xlabel("Step")
        plt.ylabel(df.columns[i])
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_autoregressive_preds(auto_preds, data_frame)