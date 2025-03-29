import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from early_stopping_pytorch import EarlyStopping
from stock_technical_indicators import StockTechnicalIndicators
import yfinance as yf
import time
import keyboard
import pandas as pd
import shutil

# Constants
seq_length = 30 # Number of time steps to look back
num_epochs = 10000 # Number of epochs
hidden_dim = 500 # Number of hidden neurons
layer_dim = 2 # Number of hidden layers
learning_rate = 0.00005 # Learning rate
training_size = 0.70  # Percentage of data to use for training

# Early stopping
patience = 1000
delta = 0.0

# Stock data
stock = '^GSPC'
period = '1y'
# Check data folder for csv file of stock data
try:
    cont_data_frame = pd.read_csv(f'data/{stock}_data_cont.csv')
    binary_data_frame = pd.read_csv(f'data/{stock}_data_binary.csv')
    print(f'Loaded Processed {stock} data from file')
except FileNotFoundError:
    try:
        stock_data = pd.read_csv(f'data/{stock}_data.csv')
        print(f'Loaded {stock} data from file')
    except FileNotFoundError:
        stock_data = yf.Ticker(stock).history(period=period, interval='1d')
        stock_data.to_csv(f'data/{stock}_data.csv')
        print(f'Loaded {stock} data from Yahoo Finance') 
    sti = StockTechnicalIndicators(stock_data)
    cont_data_frame, binary_data_frame = sti.get_dataframes(days=30, interval=1)
    cont_data_frame.to_csv(f'data/{stock}_data_cont.csv', index=False)
    binary_data_frame.to_csv(f'data/{stock}_data_binary.csv', index=False)
    print(f'Saved Processed {stock} data to file')
data_frame = cont_data_frame

features = len(data_frame.columns)
    
for i in range(features):
    # Normalize data to be between 0 and 1
    print(f'Min: {data_frame.iloc[:, i].min()}, Max: {data_frame.iloc[:, i].max()}')
    # if data_frame.iloc[:, i].max() == data_frame.iloc[:, i].min():
    #     data_frame.iloc[:, i] = 0.0
    #     continue
    data_frame.iloc[:, i] = (data_frame.iloc[:, i] - data_frame.iloc[:, i].min()) / (data_frame.iloc[:, i].max() - data_frame.iloc[:, i].min())
    
data_frame.to_csv(f'data/{stock}_data_cont_normalized.csv', index=False)
print(f'Saved Normalized {stock} data to file')

# Plot data_frame['VAL'] for len(data_frame) days
# plt.figure(figsize=(12, 6))
# plt.plot(data_frame['VAL'])
# plt.title(f'{stock} Stock Price')
# plt.xlabel('Time Step')
# plt.ylabel('Price')
# plt.show()

# def data_grabber(time_step_index, feauture_index):
#     # return np.sin(time_step_index*(feauture_index + 1)) # This is a dummy function. Replace this with your data grabber function
#     # return feauture_index
#     return data_frame.iloc[time_step_index, feauture_index]

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA is available. Using GPU.')
else:
    device = torch.device('cpu')
    print('CUDA is not available. Using CPU.')
    
# def fromiter(x, i):
#     print(f'x: {x}, i: {i}')
#     return np.fromiter((data_grabber(xi, i) for xi in x), x.dtype)

torch.set_default_device(device)

np.random.seed(0)
torch.manual_seed(0)

def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data)-seq_length):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def plot_predictions(original, predicted, time_steps, data_frame, title):
    __, axs = plt.subplots(4, features//3, figsize=(18, 8))
    for i in range(features):
        data_value_label = data_frame.columns[i]
        row = i // 3
        col = i % 3
        axs[row, col].plot(time_steps, original[:, i], label=f'Original Data {data_value_label}')
        axs[row, col].plot(time_steps, predicted.detach().numpy()[:, i], label=f'Predicted Data {data_value_label}', linestyle='--')
        axs[row, col].set_title(f'LSTM Model Predictions vs. Original Data {data_value_label}')
        axs[row, col].set_xlabel('Time Step')
        axs[row, col].set_ylabel(data_value_label)
        axs[row, col].legend()
    plt.title(title)
    plt.savefig("./output/" + title + ".png")

# Generate synthetic data
# t is a list of indices from 0 to len(data_frame)
t_full = np.linspace(0, len(data_frame), len(data_frame), endpoint=False, dtype=int)
train_upper_bound = int(training_size*len(t_full))
t_train = t_full[:train_upper_bound]
# data = np.array([fromiter(t_train, i) for i in range(features)]).T  # Generate 10 input features
data_full = data_frame.to_numpy()
# Slice data into training and validation data
data = data_full[:train_upper_bound]
X, y = create_sequences(data, seq_length)
# X, y = create_sequences(data, seq_length)

trainX = torch.tensor(X, dtype=torch.float32)
trainY = torch.tensor(y, dtype=torch.float32)

# Generate synthetic validation data
# t_val = t_full[-100:]  # Use 100 data points for validation
t_val = t_full[train_upper_bound:]
# data_val = np.array([fromiter(t_val, i) for i in range(features)]).T  # Generate 10 input features
data_val = data_full[train_upper_bound:]
X_val, y_val = create_sequences(data_val, seq_length)

valX = torch.tensor(X_val, dtype=torch.float32)
valY = torch.tensor(y_val, dtype=torch.float32)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h0=None, c0=None):
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out, hn, cn
    
model = LSTMModel(input_dim=features, hidden_dim=hidden_dim, layer_dim=layer_dim, output_dim=features)  # Update input_dim and output_dim
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

early_stopper = EarlyStopping(patience=patience, verbose=True, path='checkpoints/best_current_checkpoint.pt', delta=delta)

h0, c0 = None, None

start_time = time.time()
last_time = start_time
current_time = time.time()
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    outputs, h0, c0 = model(trainX, h0, c0)

    loss = criterion(outputs, trainY)
    loss.backward()
    optimizer.step()

    h0 = h0.detach()
    c0 = c0.detach()
    
    # Get validation loss
    with torch.no_grad():
        model.eval()
        h0_val, c0_val = None, None  # Reset hidden and cell states for validation
        predicted, _, _ = model(valX, h0_val, c0_val)
        val_loss = criterion(predicted, valY)
        val_loss_float = val_loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss_float:.4f}')
    if (epoch+1) % 10 == 0:
        early_stopper(val_loss_float, model)
        current_time = time.time()
        print(f'Time: {current_time - last_time:.2f} seconds')
        start_time = last_time
        if early_stopper.early_stop:
            print('Early stopping')
            break
    
    # Allow user to pause training and resume later (handle KeyboardInterrupt)
    # if keyboard.is_pressed('p'):
    #     print('Training paused...')
    #     # Save model
    #     torch.save(model.state_dict(), f'checkpoints/last_checkpoint.pt')
    #     print('Model saved')
    #     # Show current model performance
    #     model.eval()
    #     predicted, _, _ = model(trainX, h0, c0)
    #     original = data[seq_length:]
    #     time_steps = np.arange(seq_length, len(data))
    #     predicted = predicted.cpu()
    #     plot_predictions(original, predicted, time_steps, data_frame, 'LSTM Model Predictions vs. Original Data (Training)')
    #     # Show current validation performance
    #     predicted_val, _, _ = model(valX, h0_val, c0_val)
    #     original_val = data_val[seq_length:]
    #     time_steps_val = np.arange(seq_length, len(data_val))
    #     predicted_val = predicted_val.cpu()
    #     plot_predictions(original_val, predicted_val, time_steps_val, data_frame, 'LSTM Model Predictions vs. Original Data (Validation)')
    #     plt.show()
    #     # Resume training
    #     input('Press enter to continue training...')
    #     print('Training resumed...')
    #     continue
    # if keyboard.is_pressed('q'):
    #     print('Training stopped')
    #     break
    
current_time = time.time()  
total_time = current_time - start_time
print(f'Training stopped. Total time: {total_time:.2f} seconds')
# Copy best_current_checkpoint.pt to {stock}_final.pt
# Check if best_current_checkpoint.pt exists
print('Copying best_current_checkpoint.pt to {stock}_final.pt')
shutil.copy('checkpoints/best_current_checkpoint.pt', f'checkpoints/{stock}_final.pt')

model = LSTMModel(input_dim=features, hidden_dim=hidden_dim, layer_dim=layer_dim, output_dim=features)
model.load_state_dict(torch.load(f'checkpoints/{stock}_final.pt'))

# Plot the predictions for training data
model.eval()
predicted, _, _ = model(trainX, h0, c0)

original = data[seq_length:]
time_steps = np.arange(seq_length, len(data))

predicted = predicted.cpu()

plot_predictions(original, predicted, time_steps, data_frame, 'LSTM Model Predictions vs. Original Data (Training)')

# fig, axs = plt.subplots(4, features//3, figsize=(18, 8))
# for i in range(features):
#     data_value_label = data_frame.columns[i]
#     row = i // 3
#     col = i % 3
#     axs[row, col].plot(time_steps, original[:, i], label=f'Original Data {data_value_label}')
#     axs[row, col].plot(time_steps, predicted.detach().numpy()[:, i], label=f'Predicted Data {data_value_label}', linestyle='--')
#     axs[row, col].set_title(f'LSTM Model Predictions vs. Original Data (Training) {data_value_label}')
#     axs[row, col].set_xlabel('Time Step')
#     axs[row, col].set_ylabel(data_value_label)
#     axs[row, col].legend()
# plt.title('LSTM Model Predictions vs. Original Data (Training)')


# plt.figure(figsize=(12, 6))
# for i in range(features):
#     plt.plot(time_steps, original[:, i], label=f'Original Data {i+1}')
#     plt.plot(time_steps, predicted.detach().numpy()[:, i], label=f'Predicted Data {i+1}', linestyle='--')
# plt.title('LSTM Model Predictions vs. Original Data (Training)')
# plt.xlabel('Time Step')
# plt.ylabel('Value')
# plt.legend()

# Plot the predictions for validation data
h0, c0 = None, None  # Reset hidden and cell states for validation
predicted_val, _, _ = model(valX, h0, c0)

original_val = data_val[seq_length:]
time_steps_val = np.arange(seq_length, len(data_val))

predicted_val = predicted_val.cpu()

plot_predictions(original_val, predicted_val, time_steps_val, data_frame, 'LSTM Model Predictions vs. Original Data (Validation)')

# fig, axs = plt.subplots(4, features//3, figsize=(18, 8))
# for i in range(features):
#     data_value_label = data_frame.columns[i]
#     row = i // 3
#     col = i % 3
#     axs[row, col].plot(time_steps_val, original_val[:, i], label=f'Original Data {data_value_label}')
#     axs[row, col].plot(time_steps_val, predicted_val.detach().numpy()[:, i], label=f'Predicted Data {data_value_label}', linestyle='--')
#     axs[row, col].set_title(f'LSTM Model Predictions vs. Original Data (Validarion) {data_value_label}')
#     axs[row, col].set_xlabel('Time Step')
#     axs[row, col].set_ylabel(data_value_label)
#     axs[row, col].legend()
# plt.title('LSTM Model Predictions vs. Original Data (Validation)')

# plt.figure(figsize=(12, 6))
# for i in range(features):
#     plt.plot(time_steps_val, original_val[:, i], label=f'Original Data {i+1}')
#     plt.plot(time_steps_val, predicted_val.detach().numpy()[:, i], label=f'Predicted Data {i+1}', linestyle='--')
# plt.title('LSTM Model Predictions vs. Original Data (Validation)')
# plt.xlabel('Time Step')
# plt.ylabel('Value')
# plt.legend()

plt.show()