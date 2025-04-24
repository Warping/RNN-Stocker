import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
# from early_stopping_pytorch import EarlyStopping
from stock_technical_indicators import StockTechnicalIndicators
import yfinance as yf
import time
import pandas as pd
import io
import argparse

# Constants
seq_length = 30 # Number of time steps to look back
avg_period = 30 # Number of days to average over
num_epochs = 10000 # Number of epochs
hidden_dim = 500 # Number of hidden neurons
layer_dim = 2 # Number of hidden layers
learning_rate = 0.00005 # Learning rate
training_size = 0.70  # Percentage of data to use for training

# Early stopping
patience = 100
delta = 0.0

# Stock data
stock = '^GSPC'
period = '10y'

# Parse command line arguments
parser = argparse.ArgumentParser(description='LSTM Stock Price Prediction')
parser.add_argument('--seq_length', type=int, default=seq_length, help=f'Number of time steps to look back -- Default= {seq_length}')
parser.add_argument('--avg_period', type=int, default=avg_period, help=f'Number of days to normalize and average over -- Default= {avg_period}')
parser.add_argument('--num_epochs', type=int, default=num_epochs, help=f'Number of epochs -- Default= {num_epochs}')
parser.add_argument('--hidden_dim', type=int, default=hidden_dim, help=f'Number of hidden neurons -- Default= {hidden_dim}')
parser.add_argument('--layer_dim', type=int, default=layer_dim, help=f'Number of hidden layers -- Default= {layer_dim}')
parser.add_argument('--learning_rate', type=float, default=learning_rate, help=f'Learning rate -- Default= {learning_rate}')
parser.add_argument('--training_size', type=float, default=training_size, help=f'Percentage of data to use for training -- Default= {training_size}')
parser.add_argument('--stock', type=str, default=stock, help=f'Stock ticker -- Default= {stock}')
parser.add_argument('--period', type=str, default=period, help=f'Period to fetch data for -- Default= {period}')
parser.add_argument('--patience', type=int, default=patience, help=f'Early stopping patience -- Default= {patience}')
parser.add_argument('--delta', type=float, default=delta, help=f'Early stopping delta -- Default= {delta}')

args = parser.parse_args()
# Update constants with command line arguments
seq_length = args.seq_length
avg_period = args.avg_period
num_epochs = args.num_epochs
hidden_dim = args.hidden_dim
layer_dim = args.layer_dim
learning_rate = args.learning_rate
training_size = args.training_size
patience = args.patience
delta = args.delta
stock = args.stock
period = args.period

# Print the arguments
print(f'Sequence Length: {seq_length}')
print(f'Average Period: {avg_period}')
print(f'Number of Epochs: {num_epochs}')
print(f'Hidden Dimension: {hidden_dim}')
print(f'Layer Dimension: {layer_dim}')
print(f'Learning Rate: {learning_rate}')
print(f'Training Size: {training_size}')
print(f'Patience: {patience}')
print(f'Delta: {delta}')
print(f'Stock: {stock}')
print(f'Period: {period}')

# Check data folder for csv file of stock data
try:
    cont_data_frame = pd.read_csv(f'data/{stock}_{period}_data_cont.csv')
    binary_data_frame = pd.read_csv(f'data/{stock}_{period}_data_binary.csv')
    print(f'Loaded Processed {stock}_{period}_ data from file')
except FileNotFoundError:
    try:
        stock_data = pd.read_csv(f'data/{stock}_{period}_data.csv')
        print(f'Loaded {stock}_{period}_ data from file')
    except FileNotFoundError:
        stock_data = yf.Ticker(stock).history(period=period, interval='1d')
        stock_data.to_csv(f'data/{stock}_{period}_data.csv')
        print(f'Loaded {stock}_{period}_ data from Yahoo Finance') 
    sti = StockTechnicalIndicators(stock_data)
    cont_data_frame, binary_data_frame = sti.get_dataframes(days=30, interval=1)
    cont_data_frame.to_csv(f'data/{stock}_{period}_data_cont.csv', index=False)
    binary_data_frame.to_csv(f'data/{stock}_{period}_data_binary.csv', index=False)
    print(f'Saved Processed {stock}_{period}_ data to file')
    
# Drop unnecessary columns
# cont_data_frame = cont_data_frame.drop(columns=['SMA', 'WMA'])
features = len(cont_data_frame.columns)

# Normalize every avg_period day period to avg_period day average
print(f'Normalizing every {avg_period} day period to {avg_period} day average')
for i in range(0, len(cont_data_frame), avg_period):
    if i + avg_period > len(cont_data_frame):
        # Remove last period if it is not complete
        print(f'Removing last period {i} to {len(cont_data_frame)}')
        cont_data_frame = cont_data_frame.iloc[:i]
        break
    cont_data_frame.iloc[i:i+avg_period, :] = cont_data_frame.iloc[i:i+avg_period, :] - cont_data_frame.iloc[i:i+avg_period, :].mean()
    cont_data_frame.iloc[i:i+avg_period, :] = cont_data_frame.iloc[i:i+avg_period, :] / cont_data_frame.iloc[i:i+avg_period, :].std()
    # cont_data_frame.iloc[i:i+30, :] = cont_data_frame.iloc[i:i+30, :] / cont_data_frame.iloc[i:i+30, :].std()

print(f'Normalizing {stock}_{period}_data_frame')
for i in range(features):
    # Normalize data to be between 0 and 1
    print(f'Normalizing {cont_data_frame.columns[i]} -- Min: {cont_data_frame.iloc[:, i].min()}, Max: {cont_data_frame.iloc[:, i].max()}')
    # if data_frame.iloc[:, i].max() == data_frame.iloc[:, i].min():
    #     data_frame.iloc[:, i] = 0.0
    #     continue
    cont_data_frame.iloc[:, i] = (cont_data_frame.iloc[:, i] - cont_data_frame.iloc[:, i].min()) / (cont_data_frame.iloc[:, i].max() - cont_data_frame.iloc[:, i].min())



# data_frame = binary_data_frame
data_frame = cont_data_frame
    
data_frame.to_csv(f'data/{stock}_{period}_data_frame_normalized.csv', index=False)
print(f'Saved Normalized {stock}_{period}_ data to file')

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

def create_sequences(data, seq_length, prediction_steps=10):
    xs = []
    ys = []
    for i in range(len(data) - seq_length - prediction_steps + 1):
        x = data[i:(i + seq_length)]
        y = data[(i + seq_length):(i + seq_length + prediction_steps)]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def plot_predictions(original, predicted, time_steps, data_frame, title):
    __, axs = plt.subplots(features, 1, figsize=(15, 10 * features))
    for i in range(features):
        data_value_label = data_frame.columns[i]
        row = i
        axs[row].plot(time_steps, original[:, :, i].flatten(), label=f'Original Data {data_value_label}')
        axs[row].plot(time_steps, predicted.detach().numpy()[:, :, i].flatten(), label=f'Predicted Data {data_value_label}', linestyle='--')
        axs[row].set_title(f'LSTM Model Predictions vs. Original Data {data_value_label}')
        axs[row].set_xlabel('Time Step')
        axs[row].set_ylabel(data_value_label)
        axs[row].legend()
    plt.suptitle(title)
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

prediction_steps = 10  # Number of steps to predict ahead

X, y = create_sequences(data, seq_length, prediction_steps)
trainX = torch.tensor(X, dtype=torch.float32)
trainY = torch.tensor(y, dtype=torch.float32)

# Generate synthetic validation data
# t_val = t_full[-100:]  # Use 100 data points for validation
t_val = t_full[train_upper_bound:]
# data_val = np.array([fromiter(t_val, i) for i in range(features)]).T  # Generate 10 input features
data_val = data_full[train_upper_bound:]

X_val, y_val = create_sequences(data_val, seq_length, prediction_steps)
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
        out = self.fc(out[:, -1, :])  # Output for the last time step
        out = out.view(out.size(0), -1, features)  # Reshape to (batch_size, prediction_steps, features)
        return out, hn, cn

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_val_loss = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.trace_func = trace_func
        self.model_buffer = io.BytesIO()

    def __call__(self, val_loss, model, epoch):
        # Check if validation loss is nan
        if np.isnan(val_loss):
            self.trace_func("Validation loss is NaN. Ignoring this epoch.")
            return

        if self.best_val_loss is None:
            self.best_val_loss = val_loss
            self.save_checkpoint(val_loss, model, epoch)
        elif val_loss < self.best_val_loss - self.delta:
            # Significant improvement detected
            self.best_val_loss = val_loss
            self.save_checkpoint(val_loss, model, epoch)
            self.counter = 0  # Reset counter since improvement occurred
        else:
            # No significant improvement
            self.counter += 1
            if epoch % 10 == 0:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model, epoch):
        '''Saves model when validation loss decreases.'''
        if self.verbose and epoch % 10 == 0:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.model_buffer)
        self.model_buffer.seek(0)
        self.val_loss_min = val_loss

    def get_model(self):
        # Use the correct output_dim for multi-step predictions
        output_dim = features * prediction_steps  # Ensure this matches the model used during training
        model = LSTMModel(input_dim=features, hidden_dim=hidden_dim, layer_dim=layer_dim, output_dim=output_dim)
        model.load_state_dict(torch.load(self.model_buffer))
        return model
    
    
model = LSTMModel(
    input_dim=features,
    hidden_dim=hidden_dim,
    layer_dim=layer_dim,
    output_dim=features * prediction_steps
)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

early_stopper = EarlyStopping(patience=patience, verbose=True, delta=delta)

h0, c0 = None, None

start_time = time.time()
last_time = start_time
current_time = time.time()


# Training loop
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    outputs, h0, c0 = model(trainX, h0, c0)

    # Calculate loss for multi-step predictions
    loss = criterion(outputs, trainY)  # trainY now contains 10 steps
    loss.backward()
    optimizer.step()

    h0 = h0.detach()
    c0 = c0.detach()
    
    # Get validation loss
    with torch.no_grad():
        model.eval()
        h0_val, c0_val = None, None  # Reset hidden and cell states for validation
        predicted, _, _ = model(valX, h0_val, c0_val)
        val_loss = criterion(predicted, valY)  # valY also contains 10 steps
        val_loss_float = val_loss.item()

# print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss_float:.4f}')
    early_stopper(val_loss_float, model, epoch)
    if (epoch+1) % 10 == 0:
        current_time = time.time()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss_float:.4f}')
        print(f'Time: {current_time - last_time:.2f} seconds')
        start_time = last_time
    if early_stopper.early_stop:
        print('Early stopping')
        break


current_time = time.time()  
total_time = current_time - start_time
print(f'Training stopped. Total time: {total_time:.2f} seconds')
model = early_stopper.get_model()

# Plot the predictions for training data
model.eval()
predicted, _, _ = model(trainX, h0, c0)

original = y  # Use the target data directly, which has the correct shape
time_steps = np.arange(seq_length, len(data))

predicted = predicted.cpu()

plot_predictions(original, predicted, time_steps, data_frame, 'LSTM Model Predictions vs. Original Data (Training)')

# Plot the predictions for validation data
h0, c0 = None, None  # Reset hidden and cell states for validation
predicted_val, _, _ = model(valX, h0, c0)

original_val = data_val[seq_length:]
time_steps_val = np.arange(seq_length, len(data_val))

predicted_val = predicted_val.cpu()

plot_predictions(original_val, predicted_val, time_steps_val, data_frame, 'LSTM Model Predictions vs. Original Data (Validation)')

plt.show()