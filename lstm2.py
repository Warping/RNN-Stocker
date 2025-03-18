import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from early_stopping_pytorch import EarlyStopping

# Constants
seq_length = 10 # Number of time steps to look back
features = 2 # Number of input and output features

# Hyperparameters
num_epochs = 1000 # Number of epochs
hidden_dim = 100 # Number of hidden neurons
layer_dim = 1 # Number of hidden layers
learning_rate = 0.01 # Learning rate

# Early stopping
patience = 100
delta = 0.001

def data_grabber(time_step_index, feauture_index):
    # return np.sin(time_step_index*(feauture_index + 1)) # This is a dummy function. Replace this with your data grabber function
    return feauture_index

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA is available. Using GPU.')
else:
    device = torch.device('cpu')
    print('CUDA is not available. Using CPU.')

def fromiter(x, i):
    return np.fromiter((data_grabber(xi, i) for xi in x), x.dtype)

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

# Generate synthetic data
t = np.linspace(0, 100, 1000)
data = np.array([fromiter(t, i) for i in range(features)]).T  # Generate 10 input features
X, y = create_sequences(data, seq_length)

trainX = torch.tensor(X, dtype=torch.float32)
trainY = torch.tensor(y, dtype=torch.float32)

# Generate synthetic validation data
t_val = np.linspace(100, 200, 1000)
data_val = np.array([fromiter(t_val, i) for i in range(features)]).T  # Generate 10 input features
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

early_stopper = EarlyStopping(patience=patience, verbose=True, path='checkpoint.pt', delta=delta)

h0, c0 = None, None

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
        predicted, _, _ = model(valX, h0, c0)
        val_loss = criterion(predicted, valY)
        val_loss_float = val_loss.item()

    early_stopper(val_loss_float, model)
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss_float:.4f}')
    if early_stopper.early_stop:
        print('Early stopping')
        break
        
# Plot the predictions for training data
model.eval()
predicted, _, _ = model(trainX, h0, c0)

original = data[seq_length:]
time_steps = np.arange(seq_length, len(data))

predicted = predicted.cpu()

plt.figure(figsize=(12, 6))
for i in range(features):
    plt.plot(time_steps, original[:, i], label=f'Original Data {i+1}')
    plt.plot(time_steps, predicted.detach().numpy()[:, i], label=f'Predicted Data {i+1}', linestyle='--')
plt.title('LSTM Model Predictions vs. Original Data (Training)')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()

# Plot the predictions for validation data
predicted_val, _, _ = model(valX, h0, c0)

original_val = data_val[seq_length:]
time_steps_val = np.arange(seq_length, len(data_val))

predicted_val = predicted_val.cpu()

plt.figure(figsize=(12, 6))
for i in range(features):
    plt.plot(time_steps_val, original_val[:, i], label=f'Original Data {i+1}')
    plt.plot(time_steps_val, predicted_val.detach().numpy()[:, i], label=f'Predicted Data {i+1}', linestyle='--')
plt.title('LSTM Model Predictions vs. Original Data (Validation)')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()

plt.show()