import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from early_stopping_pytorch import EarlyStopping

import time

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA is available. Using GPU.')
else:
    device = torch.device('cpu')
    print('CUDA is not available. Using CPU.')
    
torch.set_default_device(device)


# def f(x, i):
#     return np.sin((i + 1) * x)

def f(x, i):
    return np.sin((i + 1) * x) + np.cos((i + 1) * x)

# def f(x, i):
#     return (1 / (i + 1)) * np.sin((i + 1) * x)

# def f(x, i):
#     return np.sin(i*x)

def fromiter(x, i):
    return np.fromiter((f(xi, i) for xi in x), x.dtype)


# Create Synthetic Data

def generate_data(seq_length, num_samples, num_features):
    X = []
    Y = []
    for i in range(num_samples):
        inputs = []
        y = np.zeros(seq_length)
        for j in range(num_features):
            inputs.append(np.linspace(0 + np.pi * i, (2 + i) * np.pi, seq_length))
            y += fromiter(inputs[j], j)
        features = np.zeros((seq_length, num_features))
        for j in range(num_features):
            features[:, j] = inputs[j]
        X.append(features)
        Y.append(y)
    return np.array(X), np.array(Y)

seq_length = 100
num_samples = 500
num_features = 20  # Number of input features
X, Y = generate_data(seq_length, num_samples, num_features)

X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)
print(X.shape, Y.shape)

# Define the LSTM

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)
        return out

print('Creating LSTM model...')
input_size = num_features  # Update input_size to match the number of features
hidden_size = 500
output_size = 1
num_layers = 2  # Define the number of layers

model = SimpleLSTM(input_size, hidden_size, output_size, num_layers=num_layers)
old_model = SimpleLSTM(input_size, hidden_size, output_size, num_layers=num_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00005)

early_stopper = EarlyStopping(patience=500, verbose=True, path='checkpoint.pt', delta=0.0)


print('Training the model...')
start_time = time.time()
current_time = time.time()
num_epochs = 100000

X_test = []
Y_test = []
inputs = []
y = np.zeros(seq_length)
i = num_samples
for j in range(num_features):
    inputs.append(np.linspace(0 + np.pi * i, (2 + i) * np.pi, seq_length))
    y += fromiter(inputs[j], j)
features = np.zeros((seq_length, num_features))
for j in range(num_features):
    features[:, j] = inputs[j]
X_test.append(features)
Y_test.append(y)
Y_test = np.array(Y_test)
X_test = np.array(X_test)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.float32)

for epoch in range(num_epochs):
    model.train()
    outputs = model(X)  # No need to unsqueeze as X already has the correct shape
    loss = criterion(outputs, Y.unsqueeze(2))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Validation loss
    with torch.no_grad():
        # X_test = []
        # Y_test = []
        # inputs = []
        # y = np.zeros(seq_length)
        # i = num_samples
        # for j in range(num_features):
        #     inputs.append(np.linspace(0 + np.pi * i, (2 + i) * np.pi, seq_length))
        #     y += fromiter(inputs[j], j)
        # features = np.zeros((seq_length, num_features))
        # for j in range(num_features):
        #     features[:, j] = inputs[j]
        # X_test.append(features)
        # Y_test.append(y)
        # Y_test = np.array(Y_test)
        # X_test = np.array(X_test)
        # X_test = torch.tensor(X_test, dtype=torch.float32)
        # Y_test = torch.tensor(Y_test, dtype=torch.float32)
        outputs_validation = model(X_test)
        loss_v = criterion(outputs_validation, Y_test.unsqueeze(2))
        loss_validation = loss_v.item()
    # loss_validation = 0.0
        
    loss_training = loss.item()
    # if (epoch + 1) % 2 == 0:
    #     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Validation Loss: {loss_validation:.4f}')
    early_stopper(loss_validation, model)
    if early_stopper.early_stop:
        print(f'Early stopping on epoch {epoch}')
        break
    if (epoch + 1) % 10 == 0:
        print(f'Time taken for last 10 epochs: {time.time() - current_time:.2f} seconds')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss_training:.4f}, Validation Loss: {loss_validation:.4f}')
        current_time = time.time()
        
print(f'Total time taken: {time.time() - start_time:.2f} seconds')
        
# Visualize the Predictions
        
# model.eval()

# Load the best model
torch.save(model.state_dict(), 'old_model.pt')
model.load_state_dict(torch.load('checkpoint.pt'))
model.eval()
old_model.load_state_dict(torch.load('old_model.pt'))
old_model.eval()

with torch.no_grad():
    X_test = []
    Y_test = []
    inputs = []
    y = np.zeros(seq_length)
    i = num_samples
    for j in range(num_features):
        inputs.append(np.linspace(0 + np.pi * i, (2 + i) * np.pi, seq_length))
        y += fromiter(inputs[j], j)
    features = np.zeros((seq_length, num_features))
    for j in range(num_features):
        features[:, j] = inputs[j]
    X_test.append(features)
    Y_test.append(y)
    X_test = np.array(X_test)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    predictions = model(X_test)
    predictions = predictions.cpu().numpy()
    predictions_old = old_model(X_test)
    predictions_old = predictions_old.cpu().numpy()
    
plt.figure(figsize=(10, 6))
plt.plot(Y_test[0], label='True')
plt.plot(predictions[0], label='Predicted')
plt.plot(predictions_old[0], label='Predicted (Old)')
plt.legend()
plt.show()
