import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA is available. Using GPU.')
else:
    device = torch.device('cpu')
    print('CUDA is not available. Using CPU.')
    
torch.set_default_device(device)

# Create Synthetic Data

def generate_data(seq_length, num_samples, num_features):
    X = []
    Y = []
    for i in range(num_samples):
        x1 = np.random.rand(seq_length + 1) * 2 * np.pi
        x2 = np.random.rand(seq_length + 1) * 2 * np.pi
        y = np.sin(x1) + np.cos(x2)  # Combine the two features
        features = np.zeros((seq_length, num_features))
        features[:, 0] = x1[:-1]
        features[:, 1] = x2[:-1]
        X.append(features)
        Y.append(y[:-1])
    return np.array(X), np.array(Y)

seq_length = 50
num_samples = 100
num_features = 2  # Number of input features
X, Y = generate_data(seq_length, num_samples, num_features)

X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)
print(X.shape, Y.shape)
print(X[0], Y[0])

# Define the RNN

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out)
        return out

input_size = num_features  # Update input_size to match the number of features
hidden_size = 20
output_size = 1
model = SimpleRNN(input_size, hidden_size, output_size)

# Define the LSTM model with multiple layers
# class MultiLayerLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size):
#         super(MultiLayerLSTM, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.linear = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         # Initialize hidden and cell states
#         h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size)
#         c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size)

#         # Pass the input and initial states through the LSTM layer
#         out, _ = self.lstm(x, (h0, c0))

#         # Pass the output of the LSTM layer through the linear layer
#         out = self.linear(out[:, -1, :])
#         return out

# # Example usage
# input_size = num_features
# hidden_size = 100
# num_layers = 2
# output_size = 1

# # Create an instance of the model
# model = MultiLayerLSTM(input_size, hidden_size, num_layers, output_size)

# Train the Model

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 500
for epoch in range(num_epochs):
    model.train()
    outputs = model(X)  # No need to unsqueeze as X already has the correct shape
    loss = criterion(outputs, Y.unsqueeze(2))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
# Visualize the Predictions
        
model.eval()
with torch.no_grad():
    X_cpu = X.cpu()
    Y_cpu = Y.cpu()
    predictions = model(X)
    predictions = predictions.cpu().numpy()
    

plt.figure(figsize=(20, 6))
plt.plot(Y_cpu[0].numpy(), label='True')
plt.plot(predictions[0], label='Predicted')
plt.legend()
plt.show()