import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from early_stopping_pytorch import EarlyStopping
import ruptures as rpt
import time

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA is available. Using GPU.')
else:
    device = torch.device('cpu')
    print('CUDA is not available. Using CPU.')
    
torch.set_default_device(device)

def noise(x):
    return np.random.normal(0, 0.0, x.shape)

def f(x, i):
    return (1 / (i + 1)) * np.sin(0.001*x)

def fromiter(x, i):
    return np.fromiter((f(xi, i) for xi in x), x.dtype)

def generate_data(seq_length, num_samples, num_features):
    X = []
    Y = []
    for i in range(num_samples):
        inputs = []
        y = np.zeros(seq_length)
        for j in range(num_features):
            inputs.append(np.linspace(2 * np.pi * i, 2 * np.pi * (i + 1), seq_length))
            y += fromiter(inputs[j], j)
        y += noise(inputs[0])
        features = np.zeros((seq_length, num_features))
        for j in range(num_features):
            features[:, j] = inputs[j]
        X.append(features)
        Y.append(y)
    return np.array(X), np.array(Y)

seq_length = 25
num_samples = 200
num_features = 1
X, Y = generate_data(seq_length, num_samples, num_features)
X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)
print(X.shape, Y.shape)

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

input_size = num_features
hidden_size = 500
output_size = 1
num_layers = 2
model = SimpleLSTM(input_size, hidden_size, output_size, num_layers=num_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00005)
early_stopper = EarlyStopping(patience=300, verbose=True, path='checkpoint.pt', delta=0.01)

X_test, Y_test = generate_data(seq_length, 1, num_features)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.float32)

print('Training the model...')
start_time = time.time()
num_epochs = 100000
for epoch in range(num_epochs):
    model.train()
    outputs = model(X)
    loss = criterion(outputs, Y.unsqueeze(2))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        outputs_validation = model(X_test)
        loss_v = criterion(outputs_validation, Y_test.unsqueeze(2))
        loss_validation = loss_v.item()

    early_stopper(loss_validation, model)
    if early_stopper.early_stop:
        print(f'Early stopping on epoch {epoch}')
        break
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Validation Loss: {loss_validation:.4f}')

print(f'Total time taken: {time.time() - start_time:.2f} seconds')

model.load_state_dict(torch.load('checkpoint.pt'))
model.eval()

with torch.no_grad():
    predictions = model(X)

# Flatten for CPA
Y_flat = Y.cpu().numpy().flatten()
pred_flat = predictions.cpu().numpy().flatten()
residuals = Y_flat - pred_flat

algo = rpt.Pelt(model="rbf").fit(residuals)
change_points = algo.predict(pen=5)

plt.figure(figsize=(12, 6))
plt.plot(residuals, label="Residuals (True - Predicted)")
for cp in change_points[:-1]:
    plt.axvline(cp, color='red', linestyle='--', label="Change Point" if cp == change_points[0] else "")
plt.title("Change Point Detection on Residuals")
plt.xlabel("Time Step")
plt.ylabel("Residual")
plt.legend()
plt.grid(True)
plt.show()

print("Change points detected at:", change_points)
