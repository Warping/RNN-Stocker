import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io

# === Simulated Data Setup (replace with real) ===
np.random.seed(0)
num_features = 10
num_days = 1000
seq_length = 30
prediction_steps = 10

# Create dummy data with MOM
data = np.random.randn(num_days, num_features)
mom_data = np.random.randn(num_days, 1)
full_data = np.concatenate((data, mom_data), axis=1)

columns = [f'Feat_{i}' for i in range(num_features)] + ['MOM']
data_frame = pd.DataFrame(full_data, columns=columns)

# === Sequence Preparation ===
def create_sequences(data, seq_length, prediction_steps):
    xs, ys = [], []
    for i in range(len(data) - seq_length - prediction_steps + 1):
        x = data[i:i+seq_length]
        y = data[i+seq_length:i+seq_length+prediction_steps]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

mom_index = data_frame.columns.get_loc('MOM')
features = data_frame.shape[1]
data_np = data_frame.to_numpy()
X, y_full = create_sequences(data_np, seq_length, prediction_steps)

# MOM as the only output
y_mom = y_full[:, :, mom_index]
y_mom = np.expand_dims(y_mom, axis=2)

# === Split into Train and Validation ===
split = int(0.7 * len(X))
trainX, valX = torch.tensor(X[:split], dtype=torch.float32), torch.tensor(X[split:], dtype=torch.float32)
trainY, valY = torch.tensor(y_mom[:split], dtype=torch.float32), torch.tensor(y_mom[split:], dtype=torch.float32)

# === MOM-only LSTM Model ===
class LSTMModelMOM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim=1):
        super(LSTMModelMOM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(layer_dim, x.size(0), hidden_dim)
        c0 = torch.zeros(layer_dim, x.size(0), hidden_dim)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out.view(out.size(0), -1, 1)

# === Early Stopping Class ===
class EarlyStopping:
    def __init__(self, patience=20, delta=0.0, verbose=True):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_val_loss = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.verbose = verbose
        self.model_buffer = io.BytesIO()

    def __call__(self, val_loss, model):
        if self.best_val_loss is None:
            self.best_val_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss < self.best_val_loss - self.delta:
            self.best_val_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} → {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.model_buffer)
        self.model_buffer.seek(0)
        self.val_loss_min = val_loss

    def get_model(self, input_dim, hidden_dim, layer_dim):
        model = LSTMModelMOM(input_dim, hidden_dim, layer_dim)
        model.load_state_dict(torch.load(self.model_buffer))
        return model

# === Training Parameters ===
hidden_dim = 64
layer_dim = 2
learning_rate = 0.001
num_epochs = 200
patience = 30

model = LSTMModelMOM(input_dim=features, hidden_dim=hidden_dim, layer_dim=layer_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
early_stopper = EarlyStopping(patience=patience, verbose=True)

# === Training Loop with Early Stopping ===
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(trainX)
    loss = criterion(output, trainY)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_output = model(valX)
        val_loss = criterion(val_output, valY)

    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")
    early_stopper(val_loss.item(), model)

    if early_stopper.early_stop:
        print("Early stopping triggered.")
        break

# === Load Best Model ===
model = early_stopper.get_model(input_dim=features, hidden_dim=hidden_dim, layer_dim=layer_dim)

# === Plot Predictions ===
model.eval()
with torch.no_grad():
    final_pred = model(valX).squeeze().numpy()
    actual_val = valY.squeeze().numpy()

plt.figure(figsize=(12, 5))
plt.plot(actual_val, label='Actual MOM')
plt.plot(final_pred, label='Predicted MOM', linestyle='--')
plt.title("LSTM MOM Prediction with Early Stopping")
plt.xlabel("Time Step")
plt.ylabel("MOM")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
