# Autoregressive-only version

# 1. Add near the top
output_horizon = 10  # (Still used for comparison)

# 2. Update sequence creation for training
def create_sequences(data, seq_length, output_horizon):
    xs, ys = [], []
    for i in range(len(data) - seq_length - output_horizon):
        x = data[i:(i + seq_length)]
        y = data[(i + seq_length):(i + seq_length + output_horizon)]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# 3. Update training/validation set creation
X, y = create_sequences(data, seq_length, output_horizon)
X_val, y_val = create_sequences(data_val, seq_length, output_horizon)

trainX = torch.tensor(X, dtype=torch.float32)
trainY = torch.tensor(y, dtype=torch.float32)
valX = torch.tensor(X_val, dtype=torch.float32)
valY = torch.tensor(y_val, dtype=torch.float32)

# 4. Modify model for single-step output only
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
        out = self.fc(out[:, -1, :])  # Predict just the next step
        return out.unsqueeze(1), hn, cn  # Add a step dimension to keep [batch, 1, features]

# 5. Update model creation
model = LSTMModel(input_dim=features, hidden_dim=hidden_dim, layer_dim=layer_dim, output_dim=features)

# 6. Autoregressive forecast function
def autoregressive_forecast(model, initial_seq, steps, device):
    model.eval()
    preds = []
    current_seq = initial_seq.clone()

    with torch.no_grad():
        for _ in range(steps):
            out, _, _ = model(current_seq)
            next_step = out[:, 0:1, :]  # shape [1, 1, features]
            preds.append(next_step.squeeze(1).cpu().numpy())
            current_seq = torch.cat((current_seq[:, 1:, :], next_step), dim=1)

    return np.array(preds)

# 7. Plot autoregressive predictions
def plot_autoregressive_preds(auto_preds, data_frame, title='Autoregressive Forecast'):
    features = auto_preds.shape[1]
    time_steps = np.arange(auto_preds.shape[0])

    plt.figure(figsize=(15, features * 2))
    for i in range(features):
        plt.subplot(features, 1, i + 1)
        plt.plot(time_steps, auto_preds[:, i], label=f'Predicted {data_frame.columns[i]}')
        plt.title(f'{title} - {data_frame.columns[i]}')
        plt.xlabel('Forecast Step')
        plt.ylabel(data_frame.columns[i])
        plt.grid(True)
        plt.legend()
    plt.tight_layout()
    plt.show()

# 8. Example usage after training
# initial_input = valX[-1:]
# forecast_steps = 50
# auto_preds = autoregressive_forecast(model, initial_input, forecast_steps, device)
# plot_autoregressive_preds(auto_preds, data_frame)