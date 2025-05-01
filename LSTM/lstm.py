import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import io

class MultiLayerLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, prediction_steps, features, patience, delta):
        super(MultiLayerLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.prediction_steps = prediction_steps
        self.trace_func = print  # Default trace function to print messages
        self.model_buffer = io.BytesIO()
        self.patience = patience
        self.delta = delta  # Default delta for early stopping
        self.features = features
        self.verbose = True  # Default verbose flag
        self.counter = 0  # Counter for early stopping
        self.best_val_loss = None  # Initialize best validation loss
        self.early_stop = False  # Flag for early stopping
        self.val_loss_min = np.inf  # Initialize minimum validation loss

        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim * prediction_steps)

    def forward(self, x, h0=None, c0=None):
        # Dynamically initialize hidden states if not provided
        if h0 is None or c0 is None:
            batch_size = x.size(0)  # Get the batch size from the input
            h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(x.device)

        # Forward propagate through LSTM
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # Fully connected layer
        out = self.fc(out[:, -1, :])  # Output for the last time step
        out = out.view(out.size(0), self.prediction_steps, -1)  # Reshape to match target shape
        return out, hn, cn
    
    def early_stopper(self, val_loss, epoch):
        # Check if validation loss is nan
        if np.isnan(val_loss):
            self.trace_func("Validation loss is NaN. Ignoring this epoch.")
            return

        if self.best_val_loss is None:
            self.best_val_loss = val_loss
            self.save_checkpoint(val_loss, epoch)
        elif val_loss < self.best_val_loss - self.delta:
            # Significant improvement detected
            self.best_val_loss = val_loss
            self.save_checkpoint(val_loss, epoch)
            self.counter = 0  # Reset counter since improvement occurred
        else:
            # No significant improvement
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                
    def save_checkpoint(self, val_loss, epoch):
        self.trace_func(f'Epoch: {epoch+1} - Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(self.state_dict(), self.model_buffer)
        self.model_buffer.seek(0)
        self.val_loss_min = val_loss
        
    def get_model(self):
        model = MultiLayerLSTMModel(self.input_dim, self.hidden_dim, self.layer_dim, self.output_dim, self.prediction_steps, self.features, self.patience, self.delta)
        model.load_state_dict(torch.load(self.model_buffer))
        return model
    
class DataSetGenerator:
    def __init__():
        pass

    def generate_data(self, data_frame, batch_size, training_size, validation_size, seq_length, prediction_steps, device):
        # Generate train data
        t_full = np.linspace(0, len(data_frame), len(data_frame), endpoint=False, dtype=int)
        train_upper_bound = int(training_size*len(t_full))
        val_upper_bound = int((training_size + validation_size)*len(t_full))
        # t_train = t_full[:train_upper_bound]
        data_full = data_frame.to_numpy()
        data = data_full[:train_upper_bound]

        X, y = self.create_sequences(data, seq_length, prediction_steps)
        trainX = torch.tensor(X, dtype=torch.float32)
        trainY = torch.tensor(y, dtype=torch.float32)

        # Generate validation data
        # t_val = t_full[train_upper_bound:val_upper_bound]
        data_val = data_full[train_upper_bound:val_upper_bound]

        X_val, y_val = self.create_sequences(data_val, seq_length, prediction_steps)
        valX = torch.tensor(X_val, dtype=torch.float32)
        valY = torch.tensor(y_val, dtype=torch.float32)

        # Generate test data
        # t_test = t_full[val_upper_bound:]  # Use 100 data points for testing
        data_test = data_full[val_upper_bound:]
        X_test, y_test = self.create_sequences(data_test, seq_length, prediction_steps)
        testX = torch.tensor(X_test, dtype=torch.float32)
        testY = torch.tensor(y_test, dtype=torch.float32)

        print(f'Training data shape: {trainX.shape}, {trainY.shape}')
        print(f'Validation data shape: {valX.shape}, {valY.shape}')
        print(f'Test data shape: {testX.shape}, {testY.shape}')
        # Create DataLoader for training, validation, and test datasets
        trainX, trainY = trainX.to(device), trainY.to(device)
        valX, valY = valX.to(device), valY.to(device)
        testX, testY = testX.to(device), testY.to(device)
        trainGen = torch.Generator(device=device)
        valGen = torch.Generator(device=device)
        testGen = torch.Generator(device=device)
        train_dataset = TensorDataset(trainX, trainY)
        val_dataset = TensorDataset(valX, valY)
        test_dataset = TensorDataset(testX, testY)

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=trainGen)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, generator=valGen)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, generator=testGen)
        self.data_train = data
        self.data_val = data_val
        self.data_test = data_test

        
    def create_sequences(self, data, seq_length, prediction_steps):
        xs = []
        ys = []
        for i in range(len(data) - seq_length - prediction_steps + 1):
            x = data[i:(i + seq_length)]
            y = data[(i + seq_length):(i + seq_length + prediction_steps)]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)