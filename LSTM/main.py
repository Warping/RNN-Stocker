import torch
import torch.nn as nn
import numpy as np
import arg_parser
import data_processor as dp
from lstm import DataSetGenerator, MultiLayerLSTMModel
import time
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Set up improved visualization styles
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (15, 12)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 20

# Function to set up professional-looking plots
def setup_plot_style(ax, title, xlabel, ylabel, feature_name):
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add a light gray background to highlight the plot area
    ax.set_facecolor('#f5f5f5')
    
    # Return the axis for further customization
    return ax

arg_vals = arg_parser.ArgParser()
data_proc = dp.DataProcessor()

torch.cuda.empty_cache()
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA is available. Using GPU.')
else:
    device = torch.device('cpu')
    print('CUDA is not available. Using CPU.')

torch.set_default_device(device)
np.random.seed(0)
torch.manual_seed(0)

data_proc.data_load(stock=arg_vals.stock, period=arg_vals.period)
data_proc.drop_columns(arg_vals.drop_columns)
data_proc.data_normalize(avg_period=arg_vals.avg_period, smoothing_window=arg_vals.smoothing_window)

dsg = DataSetGenerator()
dsg.generate_data(data_frame=data_proc.cont_data_frame, 
                  batch_size=arg_vals.batch_size, 
                  training_size=arg_vals.training_size, 
                  validation_size=arg_vals.validation_size, 
                  seq_length=arg_vals.seq_length, 
                  prediction_steps=arg_vals.prediction_steps, 
                  device=device)

model = MultiLayerLSTMModel(input_dim=data_proc.features,
                          hidden_dim=arg_vals.hidden_dim, 
                          layer_dim=arg_vals.layer_dim, 
                          output_dim=data_proc.features, 
                          prediction_steps=arg_vals.prediction_steps, 
                          features=data_proc.features, 
                          patience=arg_vals.patience, 
                          delta=arg_vals.delta)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=arg_vals.learning_rate)
h0, c0 = None, None

start_time = time.time()
last_time = start_time
current_time = time.time()

# Training loop with batches
for epoch in range(arg_vals.num_epochs):
    model.train()
    train_loss = 0.0
    val_loss = 0.0
    test_loss = 0.0

    for batch_X, batch_Y in dsg.train_loader:
        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)  # Move batch to device
        optimizer.zero_grad()

        # Forward pass
        outputs, _, _ = model(batch_X)

        # Reshape batch_Y if necessary
        batch_Y = batch_Y.view(batch_Y.size(0), arg_vals.prediction_steps, -1)

        # Calculate loss
        loss = criterion(outputs, batch_Y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validation loop (similar to training loop)
    val_loss = 0.0
    with torch.no_grad():
        model.eval()
        for batch_X, batch_Y in dsg.val_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            predicted, _, _ = model(batch_X)
            val_loss += criterion(predicted, batch_Y).item()

        for batch_X, batch_Y in dsg.test_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            predicted, _, _ = model(batch_X, None, None)
            test_loss += criterion(predicted, batch_Y).item()
            
    # Average losses
    train_loss /= len(dsg.train_loader)
    val_loss /= len(dsg.val_loader)
    test_loss /= len(dsg.test_loader)

    # Print progress
    print(f"Epoch [{epoch+1}/{arg_vals.num_epochs}]\nTrain Loss: \t{train_loss:.7f}\nVal Loss: \t{val_loss:.7f}\nTest Loss: \t{test_loss:.7f}")
    print(f"Epoch Time: {time.time() - last_time:.2f} seconds")
    print(f"Total time: {time.time() - start_time:.2f} seconds")
    last_time = time.time()
    # Early stopping
    model.early_stopper(val_loss, epoch)
    if model.early_stop:
        print("Early stopping")
        break
    
current_time = time.time()  
total_time = current_time - start_time
print(f'Training stopped. Total time: {total_time:.2f} seconds')
model = model.get_model()

print(f'Saving model to file...')
current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
torch.save(model.state_dict(), f'../output/{arg_vals.stock}_{arg_vals.period}_{current_date}_model.pth')

# Plot the predictions for training data
model.eval()
seq_length = arg_vals.seq_length
prediction_steps = arg_vals.prediction_steps
features = data_proc.features
prediction_smoothing = arg_vals.prediction_smoothing
data_test = dsg.data_test


ground_truth = data_test[-(seq_length+prediction_steps):]
input_2 = ground_truth[:seq_length] # Use the first seq_length data points
input_2_tensor = torch.tensor(input_2, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
predicted, _, _ = model(input_2_tensor, h0, c0)
predicted = predicted.cpu()
predicted = predicted.detach().numpy().reshape(prediction_steps, features)
# Smooth the predicted data by finding a smooth curve through the points
predicted_smooth = pd.DataFrame(predicted, columns=data_proc.cont_data_frame.columns).rolling(window=prediction_smoothing, min_periods=1).mean().to_numpy()


future_data = data_test[-(seq_length):]
future_data_smooth = data_test[-(seq_length):]
future_data_tensor = torch.tensor(future_data, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
future_predicted, _, _ = model(future_data_tensor, h0, c0)
future_predicted = future_predicted.cpu()
future_predicted = future_predicted.detach().numpy().reshape(prediction_steps, features)
# Smooth the predicted data by finding a smooth curve through the points
future_predicted_smooth = pd.DataFrame(future_predicted, columns=data_proc.cont_data_frame.columns).rolling(window=prediction_smoothing, min_periods=1).mean().to_numpy()


predicted_output = np.concatenate((input_2, predicted), axis=0)
predicted_output_smooth = np.concatenate((input_2, predicted_smooth), axis=0)
future_data = np.concatenate((future_data, future_predicted), axis=0)
future_data_smooth = np.concatenate((future_data_smooth, future_predicted_smooth), axis=0)

# Check if the sampled input is the same shape as the ground truth
print(f'Ground truth shape: {ground_truth.shape}')
print(f'Predicted shape: {predicted_output.shape}')
print(f'Predicted Smooth shape: {predicted_output_smooth.shape}')
print(f'Future Data shape: {future_data.shape}')
print(f'Future Data Smooth shape: {future_data_smooth.shape}')


# Enhanced plotting for predicted output
fig, axs = plt.subplots(features, 1, figsize=(15, 8 * features), constrained_layout=True)
if features == 1:
    axs = [axs]  # Make it iterable for the single feature case

for i in range(features):
    feature_name = data_proc.cont_data_frame.columns[i]
    ax = setup_plot_style(axs[i], 
                          f'{feature_name} - Forecast', 
                          'Time Steps', 
                          feature_name,
                          feature_name)
    
    # Plot ground truth
    ax.plot(np.arange(len(ground_truth)), ground_truth[:, i], 
            label='Historical Data', color='#1f77b4', linewidth=2.5)
    
    # Plot predictions
    ax.plot(np.arange(len(predicted_output)), predicted_output[:, i], 
            label='Predicted', color='#ff7f0e', linewidth=2, linestyle='--')
    
    # Plot smoothed predictions
    ax.plot(np.arange(len(predicted_output_smooth)), predicted_output_smooth[:, i], 
            label='Predicted (Smoothed)', color='#2ca02c', linewidth=2.5, linestyle='-')
    
    # Add vertical line and annotation at prediction start
    ax.axvline(x=seq_length, color='red', linestyle='-', alpha=0.7, linewidth=2)
    ax.text(seq_length, ax.get_ylim()[0] + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]), 
            'Prediction Start', fontsize=12, color='red', rotation=90, 
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Add shaded area for prediction region
    ax.axvspan(seq_length, len(predicted_output), alpha=0.1, color='red')
    
    # Enhance legend
    ax.legend(loc='best', frameon=True, framealpha=0.9, fancybox=True, shadow=True, fontsize=12)

# Add a main title with asset name and period
plt.suptitle(f'{arg_vals.stock} ({arg_vals.period}) - Model Predictions', 
             fontsize=22, fontweight='bold', y=1.02)

# Add subtitle with more details
prediction_date = datetime.now().strftime("%B %d, %Y")
plt.figtext(0.5, 0.01, f"Prediction generated on {prediction_date} | Training Time: {total_time:.2f} seconds", 
            ha="center", fontsize=12, bbox={"facecolor":"#f0f0f0", "alpha":0.5, "pad":5})

# Save the enhanced figure
plt.savefig(f"../output/{arg_vals.stock}_{arg_vals.period}_{current_date}_predicted_future.png", 
            dpi=300, bbox_inches='tight')

# Enhanced plotting for future data
fig, axs = plt.subplots(features, 1, figsize=(15, 8 * features), constrained_layout=True)
if features == 1:
    axs = [axs]  # Make it iterable for the single feature case

for i in range(features):
    feature_name = data_proc.cont_data_frame.columns[i]
    ax = setup_plot_style(axs[i], 
                         f'{feature_name} - Future Forecast', 
                         'Time Steps', 
                         feature_name,
                         feature_name)
    
    # Plot input data
    ax.plot(np.arange(seq_length), future_data[:seq_length, i], 
            label='Input Data', color='#1f77b4', linewidth=2.5)
    
    # Plot future predictions
    ax.plot(np.arange(seq_length, len(future_data)), future_data[seq_length:, i], 
            label='Future Forecast', color='#ff7f0e', linewidth=2, linestyle='--')
    
    # Plot smoothed future predictions
    ax.plot(np.arange(seq_length, len(future_data_smooth)), future_data_smooth[seq_length:, i], 
            label='Future Forecast (Smoothed)', color='#2ca02c', linewidth=2.5, linestyle='-')
    
    # Add vertical line and annotation at prediction start
    ax.axvline(x=seq_length, color='red', linestyle='-', alpha=0.7, linewidth=2)
    ax.text(seq_length, ax.get_ylim()[0] + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]), 
            'Prediction Start', fontsize=12, color='red', rotation=90, 
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Add shaded area for prediction region
    ax.axvspan(seq_length, len(future_data), alpha=0.1, color='red')
    
    # Enhance legend
    ax.legend(loc='best', frameon=True, framealpha=0.9, fancybox=True, shadow=True, fontsize=12)

# Add a main title with asset name and period
plt.suptitle(f'{arg_vals.stock} ({arg_vals.period}) - Future Forecast', 
             fontsize=22, fontweight='bold', y=1.02)

# Add subtitle with more details
plt.figtext(0.5, 0.01, f"Forecast generated on {prediction_date} | Looking {prediction_steps} steps ahead", 
            ha="center", fontsize=12, bbox={"facecolor":"#f0f0f0", "alpha":0.5, "pad":5})

# Save the enhanced figure
plt.savefig(f"../output/{arg_vals.stock}_{arg_vals.period}_{current_date}_future_data.png", 
            dpi=300, bbox_inches='tight')

torch.cuda.empty_cache()
plt.show()

