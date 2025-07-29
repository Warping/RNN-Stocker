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
import os

# Set up improved visualization styles
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (15, 12)
plt.rcParams['font.size'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 20  # Increased from 14 to 20
plt.rcParams['ytick.labelsize'] = 20  # Increased from 14 to 20
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['figure.titlesize'] = 20
plt.rcParams['savefig.transparent'] = False  # Ensure backgrounds are not transparent

# Function to set up professional-looking plots
def setup_plot_style(ax, title, xlabel, ylabel, feature_name):
    # ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.grid(True, linestyle='--', alpha=1.0)  # Solid grid lines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add a solid light gray background to highlight the plot area
    ax.set_facecolor('#f5f5f5')
    
    # Increase tick label font sizes
    ax.tick_params(axis='both', which='major', labelsize=20)  # Increased from 14 to 20
    ax.tick_params(axis='both', which='minor', labelsize=16)  # Increased from 12 to 16
    
    # Return the axis for further customization
    return ax

# Function to save individual feature plots as EPS files
def save_feature_as_eps(ax, feature_name, stock, period, current_date, plot_type):
    # Create output directory if it doesn't exist
    output_dir = os.path.join('..', 'output', 'eps_plots')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as EPS file
    fig = plt.figure(figsize=(10, 6))
    new_ax = fig.add_subplot(111)
    
    # Copy the content from the original axis
    for line in ax.lines:
        new_ax.plot(line.get_xdata(), line.get_ydata(), 
                   color=line.get_color(), 
                   linestyle=line.get_linestyle(),
                   linewidth=line.get_linewidth(),
                   label=line.get_label())
    
    # Copy styling
    setup_plot_style(new_ax, 
                    f'{feature_name} - {plot_type}', 
                    'Time Steps', 
                    feature_name,
                    feature_name)
    
    # Match x-axis limits from original plot
    if hasattr(ax, 'get_xlim'):
        new_ax.set_xlim(ax.get_xlim())
    
    # Enhance legend with solid background
    new_ax.legend(loc='best', frameon=True, framealpha=1.0, fancybox=True, shadow=True, fontsize=14,
                 edgecolor='black')
    
    # Set figure with solid background
    fig.patch.set_alpha(1.0)
    
    # Save the figure
    filename = f"{stock}_{period}_{current_date}_{feature_name}_{plot_type}.eps"
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, format='eps', dpi=300, bbox_inches='tight', transparent=False)
    plt.close(fig)
    
    return filepath

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

def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics for model evaluation"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Flatten the arrays to 1D
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Calculate regression metrics
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    r2 = r2_score(y_true_flat, y_pred_flat)
    
    return mse, rmse, mae, r2

# def setup_plot_style(ax, title, xlabel, ylabel, legend_label):
#     """Setup the plot style for each subplot"""
#     # ax.set_title(title, fontsize=16, fontweight='bold')
#     ax.set_xlabel(xlabel, fontsize=24)
#     ax.set_ylabel(ylabel, fontsize=24)
#     ax.grid(True, linestyle='--', alpha=0.7)
#     ax.tick_params(axis='both', which='major', labelsize=20)
#     ax.legend(loc='best', fontsize=20)
#     return ax

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

y_true = ground_truth[seq_length:]
y_pred = predicted_output[seq_length:]
mse, rmse, mae, r2 = calculate_metrics(y_true, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R² Score: {r2:.4f}")


# Enhanced plotting for predicted output
fig, axs = plt.subplots(features, 1, figsize=(15, 8 * features), constrained_layout=True)
fig.patch.set_alpha(1.0)  # Solid background for the figure

if features == 1:
    axs = [axs]  # Make it iterable for the single feature case

for i in range(features):
    feature_name = data_proc.cont_data_frame.columns[i]
    ax = setup_plot_style(axs[i], 
                          f'{feature_name} - Forecast', 
                          'Time Steps', 
                          feature_name,
                          feature_name)
    
    # Plot only the ground truth for the prediction period
    ax.plot(np.arange(seq_length, len(ground_truth)), ground_truth[seq_length:, i], 
            label='Ground Truth', color='#1f77b4', linewidth=2.5)
    
    # Plot only the predicted values (not the input sequence)
    ax.plot(np.arange(seq_length, len(predicted_output)), predicted_output[seq_length:, i], 
            label='Predicted', color='#ff7f0e', linewidth=2, linestyle='--')
    
    # Plot only the smoothed predictions
    ax.plot(np.arange(seq_length, len(predicted_output_smooth)), predicted_output_smooth[seq_length:, i], 
            label='Predicted (Smoothed)', color='#2ca02c', linewidth=2.5, linestyle='-')
    
    # Adjust x-axis to show only the prediction region
    ax.set_xlim(seq_length-1, len(predicted_output))
    
    # Enhance legend with solid background
    ax.legend(loc='best', frameon=True, framealpha=1.0, fancybox=True, shadow=True, fontsize=14,
             edgecolor='black')
    
    # Save individual feature as EPS
    save_feature_as_eps(ax, feature_name, arg_vals.stock, arg_vals.period, 
                       current_date, 'Forecast')

# Add a main title with asset name and period
plt.suptitle(f'{arg_vals.stock} ({arg_vals.period}) - Model Predictions', 
             fontsize=22, fontweight='bold', y=1.02)

# Add subtitle with more details - using solid background
prediction_date = datetime.now().strftime("%B %d, %Y")
plt.figtext(0.5, 0.01, f"Prediction generated on {prediction_date} | Training Time: {total_time:.2f} seconds", 
            ha="center", fontsize=14, bbox={"facecolor":"#f0f0f0", "alpha":1.0, "pad":5, "edgecolor":"black"})  # Increased from 12 to 14

# Save the enhanced figure
plt.savefig(f"../output/{arg_vals.stock}_{arg_vals.period}_{current_date}_predicted_future.png", 
            dpi=300, bbox_inches='tight', transparent=False)

# Enhanced plotting for future data
fig, axs = plt.subplots(features, 1, figsize=(15, 8 * features), constrained_layout=True)
fig.patch.set_alpha(1.0)  # Solid background for the figure

if features == 1:
    axs = [axs]  # Make it iterable for the single feature case

for i in range(features):
    feature_name = data_proc.cont_data_frame.columns[i]
    ax = setup_plot_style(axs[i], 
                         f'{feature_name} - Future Forecast', 
                         'Time Steps', 
                         feature_name,
                         feature_name)
    
    # Plot only the future predictions
    ax.plot(np.arange(seq_length, len(future_data)), future_data[seq_length:, i], 
            label='Future Forecast', color='#ff7f0e', linewidth=2, linestyle='--')
    
    # Plot only the smoothed future predictions
    ax.plot(np.arange(seq_length, len(future_data_smooth)), future_data_smooth[seq_length:, i], 
            label='Future Forecast (Smoothed)', color='#2ca02c', linewidth=2.5, linestyle='-')
    
    # Adjust x-axis to show only the prediction region
    ax.set_xlim(seq_length-1, len(future_data))
    
    # Enhance legend with solid background
    ax.legend(loc='best', frameon=True, framealpha=1.0, fancybox=True, shadow=True, fontsize=14,
             edgecolor='black')
    
    # Save individual feature as EPS
    save_feature_as_eps(ax, feature_name, arg_vals.stock, arg_vals.period, 
                       current_date, 'Future_Forecast')

# Add a main title with asset name and period
plt.suptitle(f'{arg_vals.stock} ({arg_vals.period}) - Future Forecast', 
             fontsize=22, fontweight='bold', y=1.02)

# Add subtitle with more details - using solid background
plt.figtext(0.5, 0.01, f"Forecast generated on {prediction_date} | Looking {prediction_steps} steps ahead", 
            ha="center", fontsize=14, bbox={"facecolor":"#f0f0f0", "alpha":1.0, "pad":5, "edgecolor":"black"})  # Increased from 12 to 14

# Save the enhanced figure
plt.savefig(f"../output/{arg_vals.stock}_{arg_vals.period}_{current_date}_future_data.png", 
            dpi=300, bbox_inches='tight', transparent=False)

print(f"EPS files saved to: {os.path.join('..', 'output', 'eps_plots')}")

torch.cuda.empty_cache()
plt.show()

