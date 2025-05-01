import argparse

class ArgParser:
    def __init__(self, 
                 seq_length=200, 
                 avg_period=30, 
                 num_epochs=100000, 
                 batch_size=4096, 
                 hidden_dim=500, 
                 layer_dim=2, 
                 learning_rate=0.0005,
                 training_size=0.80,
                 validation_size=0.10,
                 test_size=0.10,
                 prediction_steps=20,
                 prediction_smoothing=20,
                 patience=200,
                 delta=0.0,
                 verbose=False,
                 stock='IBM',
                 period='10y',
                 smoothing_window=5,
                 drop_columns=[]):

        # Parse command line arguments
        parser = argparse.ArgumentParser(description='LSTM Stock Price Prediction')
        parser.add_argument('--seq_length', type=int, default=seq_length, help=f'Number of time steps to look back -- Default= {seq_length}')
        parser.add_argument('--avg_period', type=int, default=avg_period, help=f'Number of days to normalize and average over -- Default= {avg_period}')
        parser.add_argument('--num_epochs', type=int, default=num_epochs, help=f'Number of epochs -- Default= {num_epochs}')
        parser.add_argument('--batch_size', type=int, default=batch_size, help=f'Batch size -- Default= {batch_size}')
        parser.add_argument('--hidden_dim', type=int, default=hidden_dim, help=f'Number of hidden neurons -- Default= {hidden_dim}')
        parser.add_argument('--layer_dim', type=int, default=layer_dim, help=f'Number of hidden layers -- Default= {layer_dim}')
        parser.add_argument('--learning_rate', type=float, default=learning_rate, help=f'Learning rate -- Default= {learning_rate}')
        parser.add_argument('--training_size', type=float, default=training_size, help=f'Percentage of data to use for training -- Default= {training_size}')
        parser.add_argument('--validation_size', type=float, default=validation_size, help=f'Percentage of data to use for validation -- Default= {validation_size}')
        parser.add_argument('--test_size', type=float, default=test_size, help=f'Percentage of data to use for testing -- Default= {test_size}')
        parser.add_argument('--stock', type=str, default=stock, help=f'Stock ticker -- Default= {stock}')
        parser.add_argument('--period', type=str, default=period, help=f'Period to fetch data for -- Default= {period}')
        parser.add_argument('--patience', type=int, default=patience, help=f'Early stopping patience -- Default= {patience}')
        parser.add_argument('--delta', type=float, default=delta, help=f'Early stopping delta -- Default= {delta}')
        parser.add_argument('--prediction_steps', type=int, default=prediction_steps, help=f'Number of steps to predict ahead -- Default= {prediction_steps}')
        parser.add_argument('--drop_columns', type=str, default=drop_columns, help=f'Columns to drop from the data frame -- Default= {drop_columns}')
        parser.add_argument('--prediction_smoothing', type=int, default=prediction_smoothing, help=f'Number of steps to smooth the prediction -- Default= {prediction_smoothing}')
        parser.add_argument('--verbose', type=bool, default=verbose, help=f'Print verbose output -- Default= {verbose}')
        parser.add_argument('--smoothing_window', type=int, default=smoothing_window, help=f'Smoothing window for the data -- Default= {smoothing_window}')

        args = parser.parse_args()
        # Update constants with command line arguments
        self.seq_length = args.seq_length
        self.avg_period = args.avg_period
        self.num_epochs = args.num_epochs
        self.hidden_dim = args.hidden_dim
        self.layer_dim = args.layer_dim
        self.learning_rate = args.learning_rate
        self.training_size = args.training_size
        self.validation_size = args.validation_size
        self.test_size = args.test_size
        self.patience = args.patience
        self.delta = args.delta
        self.stock = args.stock
        self.period = args.period
        self.prediction_steps = args.prediction_steps
        self.prediction_smoothing = args.prediction_smoothing
        self.drop_columns = args.drop_columns
        self.verbose = args.verbose
        self.smoothing_window = args.smoothing_window
        self.batch_size = args.batch_size
        
    def print_args(self):
        print(f'Sequence Length: {self.seq_length}')
        print(f'Average Period: {self.avg_period}')
        print(f'Number of Epochs: {self.num_epochs}')
        print(f'Batch Size: {self.batch_size}')
        print(f'Hidden Dimension: {self.hidden_dim}')
        print(f'Layer Dimension: {self.layer_dim}')
        print(f'Learning Rate: {self.learning_rate}')
        print(f'Training Size: {self.training_size}')
        print(f'Validation Size: {self.validation_size}')
        print(f'Test Size: {self.test_size}')
        print(f'Stock: {self.stock}')
        print(f'Period: {self.period}')
        print(f'Patience: {self.patience}')
        print(f'Delta: {self.delta}')
        print(f'Prediction Steps: {self.prediction_steps}')
        print(f'Drop Columns: {self.drop_columns}')
        print(f'Prediction Smoothing: {self.prediction_smoothing}')
        print(f'Verbose: {self.verbose}')
        print(f'Smoothing Window: {self.smoothing_window}')
        