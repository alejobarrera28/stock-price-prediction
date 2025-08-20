random_seed = 42

stocks = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'TSLA']
data_period = '5y'

sequence_length = 50
test_size = 0.15
validation_size = 0.15

batch_size = 32
learning_rate = 0.001
num_epochs = 100
patience = 10

hidden_size = 128
num_layers = 2
dropout_rate = 0.2

model_save_path = 'saved_models'
data_save_path = 'data/processed'

device = 'mps'

# Stock-Aware Processing Configuration
# Set to True to enable stock-specific scaling and stock-aware batching
# Set to False to use traditional global scaling and standard batching
use_stock_aware_processing = True