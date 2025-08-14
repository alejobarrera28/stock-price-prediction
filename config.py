random_seed = 42

stocks = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'TSLA']
data_period = '5y'

sequence_length = 50
test_size = 0.15
validation_size = 0.15

data_save_path = 'data/processed'

# Stock-Aware Processing Configuration
# Set to True to enable stock-specific scaling and stock-aware batching
# Set to False to use traditional global scaling and standard batching
use_stock_aware_processing = True