random_seed = 42

stocks = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "TSLA",
    "AMZN",
    "META",
    "NFLX",
    "COST",
    "MELI",
    "SBUX",
    "ABNB",
    "PYPL",
    "KO",
    "ADBE",
    "WMT",
    "V",
    "MCD",
    "NKE",
]
data_period = "10y"

sequence_length = 50
test_size = 0.15
validation_size = 0.15

batch_size = 32
learning_rate = 0.001
num_epochs = 25

hidden_size = 128
num_layers = 2
dropout_rate = 0.2

model_save_path = "saved_models"
data_save_path = "data/processed"

device = "mps"