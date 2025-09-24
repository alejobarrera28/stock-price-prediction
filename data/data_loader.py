import yfinance as yf
import pandas as pd
import config


class StockDataLoader:
    """Loads and manages stock price data using Yahoo Finance."""

    def __init__(self, symbols=None, period: str = "5y"):
        """Initialize with stock symbols and time period."""
        self.symbols = symbols or config.stocks
        self.period = period
        self.data = {}

    def fetch_stock_data(self, symbol: str) -> pd.DataFrame:
        """Fetch historical data for a single stock symbol."""
        try:
            print(f"Fetching data for {symbol}")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=self.period)

            if data.empty:
                print(f"Warning: No data found for {symbol}")
                return None

            data.columns = data.columns.str.lower()
            data["symbol"] = symbol

            return data

        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None

    def fetch_all_data(self):
        """Fetch data for all configured symbols."""
        for symbol in self.symbols:
            data = self.fetch_stock_data(symbol)
            if data is not None:
                self.data[symbol] = data

        print(f"Successfully loaded data for {len(self.data)} symbols")
        return self.data

    def get_combined_data(self) -> pd.DataFrame:
        """Combine all stock data into a single DataFrame."""
        if not self.data:
            self.fetch_all_data()

        combined_data = []
        for symbol, data in self.data.items():
            data_copy = data.copy()
            data_copy["symbol"] = symbol
            combined_data.append(data_copy)

        if combined_data:
            return pd.concat(combined_data, ignore_index=False)
        else:
            return pd.DataFrame()

    def save_data(self, filepath: str = None):
        """Save combined data to CSV file."""
        if filepath is None:
            filepath = f"{config.data_save_path}/raw_stock_data.csv"

        combined_data = self.get_combined_data()
        combined_data.to_csv(filepath, index=True)
        print(f"Data saved to {filepath}")

    def load_data(self, filepath: str = None) -> pd.DataFrame:
        """Load data from CSV file or fetch fresh data if not found."""
        if filepath is None:
            filepath = f"{config.data_save_path}/raw_stock_data.csv"

        try:
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            print(f"Data loaded from {filepath}")
            return data
        except FileNotFoundError:
            print(f"File {filepath} not found. Fetching fresh data.")
            return self.get_combined_data()


class StockAwareDataLoader:
    """Custom DataLoader that ensures batches contain sequences from the same stock."""

    def __init__(self, dataset, batch_size: int = 32, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Filter out stocks with no data
        self.stocks = [stock for stock in dataset.stock_groups.keys() 
                      if len(dataset.stock_groups[stock]) > 0]
        
        if not self.stocks:
            print("Warning: No stocks with data found in dataset")
            self.stocks = []
            
        self.current_stock_idx = 0
        self.epoch_finished = False

        # Calculate total number of batches per stock
        self.stock_batch_counts = {}
        for stock in self.stocks:
            num_sequences = len(dataset.stock_groups[stock])
            self.stock_batch_counts[stock] = (
                num_sequences + batch_size - 1
            ) // batch_size

        self.reset_epoch()

    def reset_epoch(self):
        """Reset counters for a new epoch"""
        self.current_stock_idx = 0
        self.epoch_finished = False
        self.stock_batches_used = {stock: 0 for stock in self.stocks}

    def __iter__(self):
        self.reset_epoch()
        return self

    def __next__(self):
        if self.epoch_finished or not self.stocks:
            raise StopIteration

        # Try to get a batch from current stock
        attempts = 0
        while attempts < len(self.stocks):
            current_stock = self.stocks[self.current_stock_idx]

            # Check if this stock has more batches available
            if (
                self.stock_batches_used[current_stock]
                < self.stock_batch_counts[current_stock]
            ):
                try:
                    batch_x, batch_y = self.dataset.get_stock_batch(
                        current_stock, self.batch_size
                    )
                    self.stock_batches_used[current_stock] += 1

                    # Move to next stock for next batch
                    self.current_stock_idx = (self.current_stock_idx + 1) % len(
                        self.stocks
                    )

                    return batch_x, batch_y
                except ValueError as e:
                    # This stock is exhausted or has issues, move to next
                    print(f"Warning: Issue with stock {current_stock}: {e}")
                    pass

            # Move to next stock
            self.current_stock_idx = (self.current_stock_idx + 1) % len(self.stocks)
            attempts += 1

        # All stocks exhausted
        self.epoch_finished = True
        raise StopIteration

    def __len__(self):
        return sum(self.stock_batch_counts.values())
