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
            return pd.concat(combined_data, ignore_index=True)
        else:
            return pd.DataFrame()

    def save_data(self, filepath: str = None):
        """Save combined data to CSV file."""
        if filepath is None:
            filepath = f"{config.data_save_path}/raw_stock_data.csv"

        combined_data = self.get_combined_data()
        combined_data.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")

    def load_data(self, filepath: str = None) -> pd.DataFrame:
        """Load data from CSV file or fetch fresh data if not found."""
        if filepath is None:
            filepath = f"{config.data_save_path}/raw_stock_data.csv"

        try:
            data = pd.read_csv(filepath)
            print(f"Data loaded from {filepath}")
            return data
        except FileNotFoundError:
            print(f"Warning: File {filepath} not found. Fetching fresh data.")
            return self.get_combined_data()
