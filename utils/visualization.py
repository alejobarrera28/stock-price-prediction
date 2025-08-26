import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class StockVisualization:
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        self.colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    # Exploratory Data Analysis (EDA) methods

    def plot_stock_prices(self, data: pd.DataFrame, symbols=None) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        for i, symbol in enumerate(symbols[:4]):
            symbol_data = data[data["symbol"] == symbol].copy()
            symbol_data.index = pd.to_datetime(symbol_data.index)

            axes[i].plot(
                symbol_data.index,
                symbol_data["close"],
                color=self.colors[i],
                linewidth=1.5,
            )
            axes[i].set_title(f"{symbol} Stock Price", fontsize=14, fontweight="bold")
            axes[i].set_xlabel("Date")
            axes[i].set_ylabel("Price ($)")
            axes[i].grid(True, alpha=0.3)
            axes[i].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.show()

    # Training and Prediction Visualization methods
    def plot_training_history(self, history, model_name: str, save_path: str = None):
        epochs = range(1, len(history["train_loss"]) + 1)

        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.plot(epochs, history["train_loss"], "b-", label="Training Loss")
        if "val_loss" in history:
            ax1.plot(epochs, history["val_loss"], "r-", label="Validation Loss")
        ax1.set_title(f"{model_name} - Training History")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        if "train_loss" in history and "val_loss" in history:
            train_loss = np.array(history["train_loss"])
            val_loss = np.array(history["val_loss"])
            ax2.plot(epochs, val_loss - train_loss, "g-")
            ax2.set_title("Validation - Training Loss")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Loss Difference")
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color="black", linestyle="--", alpha=0.5)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def plot_predictions_vs_actual(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        symbol: str = None,
        save_path: str = None,
    ):
        _, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))

        dates = pd.date_range(start="2023-01-01", periods=len(y_true), freq="D")

        ax1.plot(dates, y_true, label="Actual", color="blue", alpha=0.7, linewidth=2)
        ax1.plot(dates, y_pred, label="Predicted", color="red", alpha=0.7, linewidth=2)
        ax1.set_title(
            f"{model_name} - Actual vs Predicted Prices"
            + (f" ({symbol})" if symbol else ""),
            fontsize=14,
            fontweight="bold",
        )
        ax1.set_ylabel("Price ($)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.scatter(y_true, y_pred, alpha=0.6, color="green")
        min_val, max_val = min(y_true.min(), y_pred.min()), max(
            y_true.max(), y_pred.max()
        )
        ax2.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.8)
        ax2.set_xlabel("Actual Price ($)")
        ax2.set_ylabel("Predicted Price ($)")
        ax2.set_title("Actual vs Predicted Scatter Plot")
        ax2.grid(True, alpha=0.3)

        residuals = y_true - y_pred
        ax3.plot(dates, residuals, color="purple", alpha=0.7)
        ax3.axhline(y=0, color="black", linestyle="--", alpha=0.8)
        ax3.set_xlabel("Date")
        ax3.set_ylabel("Residuals ($)")
        ax3.set_title("Prediction Residuals")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
