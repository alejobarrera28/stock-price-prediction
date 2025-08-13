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

    def plot_correlation_matrix(self, data: pd.DataFrame, figsize=None):
        fig, ax = plt.subplots(figsize=figsize or (12, 10))

        plt.grid(False)

        numeric_columns = data.select_dtypes(include=[np.number]).columns
        correlation_matrix = data[numeric_columns].corr()

        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

        sns.heatmap(
            correlation_matrix,
            mask=mask,
            annot=True,
            cmap="coolwarm",
            center=0,
            square=True,
            fmt=".2f",
            cbar_kws={"shrink": 0.8},
            ax=ax,
        )

        ax.set_title("Feature Correlation Matrix")
        fig.tight_layout()

        plt.show()
