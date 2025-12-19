import pandas as pd
import numpy as np
import os
from typing import Tuple, Optional, List

class FinancialDataLoader:
    """
    A class to handle loading and preprocessing of financial market data 
    for Crisis Regime Prediction using GBDT.
    """
    
    def __init__(self, file_path: str, target_col: str = 'target', date_col: str = 'Date'):
        """
        Initialize the data loader.
        
        Args:
            file_path (str): Path to the CSV file containing market data.
            target_col (str): Name of the target column (e.g., regime label).
            date_col (str): Name of the date column.
        """
        self.file_path = file_path
        self.target_col = target_col
        self.date_col = date_col
        self.data = None

    def load_data(self) -> pd.DataFrame:
        """
        Loads data from the CSV file and parses dates.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The file {self.file_path} was not found.")
            
        print(f"Loading data from {self.file_path}...")
        self.data = pd.read_csv(self.file_path)
        
        # Ensure date column is datetime
        if self.date_col in self.data.columns:
            self.data[self.date_col] = pd.to_datetime(self.data[self.date_col])
            self.data.set_index(self.date_col, inplace=True)
            self.data.sort_index(inplace=True)
        else:
            print(f"Warning: Date column '{self.date_col}' not found. Indexing might be incorrect.")
            
        return self.data

    def clean_data(self, drop_na: bool = True, fill_method: str = 'ffill') -> pd.DataFrame:
        """
        Performs basic cleaning: handling NaNs and infinite values.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Replace infinite values with NaN
        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)

        if drop_na:
            self.data.dropna(inplace=True)
        else:
            if fill_method == 'ffill':
                self.data.fillna(method='ffill', inplace=True)
            elif fill_method == 'bfill':
                self.data.fillna(method='bfill', inplace=True)
            else:
                self.data.fillna(0, inplace=True)
                
        return self.data

    def feature_engineering(self) -> pd.DataFrame:
        """
        Adds standard financial features often used in GBDT models.
        - Log Returns
        - Volatility (Rolling Std Dev)
        - Momentum (Rolling Mean)
        """
        if self.data is None:
            return pd.DataFrame()

        # Example: Calculate Log Returns if 'Close' exists
        if 'Close' in self.data.columns:
            self.data['Log_Ret'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
            
            # Rolling Volatility (e.g., 20 days)
            self.data['Vol_20'] = self.data['Log_Ret'].rolling(window=20).std()
            
            # Simple Moving Average (e.g., 50 days)
            self.data['SMA_50'] = self.data['Close'].rolling(window=50).mean()

        # Drop NaNs created by rolling windows
        self.data.dropna(inplace=True)
        
        return self.data

    def get_features_and_target(self, drop_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separates features (X) and target (y).
        """
        if self.data is None:
            raise ValueError("Data not loaded or processed.")

        if self.target_col not in self.data.columns:
             raise ValueError(f"Target column '{self.target_col}' not found in dataset.")

        y = self.data[self.target_col]
        X = self.data.drop(columns=[self.target_col])

        if drop_cols:
            X.drop(columns=drop_cols, errors='ignore', inplace=True)

        return X, y

if __name__ == "__main__":
    # Example Usage
    loader = FinancialDataLoader(file_path='data/market_data.csv', target_col='Crisis_Label')
    try:
        df = loader.load_data()
        df = loader.clean_data()
        df = loader.feature_engineering()
        X, y = loader.get_features_and_target()
        print("Data successfully loaded and processed.")
        print(f"Features shape: {X.shape}, Target shape: {y.shape}")
    except Exception as e:
        print(f"Error during execution: {e}")