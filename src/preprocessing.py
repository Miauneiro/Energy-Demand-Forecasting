"""
Preprocessing pipeline for electricity demand forecasting.

Handles:
- Temporal feature engineering
- Normalization (MinMax scaling)
- Sequence windowing for LSTM
- Train/validation/test splitting
- PyTorch DataLoader creation
"""

# CRITICAL FIX: Preload torch DLLs before importing torch
import sys
import os

# Fix OpenMP duplicate library issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Windows Python 3.8+ DLL loading fix
if sys.platform == 'win32' and sys.version_info >= (3, 8):
    import importlib.util
    import ctypes
    
    # Find torch installation
    torch_spec = importlib.util.find_spec('torch')
    if torch_spec and torch_spec.origin:
        torch_lib_path = os.path.join(os.path.dirname(torch_spec.origin), 'lib')
        
        if os.path.exists(torch_lib_path):
            # Add to DLL search path
            os.add_dll_directory(torch_lib_path)
            
            # NUCLEAR OPTION: Preload critical DLLs manually
            critical_dlls = [
                'asmjit.dll',
                'libiomp5md.dll', 
                'fbgemm.dll'  # The problematic one
            ]
            
            for dll_name in critical_dlls:
                dll_path = os.path.join(torch_lib_path, dll_name)
                if os.path.exists(dll_path):
                    try:
                        # Force load the DLL
                        ctypes.CDLL(dll_path)
                        print(f"✅ Preloaded: {dll_name}")
                    except Exception as e:
                        print(f"⚠️  Failed to preload {dll_name}: {e}")

# Now import everything else
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import joblib
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class TemporalFeatureEngineer:
    """Extract temporal features from datetime column."""
    
    @staticmethod
    def add_temporal_features(df: pd.DataFrame, datetime_col: str = 'timestamp') -> pd.DataFrame:
        """
        Add temporal features to dataframe.
        
        Args:
            df: DataFrame with datetime column
            datetime_col: Name of datetime column
            
        Returns:
            DataFrame with added features
        """
        df = df.copy()
        
        # Ensure datetime
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        
        # Extract temporal features
        df['hour'] = df[datetime_col].dt.hour
        df['day_of_week'] = df[datetime_col].dt.dayofweek  # Monday=0, Sunday=6
        df['day_of_month'] = df[datetime_col].dt.day
        df['month'] = df[datetime_col].dt.month
        df['day_of_year'] = df[datetime_col].dt.dayofyear
        
        # Binary features
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Cyclical encoding (sine/cosine) for better continuity
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    @staticmethod
    def add_lag_features(df: pd.DataFrame, target_col: str = 'demand_mw', 
                        lags: list = [24, 168]) -> pd.DataFrame:
        """
        Add lagged features (24h = daily, 168h = weekly).
        
        Args:
            df: DataFrame with target column
            target_col: Column to create lags from
            lags: List of lag periods (in hours)
            
        Returns:
            DataFrame with lag features
        """
        df = df.copy()
        
        for lag in lags:
            df[f'lag_{lag}h'] = df[target_col].shift(lag)
        
        # Drop rows with NaN from lagging
        df = df.dropna().reset_index(drop=True)
        
        return df


class DemandDataset(Dataset):
    """PyTorch Dataset for electricity demand sequences."""
    
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            sequences: Input sequences (samples, lookback, features)
            targets: Target sequences (samples, horizon, 1)
        """
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx]


class EnergyDataPreprocessor:
    """
    Complete preprocessing pipeline for energy demand forecasting.
    
    Pipeline:
    1. Load data
    2. Add temporal features
    3. Normalize (MinMax scaling)
    4. Create sequences (windowing)
    5. Split train/val/test
    6. Create DataLoaders
    """
    
    def __init__(
        self,
        lookback: int = 168,      # 1 week of history
        horizon: int = 24,        # 1 day ahead forecast
        batch_size: int = 32,
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15,
        add_lags: bool = False    # Lag features optional (adds complexity)
    ):
        """
        Initialize preprocessor.
        
        Args:
            lookback: Hours of history to use as input
            horizon: Hours ahead to forecast
            batch_size: Batch size for DataLoaders
            train_split: Fraction for training (oldest data)
            val_split: Fraction for validation
            test_split: Fraction for testing (newest data)
            add_lags: Whether to add lag features (24h, 168h)
        """
        self.lookback = lookback
        self.horizon = horizon
        self.batch_size = batch_size
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.add_lags = add_lags
        
        # Verify splits sum to 1
        assert abs(train_split + val_split + test_split - 1.0) < 1e-6, \
            "Splits must sum to 1.0"
        
        # Scalers
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Feature engineer
        self.feature_engineer = TemporalFeatureEngineer()
        
        # Store feature names
        self.feature_cols = None
        
    def load_and_prepare_data(
        self, 
        filepath: str,
        datetime_col: str = 'timestamp',
        target_col: str = 'demand_mw'
    ) -> pd.DataFrame:
        """
        Load and prepare data with feature engineering.
        
        Args:
            filepath: Path to CSV file
            datetime_col: Name of datetime column
            target_col: Name of target column
            
        Returns:
            DataFrame with all features
        """
        print("=" * 70)
        print("  PREPROCESSING PIPELINE")
        print("=" * 70)
        print(f"\n📁 Loading data from: {filepath}")
        
        # Load
        df = pd.read_csv(filepath)
        print(f"   Original shape: {df.shape}")
        
        # Add temporal features
        print("\n🔧 Engineering temporal features...")
        df = self.feature_engineer.add_temporal_features(df, datetime_col)
        
        # Optionally add lag features
        if self.add_lags:
            print("   Adding lag features (24h, 168h)...")
            df = self.feature_engineer.add_lag_features(df, target_col)
            print(f"   Shape after lags: {df.shape}")
        
        # Define feature columns (everything except timestamp and region)
        feature_cols = [col for col in df.columns 
                       if col not in [datetime_col, 'region']]
        self.feature_cols = feature_cols
        
        print(f"\n📊 Features ({len(feature_cols)}):")
        print(f"   {feature_cols}")
        print(f"\n✅ Prepared shape: {df.shape}")
        
        return df
    
    def normalize_data(
        self,
        data: np.ndarray,
        fit: bool = True
    ) -> np.ndarray:
        """
        Normalize data using MinMax scaling.
        
        Args:
            data: Data to normalize
            fit: If True, fit scaler. If False, only transform.
            
        Returns:
            Normalized data
        """
        if fit:
            normalized = self.scaler.fit_transform(data)
        else:
            normalized = self.scaler.transform(data)
        
        return normalized
    
    def create_sequences(
        self,
        data: np.ndarray,
        target_idx: int = 0  # Index of target column in data
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM (sliding window approach).
        
        Args:
            data: Normalized data array (timesteps, features)
            target_idx: Index of target column
            
        Returns:
            X: Input sequences (samples, lookback, features)
            y: Target sequences (samples, horizon, 1)
        """
        X, y = [], []
        
        # Slide window through data
        for i in range(len(data) - self.lookback - self.horizon + 1):
            # Input: lookback hours of all features
            X.append(data[i:i + self.lookback])
            
            # Target: next horizon hours of demand only
            y.append(data[i + self.lookback:i + self.lookback + self.horizon, target_idx:target_idx+1])
        
        return np.array(X), np.array(y)
    
    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data chronologically into train/val/test.
        
        IMPORTANT: Time series must maintain chronological order!
        
        Args:
            X: Input sequences
            y: Target sequences
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        n_samples = len(X)
        
        # Calculate split indices
        train_end = int(n_samples * self.train_split)
        val_end = int(n_samples * (self.train_split + self.val_split))
        
        # Split chronologically (oldest → newest)
        X_train = X[:train_end]
        y_train = y[:train_end]
        
        X_val = X[train_end:val_end]
        y_val = y[train_end:val_end]
        
        X_test = X[val_end:]
        y_test = y[val_end:]
        
        print(f"\n📊 Data Split (chronological):")
        print(f"   Train: {len(X_train):,} samples ({self.train_split*100:.0f}%)")
        print(f"   Val:   {len(X_val):,} samples ({self.val_split*100:.0f}%)")
        print(f"   Test:  {len(X_test):,} samples ({self.test_split*100:.0f}%)")
        print(f"\n   Sequence shape: {X_train.shape}")
        print(f"   Target shape: {y_train.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_dataloaders(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
        shuffle_train: bool = True
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders.
        
        Args:
            X_train, X_val, X_test: Input sequences
            y_train, y_val, y_test: Target sequences
            shuffle_train: Whether to shuffle training data
            
        Returns:
            train_loader, val_loader, test_loader
        """
        # Create datasets
        train_dataset = DemandDataset(X_train, y_train)
        val_dataset = DemandDataset(X_val, y_val)
        test_dataset = DemandDataset(X_test, y_test)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle_train,
            drop_last=True  # Drop incomplete batch
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        print(f"\n🔄 DataLoaders created:")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        print(f"   Test batches: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader
    
    def preprocess_pipeline(
        self,
        filepath: str,
        save_dir: str = 'data/processed',
        datetime_col: str = 'timestamp',
        target_col: str = 'demand_mw'
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Complete preprocessing pipeline.
        
        Args:
            filepath: Path to raw CSV
            save_dir: Directory to save scalers
            datetime_col: Name of datetime column
            target_col: Name of target column
            
        Returns:
            train_loader, val_loader, test_loader
        """
        # Create save directory
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # 1. Load and prepare
        df = self.load_and_prepare_data(filepath, datetime_col, target_col)
        
        # 2. Extract features as numpy array
        feature_data = df[self.feature_cols].values
        
        # Find target column index
        target_idx = self.feature_cols.index(target_col)
        
        # 3. Normalize (fit on full data for now, will refit on train only)
        print("\n🔢 Normalizing data...")
        # We need to fit scaler only on training portion
        # First create sequences, then split, then fit scaler properly
        
        # Split data chronologically BEFORE normalization
        n_samples = len(feature_data)
        train_end = int(n_samples * self.train_split)
        val_end = int(n_samples * (self.train_split + self.val_split))
        
        train_data = feature_data[:train_end]
        val_data = feature_data[train_end:val_end]
        test_data = feature_data[val_end:]
        
        # Fit scaler ONLY on training data
        train_normalized = self.scaler.fit_transform(train_data)
        val_normalized = self.scaler.transform(val_data)
        test_normalized = self.scaler.transform(test_data)
        
        print(f"   Fitted scaler on training data only")
        print(f"   Target (demand) range after scaling: [{train_normalized[:, target_idx].min():.3f}, {train_normalized[:, target_idx].max():.3f}]")
        
        # 4. Create sequences for each split
        print("\n🔄 Creating sequences (windowing)...")
        print(f"   Lookback: {self.lookback} hours")
        print(f"   Horizon: {self.horizon} hours")
        
        X_train, y_train = self.create_sequences(train_normalized, target_idx)
        X_val, y_val = self.create_sequences(val_normalized, target_idx)
        X_test, y_test = self.create_sequences(test_normalized, target_idx)
        
        # Note: y contains normalized values, we'll need scaler for inverse transform later
        
        # 5. Create DataLoaders
        train_loader, val_loader, test_loader = self.create_dataloaders(
            X_train, X_val, X_test,
            y_train, y_val, y_test
        )
        
        # 6. Save scalers for deployment
        scaler_path = Path(save_dir) / 'scaler.pkl'
        joblib.dump(self.scaler, scaler_path)
        print(f"\n💾 Saved scaler to: {scaler_path}")
        
        # Save preprocessing config
        config = {
            'lookback': self.lookback,
            'horizon': self.horizon,
            'feature_cols': self.feature_cols,
            'target_col': target_col,
            'target_idx': target_idx,
            'n_features': len(self.feature_cols),
            'batch_size': self.batch_size,
            'add_lags': self.add_lags
        }
        
        import json
        config_path = Path(save_dir) / 'preprocessing_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"💾 Saved config to: {config_path}")
        
        print("\n" + "=" * 70)
        print("  ✅ PREPROCESSING COMPLETE")
        print("=" * 70)
        print("\n🎯 Next steps:")
        print("   1. Build LSTM model (src/model.py)")
        print("   2. Create training loop (src/train.py)")
        print("   3. Train and evaluate!")
        print()
        
        return train_loader, val_loader, test_loader
    
    def inverse_transform_predictions(
        self,
        predictions: np.ndarray,
        target_col_idx: int = 0
    ) -> np.ndarray:
        """
        Convert normalized predictions back to original scale.
        
        Args:
            predictions: Normalized predictions (samples, horizon, 1)
            target_col_idx: Index of target in feature columns
            
        Returns:
            Predictions in original scale (MW)
        """
        # Get min/max for target column from scaler
        target_min = self.scaler.data_min_[target_col_idx]
        target_max = self.scaler.data_max_[target_col_idx]
        
        # Inverse transform: value * (max - min) + min
        original_scale = predictions * (target_max - target_min) + target_min
        
        return original_scale


def quick_preprocess(
    filepath: str = 'data/raw/us_demand_historical.csv',
    lookback: int = 168,
    horizon: int = 24,
    batch_size: int = 256  # 🔥 MEGA BEAST - 73% VRAM utilization!
) -> Tuple[DataLoader, DataLoader, DataLoader, EnergyDataPreprocessor]:
    """
    Quick preprocessing with default settings.
    
    Args:
        filepath: Path to data file
        lookback: Hours of history
        horizon: Hours to forecast
        batch_size: Batch size
        
    Returns:
        train_loader, val_loader, test_loader, preprocessor
    """
    preprocessor = EnergyDataPreprocessor(
        lookback=lookback,
        horizon=horizon,
        batch_size=batch_size,
        add_lags=False  # Start simple, add later if needed
    )
    
    loaders = preprocessor.preprocess_pipeline(filepath)
    
    return (*loaders, preprocessor)


if __name__ == "__main__":
    # Example usage
    print("Energy Demand Preprocessing Module")
    print("=" * 70)
    print("\nUsage example:")
    print("""
    from preprocessing import quick_preprocess
    
    # Run preprocessing
    train_loader, val_loader, test_loader, preprocessor = quick_preprocess(
        filepath='data/raw/us_demand_historical.csv',
        lookback=168,  # 1 week
        horizon=24,    # 1 day ahead
        batch_size=32
    )
    
    # Check a batch
    for X_batch, y_batch in train_loader:
        print(f"Input shape: {X_batch.shape}")   # (32, 168, n_features)
        print(f"Target shape: {y_batch.shape}")  # (32, 24, 1)
        break
    """)
