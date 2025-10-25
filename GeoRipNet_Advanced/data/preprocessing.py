"""
Data Preprocessing and Feature Engineering for GeoRipNet.

Handles scaling, normalization, feature creation, and data transformation
for oil price prediction with ripple effects.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from typing import Dict, List, Tuple, Optional
import pickle
from pathlib import Path


class PerCountryScaler:
    """
    Per-country scaling to handle heterogeneous price distributions.
    
    Each country gets its own scaler for both features and targets.
    """
    
    def __init__(self, num_countries: int, scaler_type: str = 'standard'):
        """
        Args:
            num_countries: Number of countries
            scaler_type: 'standard', 'robust', or 'minmax'
        """
        self.num_countries = num_countries
        self.scaler_type = scaler_type
        
        # Create scalers for each country
        self.country_scalers = []
        for _ in range(num_countries):
            if scaler_type == 'standard':
                self.country_scalers.append(StandardScaler())
            elif scaler_type == 'robust':
                self.country_scalers.append(RobustScaler())
            elif scaler_type == 'minmax':
                self.country_scalers.append(MinMaxScaler())
            else:
                raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    def fit(self, data: np.ndarray):
        """
        Fit scalers on training data.
        
        Args:
            data: (T, num_countries) or (T, num_countries, features)
        """
        if data.ndim == 2:
            # Prices: (T, num_countries)
            for c in range(self.num_countries):
                self.country_scalers[c].fit(data[:, c].reshape(-1, 1))
        else:
            # Features: (T, num_countries, features)
            for c in range(self.num_countries):
                self.country_scalers[c].fit(data[:, c, :])
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted scalers."""
        if data.ndim == 2:
            transformed = np.zeros_like(data)
            for c in range(self.num_countries):
                transformed[:, c] = self.country_scalers[c].transform(
                    data[:, c].reshape(-1, 1)
                ).flatten()
        else:
            transformed = np.zeros_like(data)
            for c in range(self.num_countries):
                transformed[:, c, :] = self.country_scalers[c].transform(data[:, c, :])
        
        return transformed
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform to original scale."""
        if data.ndim == 2:
            original = np.zeros_like(data)
            for c in range(self.num_countries):
                original[:, c] = self.country_scalers[c].inverse_transform(
                    data[:, c].reshape(-1, 1)
                ).flatten()
        else:
            original = np.zeros_like(data)
            for c in range(self.num_countries):
                original[:, c, :] = self.country_scalers[c].inverse_transform(data[:, c, :])
        
        return original
    
    def save(self, path: str):
        """Save scalers to disk."""
        with open(path, 'wb') as f:
            pickle.dump(self.country_scalers, f)
    
    def load(self, path: str):
        """Load scalers from disk."""
        with open(path, 'rb') as f:
            self.country_scalers = pickle.load(f)


class FeatureEngineer:
    """
    Feature engineering for oil price prediction.
    
    Creates lag features, rolling statistics, technical indicators, etc.
    """
    
    @staticmethod
    def create_lag_features(
        data: np.ndarray,
        lags: List[int] = [1, 2, 3, 7, 14, 30]
    ) -> np.ndarray:
        """
        Create lagged features.
        
        Args:
            data: (T, num_features) or (T,)
            lags: List of lag periods
        
        Returns:
            lagged_features: (T, num_features * len(lags))
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        T, num_features = data.shape
        lagged = []
        
        for lag in lags:
            lagged_data = np.zeros_like(data)
            lagged_data[lag:] = data[:-lag]
            lagged.append(lagged_data)
        
        return np.concatenate(lagged, axis=-1)
    
    @staticmethod
    def create_rolling_features(
        data: np.ndarray,
        windows: List[int] = [7, 14, 30]
    ) -> Dict[str, np.ndarray]:
        """
        Create rolling window statistics.
        
        Args:
            data: (T,) or (T, 1)
            windows: List of window sizes
        
        Returns:
            Dictionary of rolling features (mean, std, min, max)
        """
        if data.ndim == 2:
            data = data.flatten()
        
        df = pd.Series(data)
        features = {}
        
        for window in windows:
            features[f'rolling_mean_{window}'] = df.rolling(window).mean().fillna(0).values
            features[f'rolling_std_{window}'] = df.rolling(window).std().fillna(0).values
            features[f'rolling_min_{window}'] = df.rolling(window).min().fillna(0).values
            features[f'rolling_max_{window}'] = df.rolling(window).max().fillna(0).values
        
        return features
    
    @staticmethod
    def create_momentum_features(
        prices: np.ndarray,
        periods: List[int] = [7, 14, 30]
    ) -> Dict[str, np.ndarray]:
        """
        Create momentum and rate-of-change features.
        
        Args:
            prices: (T,) - Price series
            periods: List of periods for momentum calculation
        
        Returns:
            Dictionary of momentum features
        """
        if prices.ndim == 2:
            prices = prices.flatten()
        
        df = pd.Series(prices)
        features = {}
        
        for period in periods:
            # Rate of change
            features[f'roc_{period}'] = df.pct_change(period).fillna(0).values
            
            # Momentum (difference)
            features[f'momentum_{period}'] = df.diff(period).fillna(0).values
        
        return features
    
    @staticmethod
    def create_volatility_features(
        prices: np.ndarray,
        windows: List[int] = [7, 14, 30]
    ) -> Dict[str, np.ndarray]:
        """
        Create volatility features.
        
        Args:
            prices: (T,) - Price series
            windows: List of window sizes
        
        Returns:
            Dictionary of volatility features
        """
        if prices.ndim == 2:
            prices = prices.flatten()
        
        df = pd.Series(prices)
        returns = df.pct_change().fillna(0)
        
        features = {}
        
        for window in windows:
            # Rolling standard deviation of returns
            features[f'volatility_{window}'] = returns.rolling(window).std().fillna(0).values
            
            # Historical volatility (annualized)
            features[f'hist_vol_{window}'] = (
                returns.rolling(window).std() * np.sqrt(252)
            ).fillna(0).values
        
        return features
    
    @staticmethod
    def create_technical_indicators(prices: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Create technical indicators (RSI, MACD, Bollinger Bands, etc.).
        
        Args:
            prices: (T,) - Price series
        
        Returns:
            Dictionary of technical indicators
        """
        if prices.ndim == 2:
            prices = prices.flatten()
        
        df = pd.Series(prices)
        features = {}
        
        # RSI (Relative Strength Index)
        delta = df.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        features['rsi'] = (100 - (100 / (1 + rs))).fillna(50).values
        
        # MACD
        ema_12 = df.ewm(span=12).mean()
        ema_26 = df.ewm(span=26).mean()
        features['macd'] = (ema_12 - ema_26).fillna(0).values
        features['macd_signal'] = (ema_12 - ema_26).ewm(span=9).mean().fillna(0).values
        
        # Bollinger Bands
        rolling_mean = df.rolling(window=20).mean()
        rolling_std = df.rolling(window=20).std()
        features['bb_upper'] = (rolling_mean + 2 * rolling_std).fillna(df.mean()).values
        features['bb_lower'] = (rolling_mean - 2 * rolling_std).fillna(df.mean()).values
        features['bb_width'] = ((rolling_std * 4) / rolling_mean).fillna(1).values
        
        return features
    
    @staticmethod
    def create_time_features(timestamps: pd.DatetimeIndex) -> Dict[str, np.ndarray]:
        """
        Create time-based features.
        
        Args:
            timestamps: DatetimeIndex
        
        Returns:
            Dictionary of time features
        """
        features = {}
        
        # Cyclical encoding of day, month
        features['day_of_week_sin'] = np.sin(2 * np.pi * timestamps.dayofweek / 7)
        features['day_of_week_cos'] = np.cos(2 * np.pi * timestamps.dayofweek / 7)
        features['day_of_month_sin'] = np.sin(2 * np.pi * timestamps.day / 31)
        features['day_of_month_cos'] = np.cos(2 * np.pi * timestamps.day / 31)
        features['month_sin'] = np.sin(2 * np.pi * timestamps.month / 12)
        features['month_cos'] = np.cos(2 * np.pi * timestamps.month / 12)
        
        # Year (normalized)
        features['year'] = (timestamps.year - timestamps.year.min()) / (timestamps.year.max() - timestamps.year.min() + 1)
        
        # Business day indicator
        features['is_business_day'] = timestamps.dayofweek.isin([0, 1, 2, 3, 4]).astype(float)
        
        return features


class DataPreprocessor:
    """
    Complete data preprocessing pipeline for GeoRipNet.
    
    Handles all preprocessing steps including scaling, feature engineering,
    and data validation.
    """
    
    def __init__(
        self,
        num_countries: int,
        num_benchmarks: int,
        scaler_type: str = 'standard',
        create_lag_features: bool = True,
        create_technical_indicators: bool = True
    ):
        self.num_countries = num_countries
        self.num_benchmarks = num_benchmarks
        self.scaler_type = scaler_type
        self.create_lag_features_flag = create_lag_features
        self.create_technical_indicators_flag = create_technical_indicators
        
        # Initialize scalers
        self.price_scaler = PerCountryScaler(num_countries, scaler_type)
        self.local_feature_scaler = PerCountryScaler(num_countries, scaler_type)
        self.benchmark_scaler = StandardScaler()
        
        self.is_fitted = False
    
    def fit(self, train_data: Dict[str, np.ndarray]):
        """
        Fit all scalers on training data.
        
        Args:
            train_data: Dictionary with 'prices', 'local_features', 'benchmark_features'
        """
        # Fit price scaler
        self.price_scaler.fit(train_data['prices'])
        
        # Fit local feature scaler
        self.local_feature_scaler.fit(train_data['local_features'])
        
        # Fit benchmark scaler
        self.benchmark_scaler.fit(train_data['benchmark_features'])
        
        self.is_fitted = True
    
    def transform(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Transform data using fitted scalers.
        
        Args:
            data: Dictionary with raw data
        
        Returns:
            Transformed data dictionary
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")
        
        transformed = data.copy()
        
        # Scale prices
        transformed['prices'] = self.price_scaler.transform(data['prices'])
        
        # Scale local features
        transformed['local_features'] = self.local_feature_scaler.transform(data['local_features'])
        
        # Scale benchmark features
        transformed['benchmark_features'] = self.benchmark_scaler.transform(data['benchmark_features'])
        
        return transformed
    
    def fit_transform(self, train_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Fit and transform in one step."""
        self.fit(train_data)
        return self.transform(train_data)
    
    def inverse_transform_prices(self, scaled_prices: np.ndarray) -> np.ndarray:
        """
        Inverse transform predictions back to original scale.
        
        Args:
            scaled_prices: (T, num_countries) - Scaled predictions
        
        Returns:
            Original scale prices
        """
        return self.price_scaler.inverse_transform(scaled_prices)
    
    def save(self, save_dir: str):
        """Save preprocessor state."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.price_scaler.save(save_path / 'price_scaler.pkl')
        self.local_feature_scaler.save(save_path / 'local_feature_scaler.pkl')
        
        with open(save_path / 'benchmark_scaler.pkl', 'wb') as f:
            pickle.dump(self.benchmark_scaler, f)
        
        # Save metadata
        metadata = {
            'num_countries': self.num_countries,
            'num_benchmarks': self.num_benchmarks,
            'scaler_type': self.scaler_type,
            'is_fitted': self.is_fitted
        }
        with open(save_path / 'preprocessor_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
    
    def load(self, save_dir: str):
        """Load preprocessor state."""
        save_path = Path(save_dir)
        
        # Load metadata
        with open(save_path / 'preprocessor_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        self.num_countries = metadata['num_countries']
        self.num_benchmarks = metadata['num_benchmarks']
        self.scaler_type = metadata['scaler_type']
        self.is_fitted = metadata['is_fitted']
        
        # Load scalers
        self.price_scaler = PerCountryScaler(self.num_countries, self.scaler_type)
        self.price_scaler.load(save_path / 'price_scaler.pkl')
        
        self.local_feature_scaler = PerCountryScaler(self.num_countries, self.scaler_type)
        self.local_feature_scaler.load(save_path / 'local_feature_scaler.pkl')
        
        with open(save_path / 'benchmark_scaler.pkl', 'rb') as f:
            self.benchmark_scaler = pickle.load(f)


if __name__ == "__main__":
    # Test preprocessing
    print("Testing Data Preprocessing...")
    
    T = 500
    num_countries = 10
    num_benchmarks = 3
    
    # Create dummy data
    prices = np.random.randn(T, num_countries) * 10 + 80
    local_features = np.random.randn(T, num_countries, 20)
    benchmark_features = np.random.randn(T, 30)
    
    # Test PerCountryScaler
    print("\n1. Testing PerCountryScaler...")
    scaler = PerCountryScaler(num_countries, 'standard')
    scaled_prices = scaler.fit_transform(prices)
    recovered_prices = scaler.inverse_transform(scaled_prices)
    
    print(f"   Original prices mean: {prices.mean():.2f}")
    print(f"   Scaled prices mean: {scaled_prices.mean():.4f}")
    print(f"   Recovered prices mean: {recovered_prices.mean():.2f}")
    print(f"   Reconstruction error: {np.abs(prices - recovered_prices).mean():.6f}")
    
    # Test feature engineering
    print("\n2. Testing FeatureEngineer...")
    fe = FeatureEngineer()
    
    test_series = prices[:, 0]
    
    lag_features = fe.create_lag_features(test_series, lags=[1, 2, 3])
    print(f"   Lag features shape: {lag_features.shape}")
    
    rolling_features = fe.create_rolling_features(test_series, windows=[7, 14])
    print(f"   Rolling features: {len(rolling_features)} types")
    
    momentum_features = fe.create_momentum_features(test_series, periods=[7, 14])
    print(f"   Momentum features: {len(momentum_features)} types")
    
    volatility_features = fe.create_volatility_features(test_series, windows=[7, 14])
    print(f"   Volatility features: {len(volatility_features)} types")
    
    technical_features = fe.create_technical_indicators(test_series)
    print(f"   Technical indicators: {len(technical_features)} types")
    print(f"   RSI range: [{technical_features['rsi'].min():.2f}, {technical_features['rsi'].max():.2f}]")
    
    # Test time features
    timestamps = pd.date_range('2020-01-01', periods=T, freq='D')
    time_features = fe.create_time_features(timestamps)
    print(f"   Time features: {len(time_features)} types")
    
    # Test complete preprocessor
    print("\n3. Testing DataPreprocessor...")
    data = {
        'prices': prices,
        'local_features': local_features,
        'benchmark_features': benchmark_features
    }
    
    preprocessor = DataPreprocessor(num_countries, num_benchmarks)
    transformed_data = preprocessor.fit_transform(data)
    
    print(f"   Transformed prices shape: {transformed_data['prices'].shape}")
    print(f"   Transformed prices mean: {transformed_data['prices'].mean():.4f}")
    print(f"   Transformed prices std: {transformed_data['prices'].std():.4f}")
    
    # Test save/load
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        preprocessor.save(tmpdir)
        
        new_preprocessor = DataPreprocessor(num_countries, num_benchmarks)
        new_preprocessor.load(tmpdir)
        
        print(f"   Preprocessor saved and loaded successfully")
        print(f"   Is fitted: {new_preprocessor.is_fitted}")
    
    print("\n✓ Preprocessing tests passed!")

