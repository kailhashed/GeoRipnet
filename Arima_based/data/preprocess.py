"""
Data Preprocessing for RippleNet-TFT
Cleans, aligns timestamps, and merges data from multiple sources
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import yaml
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Main data preprocessing class for RippleNet-TFT"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.raw_data_dir = Path(config['data']['raw_data_path'])
        self.processed_data_dir = Path(config['data']['processed_data_path'])
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Scalers for different data types
        self.scalers = {}
        
    def load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """Load all raw data files"""
        logger.info("Loading raw data files")
        
        data = {}
        
        # Load Yahoo Finance data
        yahoo_files = list(self.raw_data_dir.glob('yahoo_*.csv'))
        for file in yahoo_files:
            symbol = file.stem.replace('yahoo_', '')
            df = pd.read_csv(file)
            df['date'] = pd.to_datetime(df['date'])
            data[f'yahoo_{symbol}'] = df
            logger.info(f"Loaded {symbol} with {len(df)} records")
        
        # Load EIA data
        eia_files = list(self.raw_data_dir.glob('eia_*.csv'))
        for file in eia_files:
            series = file.stem.replace('eia_', '')
            df = pd.read_csv(file)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            elif 'period' in df.columns:
                df['date'] = pd.to_datetime(df['period'])
            data[f'eia_{series}'] = df
            logger.info(f"Loaded EIA {series} with {len(df)} records")
        
        # Load news data
        news_files = list(self.raw_data_dir.glob('news_*.csv'))
        for file in news_files:
            df = pd.read_csv(file)
            df['date'] = pd.to_datetime(df['date'])
            data['news'] = df
            logger.info(f"Loaded news data with {len(df)} records")
        
        # Load GDELT data
        gdelt_files = list(self.raw_data_dir.glob('gdelt_*.csv'))
        for file in gdelt_files:
            df = pd.read_csv(file)
            df['date'] = pd.to_datetime(df['date'])
            data['gdelt'] = df
            logger.info(f"Loaded GDELT data with {len(df)} records")
        
        return data
    
    def clean_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate price data"""
        logger.info("Cleaning price data")
        
        # Remove rows with missing prices
        df = df.dropna(subset=['close'])
        
        # Remove rows with zero or negative prices
        df = df[df['close'] > 0]
        
        # Remove extreme outliers (prices > 10x median)
        median_price = df['close'].median()
        df = df[df['close'] < median_price * 10]
        
        # Forward fill missing values for technical indicators
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(method='ffill')
        
        return df
    
    def align_timestamps(self, data: Dict[str, pd.DataFrame], 
                        start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Align all data to common daily timestamps"""
        logger.info("Aligning timestamps")
        
        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        aligned_data = {}
        
        for name, df in data.items():
            if 'date' not in df.columns:
                logger.warning(f"No date column in {name}, skipping")
                continue
            
            # Set date as index
            df_aligned = df.set_index('date')
            
            # Reindex to common date range
            df_aligned = df_aligned.reindex(date_range)
            
            # Forward fill missing values
            df_aligned = df_aligned.fillna(method='ffill')
            
            # Reset index to get date column back
            df_aligned = df_aligned.reset_index()
            df_aligned.rename(columns={'index': 'date'}, inplace=True)
            
            aligned_data[name] = df_aligned
            logger.info(f"Aligned {name} to {len(df_aligned)} records")
        
        return aligned_data
    
    def create_macro_indicators(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create macroeconomic indicators"""
        logger.info("Creating macroeconomic indicators")
        
        # Get date range from any dataset
        date_range = None
        for df in data.values():
            if 'date' in df.columns:
                date_range = df['date']
                break
        
        if date_range is None:
            logger.error("No date range found")
            return pd.DataFrame()
        
        # Create synthetic macro indicators (in practice, you'd fetch real data)
        macro_df = pd.DataFrame({'date': date_range})
        
        # Economic Policy Uncertainty (EPU) - synthetic
        np.random.seed(42)
        macro_df['epu'] = np.random.normal(100, 20, len(macro_df))
        
        # Geopolitical Risk (GPR) - synthetic
        macro_df['gpr'] = np.random.normal(50, 15, len(macro_df))
        
        # VIX-like volatility index
        macro_df['vix'] = np.random.normal(20, 5, len(macro_df))
        
        # Dollar index
        macro_df['dxy'] = np.random.normal(95, 3, len(macro_df))
        
        # Interest rates
        macro_df['fed_funds_rate'] = np.random.normal(2.5, 0.5, len(macro_df))
        
        logger.info(f"Created macro indicators with {len(macro_df)} records")
        return macro_df
    
    def merge_all_data(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge all data sources into single DataFrame"""
        logger.info("Merging all data sources")
        
        # Start with date range
        date_range = None
        for df in data.values():
            if 'date' in df.columns:
                date_range = df['date']
                break
        
        if date_range is None:
            logger.error("No date range found")
            return pd.DataFrame()
        
        merged_df = pd.DataFrame({'date': date_range})
        
        # Merge price data
        for name, df in data.items():
            if name.startswith('yahoo_'):
                symbol = name.replace('yahoo_', '')
                for col in df.columns:
                    if col != 'date':
                        merged_df[f'{symbol}_{col}'] = df[col]
        
        # Merge EIA data
        for name, df in data.items():
            if name.startswith('eia_'):
                series = name.replace('eia_', '')
                for col in df.columns:
                    if col != 'date' and col in df.columns:
                        merged_df[f'eia_{series}_{col}'] = df[col]
        
        # Merge news data
        if 'news' in data:
            news_df = data['news']
            for col in news_df.columns:
                if col != 'date':
                    merged_df[f'news_{col}'] = news_df[col]
        
        # Merge GDELT data
        if 'gdelt' in data:
            gdelt_df = data['gdelt']
            for col in gdelt_df.columns:
                if col != 'date':
                    merged_df[f'gdelt_{col}'] = gdelt_df[col]
        
        # Add macro indicators
        macro_df = self.create_macro_indicators(data)
        for col in macro_df.columns:
            if col != 'date':
                merged_df[col] = macro_df[col]
        
        # Add calendar features
        merged_df = self._add_calendar_features(merged_df)
        
        # Clean merged data
        merged_df = self._clean_merged_data(merged_df)
        
        logger.info(f"Merged data shape: {merged_df.shape}")
        return merged_df
    
    def _add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calendar-based features"""
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.dayofweek
        df['dayofyear'] = df['date'].dt.dayofyear
        df['quarter'] = df['date'].dt.quarter
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        
        # Holiday indicators (simplified)
        df['is_holiday'] = 0  # In practice, use holiday calendar
        
        return df
    
    def _clean_merged_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean merged dataset"""
        # Remove rows with all NaN values
        df = df.dropna(how='all')
        
        # Forward fill missing values
        df = df.fillna(method='ffill')
        
        # Backward fill any remaining NaN values
        df = df.fillna(method='bfill')
        
        # Remove rows with any remaining NaN values
        df = df.dropna()
        
        return df
    
    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for forecasting"""
        logger.info("Creating target variables")
        
        # Get price columns
        price_cols = [col for col in df.columns if col.endswith('_close')]
        
        for col in price_cols:
            symbol = col.replace('_close', '')
            
            # Next day price
            df[f'{symbol}_next_day'] = df[col].shift(-1)
            
            # Price change
            df[f'{symbol}_price_change'] = df[col].pct_change()
            
            # Next day price change
            df[f'{symbol}_next_day_change'] = df[f'{symbol}_next_day'].pct_change()
            
            # Direction (1 for up, 0 for down)
            df[f'{symbol}_direction'] = (df[f'{symbol}_next_day_change'] > 0).astype(int)
        
        return df
    
    def create_lagged_features(self, df: pd.DataFrame, lags: List[int] = [1, 2, 3, 5, 10, 20]) -> pd.DataFrame:
        """Create lagged features"""
        logger.info("Creating lagged features")
        
        # Get price columns
        price_cols = [col for col in df.columns if col.endswith('_close')]
        
        for col in price_cols:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # Get technical indicator columns
        tech_cols = [col for col in df.columns if any(x in col for x in ['rsi', 'macd', 'volatility', 'sma'])]
        
        for col in tech_cols:
            for lag in lags[:3]:  # Fewer lags for technical indicators
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df
    
    def scale_features(self, df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, Dict]:
        """Scale features using StandardScaler"""
        logger.info("Scaling features")
        
        scalers = {}
        df_scaled = df.copy()
        
        for col in feature_cols:
            if col in df.columns:
                scaler = StandardScaler()
                df_scaled[col] = scaler.fit_transform(df[[col]])
                scalers[col] = scaler
        
        self.scalers = scalers
        return df_scaled, scalers
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = 'merged.csv'):
        """Save processed data"""
        output_path = Path(self.config['data']['merged_data_path'])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")
        
        # Save scalers
        scaler_path = self.processed_data_dir / 'scalers.pkl'
        import pickle
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scalers, f)
        logger.info(f"Saved scalers to {scaler_path}")
    
    def preprocess_all_data(self) -> pd.DataFrame:
        """Main preprocessing pipeline"""
        logger.info("Starting data preprocessing pipeline")
        
        # Load raw data
        data = self.load_raw_data()
        
        if not data:
            logger.error("No raw data found")
            return pd.DataFrame()
        
        # Clean price data
        for name, df in data.items():
            if name.startswith('yahoo_'):
                data[name] = self.clean_price_data(df)
        
        # Align timestamps
        start_date = self.config['data']['start_date']
        end_date = self.config['data']['end_date']
        data = self.align_timestamps(data, start_date, end_date)
        
        # Merge all data
        merged_df = self.merge_all_data(data)
        
        if merged_df.empty:
            logger.error("No data after merging")
            return pd.DataFrame()
        
        # Create target variables
        merged_df = self.create_target_variables(merged_df)
        
        # Create lagged features
        merged_df = self.create_lagged_features(merged_df)
        
        # Get feature columns (exclude date and target columns)
        feature_cols = [col for col in merged_df.columns 
                       if col != 'date' and not col.endswith('_next_day') 
                       and not col.endswith('_direction')]
        
        # Scale features
        merged_df, scalers = self.scale_features(merged_df, feature_cols)
        
        # Save processed data
        self.save_processed_data(merged_df)
        
        logger.info("Data preprocessing completed successfully")
        return merged_df

def main():
    """Main function for data preprocessing"""
    import yaml
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(config)
    
    # Run preprocessing pipeline
    processed_data = preprocessor.preprocess_all_data()
    
    if not processed_data.empty:
        print(f"Preprocessing completed successfully! Data shape: {processed_data.shape}")
    else:
        print("Preprocessing failed - no data produced")

if __name__ == "__main__":
    main()
