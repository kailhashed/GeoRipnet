"""
Simplified preprocessing for RippleNet-TFT
Works with available Yahoo Finance data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_yahoo_data():
    """Load Yahoo Finance data"""
    logger.info("Loading Yahoo Finance data")
    
    data_dir = Path("data/raw")
    symbols = ["CL=F", "NG=F", "MTF=F"]
    
    all_data = {}
    
    for symbol in symbols:
        file_path = data_dir / f"yahoo_{symbol}.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'], utc=True)
            df = df.set_index('date')
            
            # Clean column names
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]
            
            all_data[symbol] = df
            logger.info(f"Loaded {symbol}: {len(df)} records")
        else:
            logger.warning(f"File not found: {file_path}")
    
    return all_data

def create_merged_dataset(yahoo_data):
    """Create merged dataset from Yahoo Finance data"""
    logger.info("Creating merged dataset")
    
    # Get common date range
    all_dates = set()
    for symbol, df in yahoo_data.items():
        all_dates.update(df.index)
    
    date_range = pd.date_range(
        start=min(all_dates), 
        end=max(all_dates), 
        freq='D'
    )
    
    # Create base DataFrame
    merged_df = pd.DataFrame(index=date_range)
    merged_df.index.name = 'date'
    
    # Add price data for each symbol
    for symbol, df in yahoo_data.items():
        # Resample to daily and forward fill
        df_daily = df.resample('D').last().fillna(method='ffill')
        
        # Add price columns
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df_daily.columns:
                merged_df[f'{symbol}_{col}'] = df_daily[col]
        
        # Add technical indicators if available
        for col in ['rsi', 'macd', 'volatility', 'sma_20', 'sma_50']:
            if col in df_daily.columns:
                merged_df[f'{symbol}_{col}'] = df_daily[col]
    
    # Add calendar features
    merged_df['year'] = merged_df.index.year
    merged_df['month'] = merged_df.index.month
    merged_df['day'] = merged_df.index.day
    merged_df['dayofweek'] = merged_df.index.dayofweek
    merged_df['dayofyear'] = merged_df.index.dayofyear
    merged_df['quarter'] = merged_df.index.quarter
    merged_df['is_weekend'] = (merged_df.index.dayofweek >= 5).astype(int)
    merged_df['is_holiday'] = 0
    
    # Add synthetic macro indicators
    np.random.seed(42)
    n_days = len(merged_df)
    merged_df['epu'] = np.random.normal(100, 20, n_days)
    merged_df['gpr'] = np.random.normal(50, 15, n_days)
    merged_df['vix'] = np.random.normal(20, 5, n_days)
    merged_df['dxy'] = np.random.normal(95, 3, n_days)
    merged_df['fed_funds_rate'] = np.random.normal(2.5, 0.5, n_days)
    
    # Add synthetic news features
    merged_df['news_sentiment'] = np.random.normal(0, 1, n_days)
    merged_df['news_volume'] = np.random.poisson(10, n_days)
    
    # Add synthetic GDELT features
    merged_df['gdelt_tone'] = np.random.normal(0, 2, n_days)
    merged_df['gdelt_goldstein'] = np.random.normal(0, 1, n_days)
    
    # Create target variables (next day prices)
    for symbol in ["CL=F", "NG=F", "MTF=F"]:
        close_col = f'{symbol}_close'
        if close_col in merged_df.columns:
            merged_df[f'{symbol}_next_day'] = merged_df[close_col].shift(-1)
            merged_df[f'{symbol}_price_change'] = merged_df[close_col].pct_change()
            merged_df[f'{symbol}_next_day_change'] = merged_df[f'{symbol}_next_day'].pct_change()
            merged_df[f'{symbol}_direction'] = (merged_df[f'{symbol}_next_day_change'] > 0).astype(int)
    
    # Create lagged features
    for symbol in ["CL=F", "NG=F", "MTF=F"]:
        close_col = f'{symbol}_close'
        if close_col in merged_df.columns:
            for lag in [1, 2, 3, 5, 10, 20]:
                merged_df[f'{close_col}_lag_{lag}'] = merged_df[close_col].shift(lag)
    
    # Fill missing values
    merged_df = merged_df.fillna(method='ffill').fillna(method='bfill')
    
    # Remove rows with any remaining NaN values
    merged_df = merged_df.dropna()
    
    logger.info(f"Created merged dataset: {merged_df.shape}")
    return merged_df

def save_processed_data(merged_df):
    """Save processed data"""
    logger.info("Saving processed data")
    
    # Create output directory
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    # Save merged data
    output_path = output_dir / "merged.csv"
    merged_df.to_csv(output_path)
    logger.info(f"Saved merged data to {output_path}")
    
    # Save summary
    summary = {
        'total_records': len(merged_df),
        'date_range': {
            'start': str(merged_df.index.min()),
            'end': str(merged_df.index.max())
        },
        'columns': list(merged_df.columns),
        'price_columns': [col for col in merged_df.columns if col.endswith('_close')],
        'target_columns': [col for col in merged_df.columns if col.endswith('_next_day')]
    }
    
    summary_path = output_dir / "data_summary.json"
    import json
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Saved data summary to {summary_path}")
    return output_path

def main():
    """Main preprocessing function"""
    logger.info("Starting simplified preprocessing")
    
    # Load Yahoo Finance data
    yahoo_data = load_yahoo_data()
    
    if not yahoo_data:
        logger.error("No Yahoo Finance data found")
        return
    
    # Create merged dataset
    merged_df = create_merged_dataset(yahoo_data)
    
    if merged_df.empty:
        logger.error("No data after merging")
        return
    
    # Save processed data
    output_path = save_processed_data(merged_df)
    
    logger.info("Preprocessing completed successfully!")
    logger.info(f"Final dataset shape: {merged_df.shape}")
    logger.info(f"Date range: {merged_df.index.min()} to {merged_df.index.max()}")
    logger.info(f"Price columns: {[col for col in merged_df.columns if col.endswith('_close')]}")
    logger.info(f"Target columns: {[col for col in merged_df.columns if col.endswith('_next_day')]}")

if __name__ == "__main__":
    main()
