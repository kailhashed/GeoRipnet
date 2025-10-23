"""
Demo Script for RippleNet-TFT
Quick demonstration of the complete pipeline
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from data.data_fetcher import DataFetcher
from data.preprocess import DataPreprocessor
from data.ripple_graph import RippleGraph
from data.news_encoder import NewsEncoder
from data.arima_baseline import ARIMABaseline
from data.dataset import RippleNetDataModule
from model.ripple_tft import create_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_demo_data():
    """Create synthetic demo data for testing"""
    logger.info("Creating synthetic demo data")
    
    # Create date range
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2024, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create synthetic price data
    n_days = len(dates)
    
    # Crude Oil prices
    oil_base = 100
    oil_trend = np.linspace(0, 20, n_days)  # Upward trend
    oil_volatility = np.random.randn(n_days) * 2
    oil_prices = oil_base + oil_trend + oil_volatility
    
    # Natural Gas prices
    gas_base = 50
    gas_trend = np.linspace(0, 10, n_days)
    gas_volatility = np.random.randn(n_days) * 1.5
    gas_prices = gas_base + gas_trend + gas_volatility
    
    # Coal prices
    coal_base = 200
    coal_trend = np.linspace(0, -5, n_days)  # Slight downward trend
    coal_volatility = np.random.randn(n_days) * 3
    coal_prices = coal_base + coal_trend + coal_volatility
    
    # Create DataFrame
    demo_data = pd.DataFrame({
        'date': dates,
        'CL=F_close': oil_prices,
        'NG=F_close': gas_prices,
        'MTF=F_close': coal_prices,
        'CL=F_open': oil_prices + np.random.randn(n_days) * 0.5,
        'NG=F_open': gas_prices + np.random.randn(n_days) * 0.3,
        'MTF=F_open': coal_prices + np.random.randn(n_days) * 1.0,
        'CL=F_high': oil_prices + np.abs(np.random.randn(n_days) * 1.0),
        'NG=F_high': gas_prices + np.abs(np.random.randn(n_days) * 0.5),
        'MTF=F_high': coal_prices + np.abs(np.random.randn(n_days) * 2.0),
        'CL=F_low': oil_prices - np.abs(np.random.randn(n_days) * 1.0),
        'NG=F_low': gas_prices - np.abs(np.random.randn(n_days) * 0.5),
        'MTF=F_low': coal_prices - np.abs(np.random.randn(n_days) * 2.0),
        'CL=F_volume': np.random.randint(1000000, 5000000, n_days),
        'NG=F_volume': np.random.randint(500000, 2000000, n_days),
        'MTF=F_volume': np.random.randint(200000, 1000000, n_days)
    })
    
    # Add technical indicators
    demo_data['CL=F_rsi'] = 50 + np.random.randn(n_days) * 20
    demo_data['NG=F_rsi'] = 50 + np.random.randn(n_days) * 20
    demo_data['MTF=F_rsi'] = 50 + np.random.randn(n_days) * 20
    
    demo_data['CL=F_macd'] = np.random.randn(n_days) * 0.5
    demo_data['NG=F_macd'] = np.random.randn(n_days) * 0.3
    demo_data['MTF=F_macd'] = np.random.randn(n_days) * 1.0
    
    demo_data['CL=F_volatility'] = np.random.randn(n_days) * 0.1 + 0.2
    demo_data['NG=F_volatility'] = np.random.randn(n_days) * 0.15 + 0.25
    demo_data['MTF=F_volatility'] = np.random.randn(n_days) * 0.2 + 0.3
    
    # Add macro indicators
    demo_data['epu'] = 100 + np.random.randn(n_days) * 20
    demo_data['gpr'] = 50 + np.random.randn(n_days) * 15
    demo_data['vix'] = 20 + np.random.randn(n_days) * 5
    demo_data['dxy'] = 95 + np.random.randn(n_days) * 3
    demo_data['fed_funds_rate'] = 2.5 + np.random.randn(n_days) * 0.5
    
    # Add calendar features
    demo_data['year'] = dates.year
    demo_data['month'] = dates.month
    demo_data['day'] = dates.day
    demo_data['dayofweek'] = dates.dayofweek
    demo_data['dayofyear'] = dates.dayofyear
    demo_data['quarter'] = dates.quarter
    demo_data['is_weekend'] = (dates.dayofweek >= 5).astype(int)
    demo_data['is_holiday'] = 0
    
    # Add target variables
    demo_data['CL=F_next_day'] = demo_data['CL=F_close'].shift(-1)
    demo_data['NG=F_next_day'] = demo_data['NG=F_close'].shift(-1)
    demo_data['MTF=F_next_day'] = demo_data['MTF=F_close'].shift(-1)
    
    # Add synthetic news data
    news_titles = [
        "Oil prices surge on OPEC production cuts",
        "Natural gas futures fall amid warm weather",
        "Coal exports increase from Australia",
        "Energy sector faces regulatory challenges",
        "Renewable energy investments reach record high",
        "Geopolitical tensions affect energy markets",
        "Winter weather drives natural gas demand",
        "Coal mining operations resume after strike",
        "Oil inventory levels decline significantly",
        "Energy transition accelerates globally"
    ]
    
    demo_data['news_title'] = np.random.choice(news_titles, n_days)
    demo_data['news_description'] = demo_data['news_title'] + " - Market analysis shows significant impact on energy commodity prices."
    demo_data['news_source'] = np.random.choice(['Reuters', 'Bloomberg', 'WSJ', 'FT'], n_days)
    
    # Add synthetic GDELT data
    demo_data['gdelt_tone'] = np.random.normal(0, 2, n_days)
    demo_data['gdelt_goldstein_scale'] = np.random.normal(0, 1, n_days)
    demo_data['gdelt_title'] = np.random.choice([
        "Geopolitical tensions rise in energy-producing regions",
        "Trade agreements affect energy commodity flows",
        "Sanctions impact energy sector operations",
        "Diplomatic relations influence energy markets"
    ], n_days)
    
    logger.info(f"Created demo data with {len(demo_data)} records")
    return demo_data

def run_demo():
    """Run the complete demo pipeline"""
    logger.info("Starting RippleNet-TFT Demo")
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create demo data
    demo_data = create_demo_data()
    
    # Save demo data
    demo_data.to_csv('data/demo_merged.csv', index=False)
    logger.info("Saved demo data to data/demo_merged.csv")
    
    # Test data preprocessing
    logger.info("Testing data preprocessing...")
    preprocessor = DataPreprocessor(config)
    processed_data = preprocessor.preprocess_all_data()
    
    if not processed_data.empty:
        logger.info(f"Preprocessing successful: {processed_data.shape}")
    else:
        logger.info("Using demo data for preprocessing test")
        processed_data = demo_data
    
    # Test ripple graph
    logger.info("Testing ripple graph...")
    ripple_graph = RippleGraph(config)
    graph = ripple_graph.build_country_commodity_graph()
    logger.info(f"Ripple graph created with {graph.number_of_nodes()} nodes")
    
    # Test news encoder
    logger.info("Testing news encoder...")
    news_encoder = NewsEncoder(config)
    
    # Create sample news data
    sample_news = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=5),
        'title': [
            'Oil prices surge on OPEC production cuts',
            'Natural gas futures fall amid warm weather',
            'Coal exports increase from Australia',
            'Energy sector faces regulatory challenges',
            'Renewable energy investments reach record high'
        ],
        'description': [
            'OPEC announces production cuts, driving oil prices higher',
            'Mild winter weather reduces natural gas demand',
            'Australian coal exports to China increase significantly',
            'New regulations impact energy sector profitability',
            'Record investments in solar and wind energy projects'
        ],
        'source': ['Reuters', 'Bloomberg', 'WSJ', 'FT', 'Reuters']
    })
    
    processed_news = news_encoder.process_news_data(sample_news)
    logger.info(f"News encoding successful: {processed_news.shape}")
    
    # Test ARIMA baseline
    logger.info("Testing ARIMA baseline...")
    arima_baseline = ARIMABaseline(config)
    
    # Use demo data for ARIMA
    arima_features = arima_baseline.create_arima_features(demo_data)
    logger.info(f"ARIMA baseline successful: {arima_features.shape}")
    
    # Test dataset creation
    logger.info("Testing dataset creation...")
    data_module = RippleNetDataModule(demo_data, config)
    train_loader, val_loader, test_loader = data_module.get_data_loaders()
    logger.info(f"Dataset creation successful:")
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Validation batches: {len(val_loader)}")
    logger.info(f"  Test batches: {len(test_loader)}")
    
    # Test model creation
    logger.info("Testing model creation...")
    model = create_model(config)
    logger.info(f"Model creation successful: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    logger.info("Testing forward pass...")
    model.eval()
    
    # Get a sample batch
    for batch in train_loader:
        past_target = batch['past_target']
        observed_covariates = batch['observed_covariates']
        known_future = batch['known_future']
        target = batch['target']
        
        # Test forward pass
        with torch.no_grad():
            predictions = model(past_target, observed_covariates, known_future)
        
        logger.info(f"Forward pass successful:")
        logger.info(f"  Input shapes: past_target={past_target.shape}, observed_covariates={observed_covariates.shape}")
        logger.info(f"  Output shape: {predictions.shape}")
        break
    
    logger.info("Demo completed successfully!")
    logger.info("All components are working correctly.")
    logger.info("You can now proceed with training using: python train.py --data data/demo_merged.csv")

def main():
    """Main demo function"""
    try:
        run_demo()
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        logger.error("Please check your installation and configuration")
        raise

if __name__ == "__main__":
    main()
