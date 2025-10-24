#!/usr/bin/env python3
"""
Real-time RippleNet-TFT Predictor
Deployed on NVIDIA DGX A100 for live energy commodity price predictions
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import json
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealtimeRippleNetTFT:
    """Real-time prediction system for energy commodity prices"""
    
    def __init__(self, model_path: str, config_path: str):
        self.model_path = model_path
        self.config_path = config_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scalers = None
        self.config = None
        
        # Load configuration
        self.load_config()
        
        # Load model and scalers
        self.load_model()
        
        # Initialize data fetchers
        self.setup_data_fetchers()
        
    def load_config(self):
        """Load configuration from YAML file"""
        import yaml
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        logger.info("Configuration loaded successfully")
        
    def load_model(self):
        """Load trained model and scalers"""
        try:
            # Load model
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model = checkpoint['model']
            self.model.eval()
            
            # Load scalers
            self.scalers = checkpoint['scalers']
            
            logger.info(f"Model loaded successfully from {self.model_path}")
            logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
            
    def setup_data_fetchers(self):
        """Setup real-time data fetchers"""
        self.symbols = ['CL=F', 'NG=F', 'MTF=F']  # Oil, Natural Gas, Coal
        self.lookback_window = self.config['training']['lookback_window']
        
    def fetch_realtime_data(self) -> pd.DataFrame:
        """Fetch real-time market data"""
        logger.info("Fetching real-time market data...")
        
        all_data = []
        
        for symbol in self.symbols:
            try:
                # Fetch data using yfinance
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1y", interval="1d")
                
                if not data.empty:
                    # Reset index to get date as column
                    data = data.reset_index()
                    data['symbol'] = symbol
                    data['date'] = pd.to_datetime(data['Date'])
                    data = data.drop('Date', axis=1)
                    
                    # Rename columns to lowercase
                    data.columns = [col.lower().replace(' ', '_') for col in data.columns]
                    
                    all_data.append(data)
                    logger.info(f"Fetched {len(data)} records for {symbol}")
                else:
                    logger.warning(f"No data received for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                
        if not all_data:
            raise ValueError("No market data could be fetched")
            
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Pivot to get prices by symbol
        price_df = combined_df.pivot_table(
            index='date', 
            columns='symbol', 
            values=['open', 'high', 'low', 'close', 'volume'],
            aggfunc='last'
        )
        
        # Flatten column names
        price_df.columns = [f"{col[1]}_{col[0]}" for col in price_df.columns]
        price_df = price_df.fillna(method='ffill').fillna(method='bfill')
        
        logger.info(f"Combined data shape: {price_df.shape}")
        return price_df
        
    def fetch_realtime_news(self) -> Tuple[np.ndarray, np.ndarray]:
        """Fetch real-time news sentiment using FinBERT"""
        try:
            # This would integrate with your news API
            # For now, return zeros as placeholder
            logger.info("Fetching real-time news sentiment...")
            
            # Placeholder - in production, integrate with news API
            news_sentiment = np.zeros(1)
            news_volume = np.zeros(1)
            
            return news_sentiment, news_volume
            
        except Exception as e:
            logger.error(f"Error fetching news data: {e}")
            return np.zeros(1), np.zeros(1)
            
    def fetch_realtime_gdelt(self) -> Tuple[np.ndarray, np.ndarray]:
        """Fetch real-time GDELT data"""
        try:
            logger.info("Fetching real-time GDELT data...")
            
            # Placeholder - in production, integrate with GDELT API
            gdelt_tone = np.zeros(1)
            gdelt_goldstein = np.zeros(1)
            
            return gdelt_tone, gdelt_goldstein
            
        except Exception as e:
            logger.error(f"Error fetching GDELT data: {e}")
            return np.zeros(1), np.zeros(1)
            
    def create_features(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators and features"""
        logger.info("Creating technical indicators...")
        
        for symbol in self.symbols:
            symbol_key = symbol.replace('=', '')
            
            # RSI
            price_df[f'{symbol_key}_rsi'] = self.calculate_rsi(price_df[f'{symbol_key}_close'])
            
            # MACD
            macd, macd_signal, macd_hist = self.calculate_macd(price_df[f'{symbol_key}_close'])
            price_df[f'{symbol_key}_macd'] = macd
            price_df[f'{symbol_key}_macd_signal'] = macd_signal
            price_df[f'{symbol_key}_macd_hist'] = macd_hist
            
            # Volatility
            price_df[f'{symbol_key}_volatility'] = price_df[f'{symbol_key}_close'].rolling(20).std()
            
            # Moving averages
            price_df[f'{symbol_key}_sma_20'] = price_df[f'{symbol_key}_close'].rolling(20).mean()
            price_df[f'{symbol_key}_sma_50'] = price_df[f'{symbol_key}_close'].rolling(50).mean()
            
        # Add time features
        price_df['year'] = price_df.index.year
        price_df['month'] = price_df.index.month
        price_df['day'] = price_df.index.day
        price_df['dayofweek'] = price_df.index.dayofweek
        price_df['dayofyear'] = price_df.index.dayofyear
        price_df['quarter'] = price_df.index.quarter
        price_df['is_weekend'] = (price_df['dayofweek'] >= 5).astype(int)
        price_df['is_holiday'] = 0  # Placeholder for holiday detection
        
        # Fill NaN values
        price_df = price_df.fillna(method='ffill').fillna(method='bfill')
        
        return price_df
        
    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
        
    def prepare_prediction_data(self, price_df: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """Prepare data for prediction"""
        logger.info("Preparing prediction data...")
        
        # Get the last lookback_window days
        recent_data = price_df.tail(self.lookback_window)
        
        # Separate features
        target_columns = [col for col in recent_data.columns if col.endswith('_close')]
        past_target = recent_data[target_columns].values
        
        # Observed covariates (news, economic indicators)
        observed_cov_columns = ['news_sentiment', 'news_volume', 'gdelt_tone', 'gdelt_goldstein']
        observed_cov = np.zeros((self.lookback_window, len(observed_cov_columns)))
        
        # Known future features (time features)
        known_future_columns = ['year', 'month', 'day', 'dayofweek', 'dayofyear', 'quarter', 'is_weekend', 'is_holiday']
        known_future = recent_data[known_future_columns].values
        
        # Convert to tensors
        past_target_tensor = torch.FloatTensor(past_target).unsqueeze(0)  # Add batch dimension
        observed_cov_tensor = torch.FloatTensor(observed_cov).unsqueeze(0)
        known_future_tensor = torch.FloatTensor(known_future).unsqueeze(0)
        
        return {
            'past_target': past_target_tensor,
            'observed_covariates': observed_cov_tensor,
            'known_future': known_future_tensor
        }
        
    def predict(self, data: Dict[str, torch.Tensor]) -> np.ndarray:
        """Make prediction using the trained model"""
        logger.info("Making prediction...")
        
        with torch.no_grad():
            # Move data to device
            past_target = data['past_target'].to(self.device)
            observed_cov = data['observed_covariates'].to(self.device)
            known_future = data['known_future'].to(self.device)
            
            # Make prediction
            prediction = self.model(past_target, observed_cov, known_future)
            
            # Move back to CPU and convert to numpy
            prediction = prediction.cpu().numpy()
            
            # Inverse transform if scalers are available
            if self.scalers and 'target' in self.scalers:
                prediction = self.scalers['target'].inverse_transform(prediction)
                
        return prediction
        
    def run_realtime_prediction(self) -> Dict[str, float]:
        """Run complete real-time prediction pipeline"""
        logger.info("🚀 Starting real-time prediction pipeline...")
        
        try:
            # Fetch real-time data
            price_df = self.fetch_realtime_data()
            
            # Create features
            price_df = self.create_features(price_df)
            
            # Prepare prediction data
            prediction_data = self.prepare_prediction_data(price_df)
            
            # Make prediction
            prediction = self.predict(prediction_data)
            
            # Format results
            symbols = ['CL=F', 'NG=F', 'MTF=F']
            results = {}
            
            for i, symbol in enumerate(symbols):
                if i < prediction.shape[1]:
                    results[f'{symbol}_prediction'] = float(prediction[0, i])
                else:
                    results[f'{symbol}_prediction'] = 0.0
                    
            # Add metadata
            results['timestamp'] = datetime.now().isoformat()
            results['model_version'] = 'RippleNet-TFT-v1.0'
            results['device'] = str(self.device)
            
            logger.info(f"✅ Prediction completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
            
    def run_continuous_predictions(self, interval_minutes: int = 60):
        """Run continuous predictions at specified intervals"""
        logger.info(f"🔄 Starting continuous predictions every {interval_minutes} minutes...")
        
        while True:
            try:
                # Run prediction
                results = self.run_realtime_prediction()
                
                # Save results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_file = f"results/realtime_prediction_{timestamp}.json"
                
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)
                    
                logger.info(f"📁 Results saved to {results_file}")
                
                # Wait for next interval
                time.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                logger.info("🛑 Stopping continuous predictions...")
                break
            except Exception as e:
                logger.error(f"❌ Error in continuous predictions: {e}")
                time.sleep(60)  # Wait 1 minute before retrying

def main():
    """Main function for real-time prediction"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RippleNet-TFT Real-time Predictor')
    parser.add_argument('--model', default='checkpoints/best_model.pt', help='Path to model checkpoint')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--continuous', action='store_true', help='Run continuous predictions')
    parser.add_argument('--interval', type=int, default=60, help='Prediction interval in minutes')
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = RealtimeRippleNetTFT(args.model, args.config)
    
    if args.continuous:
        # Run continuous predictions
        predictor.run_continuous_predictions(args.interval)
    else:
        # Run single prediction
        results = predictor.run_realtime_prediction()
        print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
