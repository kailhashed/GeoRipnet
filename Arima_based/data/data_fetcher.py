"""
Data Fetcher for RippleNet-TFT
Fetches data from various free sources: Yahoo Finance, EIA, NewsAPI, GDELT, UN Comtrade
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import json
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional, Tuple
import zipfile
import io
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataFetcher:
    """Main data fetcher class for RippleNet-TFT"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_dir = Path(config['data']['raw_data_path'])
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # API keys from environment
        self.news_api_key = os.getenv('NEWS_API_KEY', '')
        self.eia_api_key = os.getenv('EIA_API_KEY', '')
        
        # Rate limiting
        self.request_delay = 0.1  # seconds between requests
        
    def fetch_yahoo_finance_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV data from Yahoo Finance"""
        logger.info(f"Fetching Yahoo Finance data for symbols: {symbols}")
        
        data = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date)
                
                if df.empty:
                    logger.warning(f"No data found for {symbol}")
                    continue
                
                # Clean column names
                df.columns = [col.lower().replace(' ', '_') for col in df.columns]
                df.index.name = 'date'
                df = df.reset_index()
                
                # Add technical indicators
                df = self._add_technical_indicators(df)
                
                data[symbol] = df
                logger.info(f"Fetched {len(df)} records for {symbol}")
                time.sleep(self.request_delay)
                
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                continue
        
        return data
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to price data"""
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'])
        
        # MACD
        macd_line, signal_line, histogram = self._calculate_macd(df['close'])
        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_histogram'] = histogram
        
        # Volatility (rolling 20-day)
        df['volatility'] = df['close'].pct_change().rolling(20).std() * np.sqrt(252)
        
        # Moving averages
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def fetch_eia_data(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Fetch energy data from EIA API"""
        logger.info("Fetching EIA data")
        
        if not self.eia_api_key:
            logger.warning("No EIA API key provided, skipping EIA data")
            return {}
        
        # EIA series IDs for energy commodities
        eia_series = {
            'crude_oil_production': 'PET.WCRFPUS2.W',
            'natural_gas_production': 'NG.N9010US2.M',
            'coal_production': 'COAL.PROD_TOT.US-TON',
            'crude_oil_storage': 'PET.WCESTUS1.W',
            'natural_gas_storage': 'NG.NW2_EPG0_SWO_R48_BCF.W'
        }
        
        data = {}
        base_url = "https://api.eia.gov/v2/seriesid/"
        
        for name, series_id in eia_series.items():
            try:
                url = f"{base_url}{series_id}"
                params = {
                    'api_key': self.eia_api_key,
                    'start': start_date,
                    'end': end_date,
                    'sort': [{'column': 'period', 'direction': 'desc'}]
                }
                
                response = requests.get(url, params=params)
                response.raise_for_status()
                
                json_data = response.json()
                if 'data' in json_data and json_data['data']:
                    df = pd.DataFrame(json_data['data'])
                    df['date'] = pd.to_datetime(df['period'])
                    df = df.set_index('date').sort_index()
                    data[name] = df
                    logger.info(f"Fetched EIA data for {name}")
                
                time.sleep(self.request_delay)
                
            except Exception as e:
                logger.error(f"Error fetching EIA data for {name}: {e}")
                continue
        
        return data
    
    def fetch_news_data(self, start_date: str, end_date: str, keywords: List[str] = None) -> pd.DataFrame:
        """Load real news data from existing files"""
        logger.info("Loading real news data from data/raw/news_headlines.csv")
        
        # Load real news data
        news_file = Path(self.config['data']['raw_data_path']) / 'news_headlines.csv'
        if news_file.exists():
            df = pd.read_csv(news_file)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            
            # Filter by date range
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            df = df[(df.index.date >= start_dt.date()) & (df.index.date <= end_dt.date())]
            
            # Group by date and aggregate
            df_aggregated = df.groupby(df.index.date).agg({
                'title': lambda x: ' | '.join(x),
                'description': lambda x: ' | '.join(x),
                'source': lambda x: ', '.join(x.unique()),
                'url': lambda x: ', '.join(x),
                'keyword': lambda x: ', '.join(x.unique())
            }).reset_index()
            df_aggregated.columns = ['date', 'title', 'description', 'source', 'url', 'keyword']
            
            logger.info(f"Loaded {len(df_aggregated)} days of real news data")
            return df_aggregated
        else:
            logger.warning("Real news data not found, returning empty DataFrame")
            return pd.DataFrame()
    
    def fetch_gdelt_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load real GDELT event data from existing files"""
        logger.info("Loading real GDELT data from data/raw/gdelt_events.csv")
        
        # Load real GDELT data
        gdelt_file = Path(self.config['data']['raw_data_path']) / 'gdelt_events.csv'
        if gdelt_file.exists():
            df = pd.read_csv(gdelt_file)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            
            # Filter by date range
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            df = df[(df.index.date >= start_dt.date()) & (df.index.date <= end_dt.date())]
            
            # Group by date and aggregate
            df_aggregated = df.groupby(df.index.date).agg({
                'title': lambda x: ' | '.join(x),
                'tone': 'mean',
                'goldstein_scale': 'mean',
                'country': lambda x: ', '.join(x.unique()),
                'event_type': lambda x: ', '.join(x.unique())
            }).reset_index()
            df_aggregated.columns = ['date', 'title', 'tone', 'goldstein_scale', 'country', 'event_type']
            
            logger.info(f"Loaded {len(df_aggregated)} days of real GDELT event data")
            return df_aggregated
        else:
            logger.warning("Real GDELT data not found, returning empty DataFrame")
            return pd.DataFrame()
    
    def download_comtrade_data(self) -> pd.DataFrame:
        """Download UN Comtrade trade data"""
        logger.info("Downloading UN Comtrade data")
        
        comtrade_dir = Path(self.config['data']['comtrade_path'])
        comtrade_dir.mkdir(parents=True, exist_ok=True)
        
        # UN Comtrade API for energy commodities
        # This is a simplified version - in practice, you'd need to handle
        # the complex UN Comtrade API or download CSV files
        
        # Energy commodity codes
        energy_commodities = {
            'crude_oil': '270900',  # Crude petroleum oils
            'natural_gas': '271111',  # Natural gas
            'coal': '270112',  # Bituminous coal
            'refined_oil': '271012'  # Light petroleum oils
        }
        
        # This would require more complex implementation
        # For now, return empty DataFrame
        logger.warning("UN Comtrade integration not fully implemented")
        return pd.DataFrame()
    
    def save_data(self, data: Dict[str, pd.DataFrame], data_type: str):
        """Save fetched data to files"""
        for name, df in data.items():
            if not df.empty:
                file_path = self.data_dir / f"{data_type}_{name}.csv"
                df.to_csv(file_path, index=False)
                logger.info(f"Saved {data_type}_{name}.csv with {len(df)} records")
    
    def fetch_all_data(self, start_date: str, end_date: str) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Fetch all data sources"""
        logger.info(f"Fetching all data from {start_date} to {end_date}")
        
        all_data = {}
        
        # Yahoo Finance data
        yahoo_data = self.fetch_yahoo_finance_data(
            self.config['data']['yahoo_symbols'],
            start_date, end_date
        )
        all_data['yahoo'] = yahoo_data
        self.save_data(yahoo_data, 'yahoo')
        
        # EIA data
        eia_data = self.fetch_eia_data(start_date, end_date)
        all_data['eia'] = eia_data
        self.save_data(eia_data, 'eia')
        
        # News data
        news_data = self.fetch_news_data(start_date, end_date)
        if not news_data.empty:
            all_data['news'] = {'headlines': news_data}
            self.save_data({'headlines': news_data}, 'news')
        
        # GDELT data
        gdelt_data = self.fetch_gdelt_data(start_date, end_date)
        if not gdelt_data.empty:
            all_data['gdelt'] = {'events': gdelt_data}
            self.save_data({'events': gdelt_data}, 'gdelt')
        
        # Comtrade data
        comtrade_data = self.download_comtrade_data()
        if not comtrade_data.empty:
            all_data['comtrade'] = {'trade': comtrade_data}
            self.save_data({'trade': comtrade_data}, 'comtrade')
        
        logger.info("Data fetching completed")
        return all_data

def main():
    """Main function for data fetching"""
    import yaml
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize fetcher
    fetcher = DataFetcher(config)
    
    # Fetch all data
    start_date = config['data']['start_date']
    end_date = config['data']['end_date']
    
    all_data = fetcher.fetch_all_data(start_date, end_date)
    
    print("Data fetching completed successfully!")

if __name__ == "__main__":
    main()
