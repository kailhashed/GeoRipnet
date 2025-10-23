"""
Real-time data fetcher using curl to bypass SSL issues.
Fetches real market data and stores in data folder.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import subprocess
import time
from typing import Dict, List, Optional
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealtimeDataFetcher:
    """Real-time data fetcher using curl to bypass SSL issues."""
    
    def __init__(self, data_dir: str = "data"):
        self.curl_cmd = "curl"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def fetch_yahoo_data_curl(self, symbol: str) -> pd.DataFrame:
        """Fetch Yahoo Finance data using curl."""
        try:
            # Yahoo Finance API URL
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            params = "range=5y&interval=1d&includePrePost=false"
            
            # Build curl command
            cmd = [
                self.curl_cmd,
                "-s",  # Silent
                "-L",  # Follow redirects
                "-H", f"User-Agent: {self.headers['User-Agent']}",
                f"{url}?{params}"
            ]
            
            # Execute curl command
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and result.stdout.strip():
                try:
                    data = json.loads(result.stdout)
                    
                    if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
                        result_data = data['chart']['result'][0]
                        
                        if 'timestamp' in result_data and 'indicators' in result_data:
                            timestamps = result_data['timestamp']
                            quotes = result_data['indicators']['quote'][0]
                            
                            # Create DataFrame
                            df = pd.DataFrame({
                                'Open': quotes['open'],
                                'High': quotes['high'],
                                'Low': quotes['low'],
                                'Close': quotes['close'],
                                'Volume': quotes['volume']
                            })
                            
                            # Set datetime index
                            df.index = pd.to_datetime(timestamps, unit='s')
                            df = df.dropna()
                            
                            return df
                    else:
                        logger.warning(f"Yahoo Finance API error for {symbol}: No chart data")
                        return pd.DataFrame()
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON decode error for {symbol}: {e}")
                    return pd.DataFrame()
            else:
                logger.warning(f"curl failed for {symbol}: {result.stderr}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance data with curl for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_alternative_symbols(self, symbol: str) -> List[str]:
        """Get alternative symbols for commodities."""
        alternatives = {
            'CL=F': ['CL', 'CL=F', 'CLZ25', 'CLH25'],  # Crude Oil alternatives
            'NG=F': ['NG', 'NG=F', 'NGZ25', 'NGH25'],  # Natural Gas alternatives
            'CO1=F': ['CO', 'CO=F', 'COZ25', 'COH25'], # Coal alternatives
            'EL=F': ['EL', 'EL=F', 'ELZ25', 'ELH25']   # Electricity alternatives
        }
        return alternatives.get(symbol, [symbol])
    
    def save_data_to_file(self, symbol: str, df: pd.DataFrame) -> str:
        """Save data to CSV file in data directory."""
        filename = f"{symbol.replace('=', '_').replace('.', '_')}_data.csv"
        filepath = self.data_dir / filename
        df.to_csv(filepath)
        logger.info(f"Saved {len(df)} records for {symbol} to {filepath}")
        return str(filepath)
    
    def load_data_from_file(self, symbol: str) -> pd.DataFrame:
        """Load data from CSV file in data directory."""
        filename = f"{symbol.replace('=', '_').replace('.', '_')}_data.csv"
        filepath = self.data_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            logger.info(f"Loaded {len(df)} records for {symbol} from {filepath}")
            return df
        else:
            logger.warning(f"No saved data found for {symbol}")
            return pd.DataFrame()
    
    def fetch_real_market_data(self, symbols: List[str], force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        """Fetch real market data using curl with caching."""
        print("📊 Fetching REAL market data using curl...")
        
        market_data = {}
        
        for symbol in symbols:
            print(f"  Fetching {symbol}...")
            df = None
            
            # Check if we have cached data and don't need to refresh
            if not force_refresh:
                df = self.load_data_from_file(symbol)
                if not df.empty:
                    market_data[symbol] = df
                    print(f"    ✅ {symbol}: {len(df)} records (CACHED DATA)")
                    print(f"    📅 Date range: {df.index[0].date()} to {df.index[-1].date()}")
                    print(f"    💰 Latest price: ${df['Close'].iloc[-1]:.2f}")
                    continue
            
            # Try original symbol first
            try:
                print(f"    Trying {symbol}...")
                df = self.fetch_yahoo_data_curl(symbol)
                
                if not df.empty and len(df) > 100:
                    market_data[symbol] = df
                    self.save_data_to_file(symbol, df)
                    print(f"    ✅ {symbol}: {len(df)} records from Yahoo Finance curl")
                    print(f"    📅 Date range: {df.index[0].date()} to {df.index[-1].date()}")
                    print(f"    💰 Latest price: ${df['Close'].iloc[-1]:.2f}")
                    continue
                else:
                    print(f"    ⚠️  {symbol}: Insufficient data")
            except Exception as e:
                print(f"    ❌ {symbol}: {e}")
            
            # Try alternative symbols
            alternatives = self.fetch_alternative_symbols(symbol)
            for alt_symbol in alternatives:
                if alt_symbol != symbol:  # Skip if already tried
                    try:
                        print(f"    Trying alternative {alt_symbol}...")
                        df = self.fetch_yahoo_data_curl(alt_symbol)
                        
                        if not df.empty and len(df) > 100:
                            market_data[symbol] = df  # Use original symbol as key
                            self.save_data_to_file(symbol, df)
                            print(f"    ✅ {symbol} (via {alt_symbol}): {len(df)} records from Yahoo Finance curl")
                            print(f"    📅 Date range: {df.index[0].date()} to {df.index[-1].date()}")
                            print(f"    💰 Latest price: ${df['Close'].iloc[-1]:.2f}")
                            break
                        else:
                            print(f"    ⚠️  {alt_symbol}: Insufficient data")
                    except Exception as e:
                        print(f"    ❌ {alt_symbol}: {e}")
                        continue
            
            if symbol not in market_data:
                print(f"    ❌ All sources failed for {symbol}")
        
        return market_data

def fetch_realtime_data(symbols: List[str] = None, force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
    """Fetch real-time data for energy commodities."""
    if symbols is None:
        symbols = ['CL=F', 'NG=F', 'CO1=F', 'EL=F']
    
    fetcher = RealtimeDataFetcher()
    return fetcher.fetch_real_market_data(symbols, force_refresh)

if __name__ == "__main__":
    # Test the real-time data fetcher
    data = fetch_realtime_data()
    print(f"\n📊 Data Summary:")
    for symbol, df in data.items():
        print(f"  {symbol}: {len(df)} records, Latest: ${df['Close'].iloc[-1]:.2f}")