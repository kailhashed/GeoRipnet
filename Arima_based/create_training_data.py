"""
Create training data for RippleNet-TFT
Simple approach using available Yahoo Finance data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_real_news_sentiment(data_dir: Path, date_index: pd.DatetimeIndex) -> tuple:
    """Load real news data and calculate sentiment scores using FinBERT"""
    logger.info("Loading real news sentiment data with FinBERT")
    
    news_file = data_dir / 'news_headlines.csv'
    logger.info(f"Looking for news file at: {news_file}")
    logger.info(f"File exists: {news_file.exists()}")
    if not news_file.exists():
        logger.warning("Real news data not found, using zeros")
        return np.zeros(len(date_index)), np.zeros(len(date_index))
    
    # Load news data
    news_df = pd.read_csv(news_file)
    news_df['date'] = pd.to_datetime(news_df['date'])
    news_df = news_df.set_index('date')
    
    # Initialize FinBERT sentiment analyzer
    try:
        # Use FinBERT for financial sentiment analysis
        model_name = "ProsusAI/finbert"
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            device=0 if torch.cuda.is_available() else -1
        )
        logger.info("FinBERT model loaded successfully")
        use_finbert = True
    except Exception as e:
        logger.warning(f"Failed to load FinBERT: {e}. Falling back to VADER")
        analyzer = SentimentIntensityAnalyzer()
        use_finbert = False
    
    # Calculate daily sentiment
    daily_sentiment = []
    daily_volume = []
    
    for date in date_index:
        # Get news for this date
        if hasattr(date, 'date'):
            target_date = date.date()
        else:
            target_date = date
        day_news = news_df[news_df.index.date == target_date]
        
        if len(day_news) > 0:
            # Calculate sentiment for all headlines
            sentiments = []
            for _, row in day_news.iterrows():
                title = str(row.get('title', ''))
                description = str(row.get('description', ''))
                combined_text = f"{title} {description}"
                
                if combined_text.strip():
                    if use_finbert:
                        # Use FinBERT for financial sentiment
                        try:
                            result = sentiment_pipeline(combined_text[:512])  # Limit length
                            # FinBERT returns: [{'label': 'positive/negative', 'score': 0.xx}]
                            if result[0]['label'] == 'positive':
                                sentiment_score = result[0]['score']
                            else:  # negative
                                sentiment_score = -result[0]['score']
                            sentiments.append(sentiment_score)
                        except Exception as e:
                            logger.warning(f"FinBERT error for text: {e}")
                            # Fallback to VADER
                            scores = analyzer.polarity_scores(combined_text)
                            sentiments.append(scores['compound'])
                    else:
                        # Use VADER fallback
                        scores = analyzer.polarity_scores(combined_text)
                        sentiments.append(scores['compound'])
            
            # Average sentiment for the day
            avg_sentiment = np.mean(sentiments) if sentiments else 0
            volume = len(day_news)
        else:
            avg_sentiment = 0
            volume = 0
        
        daily_sentiment.append(avg_sentiment)
        daily_volume.append(volume)
    
    logger.info(f"Calculated sentiment for {len(daily_sentiment)} days using {'FinBERT' if use_finbert else 'VADER'}")
    return np.array(daily_sentiment), np.array(daily_volume)

def load_real_gdelt_data(data_dir: Path, date_index: pd.DatetimeIndex) -> tuple:
    """Load real GDELT data and extract tone and Goldstein scale"""
    logger.info("Loading real GDELT data")
    
    gdelt_file = data_dir / 'gdelt_events.csv'
    if not gdelt_file.exists():
        logger.warning("Real GDELT data not found, using zeros")
        return np.zeros(len(date_index)), np.zeros(len(date_index))
    
    # Load GDELT data
    gdelt_df = pd.read_csv(gdelt_file)
    gdelt_df['date'] = pd.to_datetime(gdelt_df['date'])
    gdelt_df = gdelt_df.set_index('date')
    
    # Calculate daily tone and Goldstein scale
    daily_tone = []
    daily_goldstein = []
    
    for date in date_index:
        # Get events for this date
        if hasattr(date, 'date'):
            target_date = date.date()
        else:
            target_date = date
        day_events = gdelt_df[gdelt_df.index.date == target_date]
        
        if len(day_events) > 0:
            # Average tone and Goldstein scale for the day
            avg_tone = day_events['tone'].mean() if 'tone' in day_events.columns else 0
            avg_goldstein = day_events['goldstein_scale'].mean() if 'goldstein_scale' in day_events.columns else 0
        else:
            avg_tone = 0
            avg_goldstein = 0
        
        daily_tone.append(avg_tone)
        daily_goldstein.append(avg_goldstein)
    
    logger.info(f"Calculated GDELT features for {len(daily_tone)} days")
    return np.array(daily_tone), np.array(daily_goldstein)

def create_training_dataset():
    """Create training dataset from available data"""
    logger.info("Creating training dataset")
    
    # Load Yahoo Finance data
    data_dir = Path("data/raw")
    symbols = ["CL=F", "NG=F", "MTF=F"]
    
    all_dataframes = []
    
    for symbol in symbols:
        file_path = data_dir / f"yahoo_{symbol}.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'], utc=True)
            df['symbol'] = symbol
            all_dataframes.append(df)
            logger.info(f"Loaded {symbol}: {len(df)} records")
    
    if not all_dataframes:
        logger.error("No data found")
        return None
    
    # Combine all data
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    logger.info(f"Combined data: {len(combined_df)} records")
    
    # Pivot to get prices by symbol
    price_df = combined_df.pivot_table(
        index='date', 
        columns='symbol', 
        values=['open', 'high', 'low', 'close', 'volume'],
        aggfunc='last'
    )
    
    # Flatten column names
    price_df.columns = [f"{col[1]}_{col[0]}" for col in price_df.columns]
    price_df = price_df.reset_index()
    
    # Fill missing values
    price_df = price_df.fillna(method='ffill').fillna(method='bfill')
    
    # Add calendar features
    price_df['year'] = price_df['date'].dt.year
    price_df['month'] = price_df['date'].dt.month
    price_df['day'] = price_df['date'].dt.day
    price_df['dayofweek'] = price_df['date'].dt.dayofweek
    price_df['dayofyear'] = price_df['date'].dt.dayofyear
    price_df['quarter'] = price_df['date'].dt.quarter
    price_df['is_weekend'] = (price_df['date'].dt.dayofweek >= 5).astype(int)
    price_df['is_holiday'] = 0
    
    # Add synthetic macro indicators
    np.random.seed(42)
    n_days = len(price_df)
    price_df['epu'] = np.random.normal(100, 20, n_days)
    price_df['gpr'] = np.random.normal(50, 15, n_days)
    price_df['vix'] = np.random.normal(20, 5, n_days)
    price_df['dxy'] = np.random.normal(95, 3, n_days)
    price_df['fed_funds_rate'] = np.random.normal(2.5, 0.5, n_days)
    
    # Load real news data and calculate sentiment
    news_sentiment, news_volume = load_real_news_sentiment(data_dir, price_df.index)
    price_df['news_sentiment'] = news_sentiment
    price_df['news_volume'] = news_volume
    
    # Load real GDELT data
    gdelt_tone, gdelt_goldstein = load_real_gdelt_data(data_dir, price_df.index)
    price_df['gdelt_tone'] = gdelt_tone
    price_df['gdelt_goldstein'] = gdelt_goldstein
    
    # Create target variables (next day prices)
    for symbol in symbols:
        close_col = f'{symbol}_close'
        if close_col in price_df.columns:
            price_df[f'{symbol}_next_day'] = price_df[close_col].shift(-1)
            price_df[f'{symbol}_price_change'] = price_df[close_col].pct_change()
            price_df[f'{symbol}_next_day_change'] = price_df[f'{symbol}_next_day'].pct_change()
            price_df[f'{symbol}_direction'] = (price_df[f'{symbol}_next_day_change'] > 0).astype(int)
    
    # Create lagged features
    for symbol in symbols:
        close_col = f'{symbol}_close'
        if close_col in price_df.columns:
            for lag in [1, 2, 3, 5, 10, 20]:
                price_df[f'{close_col}_lag_{lag}'] = price_df[close_col].shift(lag)
    
    # Add technical indicators
    for symbol in symbols:
        close_col = f'{symbol}_close'
        if close_col in price_df.columns:
            # RSI
            price_df[f'{symbol}_rsi'] = calculate_rsi(price_df[close_col])
            # MACD
            macd, signal, hist = calculate_macd(price_df[close_col])
            price_df[f'{symbol}_macd'] = macd
            price_df[f'{symbol}_macd_signal'] = signal
            price_df[f'{symbol}_macd_hist'] = hist
            # Volatility
            price_df[f'{symbol}_volatility'] = price_df[close_col].pct_change().rolling(20).std() * np.sqrt(252)
            # Moving averages
            price_df[f'{symbol}_sma_20'] = price_df[close_col].rolling(20).mean()
            price_df[f'{symbol}_sma_50'] = price_df[close_col].rolling(50).mean()
    
    # Fill missing values
    price_df = price_df.fillna(method='ffill').fillna(method='bfill')
    
    # Remove rows with any remaining NaN values
    price_df = price_df.dropna()
    
    logger.info(f"Final dataset shape: {price_df.shape}")
    return price_df

def calculate_rsi(prices, window=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def save_dataset(df):
    """Save the dataset"""
    logger.info("Saving dataset")
    
    # Create output directory
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    # Save merged data
    output_path = output_dir / "merged.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved dataset to {output_path}")
    
    # Save summary
    summary = {
        'total_records': len(df),
        'date_range': {
            'start': str(df['date'].min()),
            'end': str(df['date'].max())
        },
        'columns': list(df.columns),
        'price_columns': [col for col in df.columns if col.endswith('_close')],
        'target_columns': [col for col in df.columns if col.endswith('_next_day')]
    }
    
    summary_path = output_dir / "data_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Saved data summary to {summary_path}")
    return output_path

def main():
    """Main function"""
    logger.info("Creating training dataset for RippleNet-TFT")
    
    # Create dataset
    df = create_training_dataset()
    
    if df is None or df.empty:
        logger.error("Failed to create dataset")
        return
    
    # Save dataset
    output_path = save_dataset(df)
    
    logger.info("Dataset creation completed successfully!")
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"Price columns: {[col for col in df.columns if col.endswith('_close')]}")
    logger.info(f"Target columns: {[col for col in df.columns if col.endswith('_next_day')]}")

if __name__ == "__main__":
    main()

