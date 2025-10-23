"""
News Encoder for RippleNet-TFT
Uses FinBERT to encode news headlines into semantic embeddings
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import yaml
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsEncoder:
    """News encoder using FinBERT for semantic embeddings"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model_name = config['model']['news_encoder']['model_name']
        self.max_length = config['model']['news_encoder']['max_length']
        self.batch_size = config['model']['news_encoder']['batch_size']
        self.embedding_dim = config['model']['news_encoder']['embedding_dim']
        self.use_attention_pooling = config['model']['news_encoder']['attention_pooling']
        
        # Initialize tokenizer and model
        self._load_model()
        
        # Fallback sentiment analyzer
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
    def _load_model(self):
        """Load FinBERT model and tokenizer"""
        logger.info(f"Loading FinBERT model: {self.model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.eval()
            logger.info("FinBERT model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load FinBERT model: {e}")
            logger.info("Falling back to VADER sentiment analysis")
            self.model = None
            self.tokenizer = None
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess news text"""
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to string
        text = str(text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def encode_headlines_finbert(self, headlines: List[str]) -> np.ndarray:
        """Encode headlines using FinBERT"""
        if self.model is None or self.tokenizer is None:
            logger.warning("FinBERT model not available, using fallback")
            return self.encode_headlines_fallback(headlines)
        
        logger.info(f"Encoding {len(headlines)} headlines with FinBERT")
        
        # Preprocess headlines
        processed_headlines = [self.preprocess_text(h) for h in headlines]
        processed_headlines = [h for h in processed_headlines if h.strip()]
        
        if not processed_headlines:
            logger.warning("No valid headlines to encode")
            return np.zeros((1, self.embedding_dim))
        
        # Tokenize and encode in batches
        embeddings = []
        
        for i in range(0, len(processed_headlines), self.batch_size):
            batch_headlines = processed_headlines[i:i + self.batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_headlines,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                if self.use_attention_pooling:
                    # Use attention pooling
                    attention_weights = torch.softmax(
                        torch.mean(outputs.last_hidden_state, dim=-1), dim=-1
                    )
                    batch_embeddings = torch.sum(
                        outputs.last_hidden_state * attention_weights.unsqueeze(-1), dim=1
                    )
                else:
                    # Use [CLS] token embedding
                    batch_embeddings = outputs.last_hidden_state[:, 0, :]
                
                embeddings.append(batch_embeddings.numpy())
        
        # Concatenate all embeddings
        all_embeddings = np.vstack(embeddings)
        
        logger.info(f"Encoded {len(headlines)} headlines into {all_embeddings.shape} embeddings")
        return all_embeddings
    
    def encode_headlines_fallback(self, headlines: List[str]) -> np.ndarray:
        """Fallback encoding using VADER sentiment and TF-IDF"""
        logger.info("Using fallback encoding method")
        
        # Preprocess headlines
        processed_headlines = [self.preprocess_text(h) for h in headlines]
        processed_headlines = [h for h in processed_headlines if h.strip()]
        
        if not processed_headlines:
            return np.zeros((1, 10))  # Return small embedding
        
        # VADER sentiment analysis
        sentiment_scores = []
        for headline in processed_headlines:
            scores = self.sentiment_analyzer.polarity_scores(headline)
            sentiment_scores.append([
                scores['pos'], scores['neg'], scores['neu'], scores['compound']
            ])
        
        sentiment_scores = np.array(sentiment_scores)
        
        # TF-IDF features
        try:
            tfidf = TfidfVectorizer(max_features=6, stop_words='english')
            tfidf_features = tfidf.fit_transform(processed_headlines).toarray()
        except:
            tfidf_features = np.zeros((len(processed_headlines), 6))
        
        # Combine features
        combined_features = np.hstack([sentiment_scores, tfidf_features])
        
        logger.info(f"Created {combined_features.shape} fallback embeddings")
        return combined_features
    
    def encode_daily_news(self, news_data: pd.DataFrame) -> pd.DataFrame:
        """Encode daily news data"""
        logger.info("Encoding daily news data")
        
        if news_data.empty:
            logger.warning("No news data provided")
            return pd.DataFrame()
        
        # Group by date
        daily_embeddings = []
        
        for date, group in news_data.groupby('date'):
            headlines = group['title'].tolist()
            descriptions = group['description'].tolist()
            
            # Combine headlines and descriptions
            combined_text = []
            for h, d in zip(headlines, descriptions):
                if pd.notna(h) and pd.notna(d):
                    combined_text.append(f"{h} {d}")
                elif pd.notna(h):
                    combined_text.append(h)
                elif pd.notna(d):
                    combined_text.append(d)
            
            if combined_text:
                # Encode headlines
                embeddings = self.encode_headlines_finbert(combined_text)
                
                # Aggregate embeddings (mean pooling)
                if len(embeddings) > 1:
                    daily_embedding = np.mean(embeddings, axis=0)
                else:
                    daily_embedding = embeddings[0]
                
                daily_embeddings.append({
                    'date': date,
                    'news_embedding': daily_embedding,
                    'num_articles': len(combined_text)
                })
            else:
                # No articles for this date
                if self.model is not None:
                    embedding_dim = self.embedding_dim
                else:
                    embedding_dim = 10  # Fallback dimension
                
                daily_embeddings.append({
                    'date': date,
                    'news_embedding': np.zeros(embedding_dim),
                    'num_articles': 0
                })
        
        # Convert to DataFrame
        embeddings_df = pd.DataFrame(daily_embeddings)
        
        # Flatten embeddings into columns
        if not embeddings_df.empty and 'news_embedding' in embeddings_df.columns:
            embedding_cols = []
            for i in range(len(embeddings_df['news_embedding'].iloc[0])):
                embeddings_df[f'news_embedding_{i}'] = embeddings_df['news_embedding'].apply(
                    lambda x: x[i] if len(x) > i else 0
                )
                embedding_cols.append(f'news_embedding_{i}')
            
            # Drop the original embedding column
            embeddings_df = embeddings_df.drop('news_embedding', axis=1)
            
            logger.info(f"Created {len(embedding_cols)} news embedding features")
        
        logger.info(f"Encoded news for {len(embeddings_df)} days")
        return embeddings_df
    
    def create_news_features(self, news_data: pd.DataFrame) -> pd.DataFrame:
        """Create additional news features"""
        logger.info("Creating additional news features")
        
        if news_data.empty:
            return pd.DataFrame()
        
        features = []
        
        for date, group in news_data.groupby('date'):
            # Basic features
            num_articles = len(group)
            avg_title_length = group['title'].str.len().mean() if 'title' in group.columns else 0
            avg_desc_length = group['description'].str.len().mean() if 'description' in group.columns else 0
            
            # Source diversity
            unique_sources = group['source'].nunique() if 'source' in group.columns else 1
            
            # Keyword analysis
            energy_keywords = ['oil', 'gas', 'energy', 'crude', 'petroleum', 'OPEC', 'sanctions']
            keyword_counts = {}
            for keyword in energy_keywords:
                if 'title' in group.columns:
                    count = group['title'].str.lower().str.count(keyword).sum()
                else:
                    count = 0
                keyword_counts[f'keyword_{keyword}'] = count
            
            # Sentiment analysis (if not using FinBERT)
            if self.model is None and 'title' in group.columns:
                sentiments = []
                for title in group['title']:
                    if pd.notna(title):
                        scores = self.sentiment_analyzer.polarity_scores(str(title))
                        sentiments.append(scores['compound'])
                
                avg_sentiment = np.mean(sentiments) if sentiments else 0
            else:
                avg_sentiment = 0
            
            features.append({
                'date': date,
                'num_articles': num_articles,
                'avg_title_length': avg_title_length,
                'avg_desc_length': avg_desc_length,
                'unique_sources': unique_sources,
                'avg_sentiment': avg_sentiment,
                **keyword_counts
            })
        
        return pd.DataFrame(features)
    
    def process_news_data(self, news_data: pd.DataFrame) -> pd.DataFrame:
        """Main news processing pipeline"""
        logger.info("Processing news data")
        
        if news_data.empty:
            logger.warning("No news data to process")
            return pd.DataFrame()
        
        # Encode daily news
        embeddings_df = self.encode_daily_news(news_data)
        
        # Create additional features
        features_df = self.create_news_features(news_data)
        
        # Merge embeddings and features
        if not embeddings_df.empty and not features_df.empty:
            merged_df = pd.merge(embeddings_df, features_df, on='date', how='outer')
        elif not embeddings_df.empty:
            merged_df = embeddings_df
        elif not features_df.empty:
            merged_df = features_df
        else:
            merged_df = pd.DataFrame()
        
        # Fill missing values
        merged_df = merged_df.fillna(0)
        
        logger.info(f"Processed news data with shape: {merged_df.shape}")
        return merged_df

def main():
    """Main function for news encoding"""
    import yaml
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize news encoder
    encoder = NewsEncoder(config)
    
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
    
    # Process news data
    processed_news = encoder.process_news_data(sample_news)
    
    print(f"Processed news data shape: {processed_news.shape}")
    print("News encoding completed successfully!")

if __name__ == "__main__":
    main()
