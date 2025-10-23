"""
Complete Real-time LSTM model with market data and news sentiment.
Uses real-time data fetchers and stores results in organized folders.
"""

import sys
import os
import json
import yaml
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Optional, List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import our data fetchers
from realtime_data_fetcher import fetch_realtime_data
from robust_news_fetcher import fetch_robust_news

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

class RealtimeAttention(nn.Module):
    """Attention mechanism for real-time feature importance learning."""
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.attention_weights = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate attention scores
        attention_scores = self.attention_weights(x)  # [batch, seq_len, 1]
        attention_weights = torch.softmax(attention_scores, dim=1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention
        attended_features = x * attention_weights
        
        # Global average pooling
        output = torch.mean(attended_features, dim=1)
        
        return output, attention_weights

class TemporalStream(nn.Module):
    """Temporal stream for processing market data sequences."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.3):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bi-directional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply dropout and projection
        output = self.dropout(lstm_out)
        output = self.output_projection(output)
        
        return output

class ContextualStream(nn.Module):
    """Contextual stream for processing news sentiment and geopolitical features."""
    
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.3):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Dense layers for contextual features
        self.dense_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense_layers(x)

class RealtimeBiLSTM(nn.Module):
    """Real-time Bi-LSTM model for energy price forecasting."""
    
    def __init__(self, config: dict):
        super().__init__()
        
        self.config = config
        model_config = config['model']
        
        # Model parameters
        self.temporal_input_size = model_config.get('temporal_input_size', 20)
        self.contextual_input_size = model_config.get('contextual_input_size', 10)
        self.hidden_size = model_config['hidden_size']
        self.num_layers = model_config['num_layers']
        self.dropout = model_config['dropout']
        self.sequence_length = model_config['sequence_length']
        
        # Temporal stream
        self.temporal_stream = TemporalStream(
            input_size=self.temporal_input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        )
        
        # Contextual stream
        self.contextual_stream = ContextualStream(
            input_size=self.contextual_input_size,
            hidden_size=self.hidden_size,
            dropout=self.dropout
        )
        
        # Attention mechanism
        self.attention = RealtimeAttention(
            d_model=self.hidden_size,
            dropout=self.dropout
        )
        
        # Feature fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Price prediction head
        self.price_predictor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size // 2, 1)
        )
        
    def forward(self, temporal_input: torch.Tensor, contextual_input: torch.Tensor) -> dict:
        """Forward pass of the real-time model."""
        
        # Process temporal stream
        temporal_output = self.temporal_stream(temporal_input)
        
        # Process contextual stream
        contextual_output = self.contextual_stream(contextual_input)
        
        # Apply attention to temporal features
        attended_temporal, attention_weights = self.attention(temporal_output)
        
        # Combine temporal and contextual features
        combined_features = torch.cat([attended_temporal, contextual_output], dim=-1)
        
        # Apply fusion layer
        fused_features = self.fusion_layer(combined_features)
        
        # Generate price prediction
        price_prediction = self.price_predictor(fused_features)
        
        return {
            'price_prediction': price_prediction,
            'attention_weights': attention_weights,
            'fused_features': fused_features
        }

def setup_environment():
    """Setup the environment and load configuration."""
    print("🔧 Setting up environment...")
    
    try:
        with open('/home/srmist54/geopolitical_energy_forecasting/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print("✅ Configuration loaded")
    except Exception as e:
        print(f"⚠️  Could not load config: {e}")
        config = {
            'model': {
                'sequence_length': 30,
                'temporal_input_size': 20,
                'contextual_input_size': 10,
                'hidden_size': 64,
                'num_layers': 2,
                'dropout': 0.3,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 50
            }
        }
    
    return config

def calculate_technical_indicators(df):
    """Calculate comprehensive technical indicators."""
    df = df.copy()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = np.clip(df['rsi'], 0, 100)
    
    # MACD
    ema_12 = df['Close'].ewm(span=12).mean()
    ema_26 = df['Close'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['bb_position'] = np.clip(df['bb_position'], 0, 1)
    
    # Moving Averages
    df['ma_5'] = df['Close'].rolling(window=5).mean()
    df['ma_10'] = df['Close'].rolling(window=10).mean()
    df['ma_20'] = df['Close'].rolling(window=20).mean()
    df['ma_50'] = df['Close'].rolling(window=50).mean()
    
    # Volatility
    df['volatility'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
    df['volatility'] = np.clip(df['volatility'], 0, 2)
    
    # Volume indicators
    df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
    df['volume_ratio'] = np.clip(df['volume_ratio'], 0, 10)
    
    # Price momentum
    df['momentum_5'] = df['Close'].pct_change(5)
    df['momentum_10'] = df['Close'].pct_change(10)
    df['momentum_20'] = df['Close'].pct_change(20)
    
    # Price relative to moving averages
    df['price_vs_ma5'] = (df['Close'] - df['ma_5']) / df['ma_5']
    df['price_vs_ma20'] = (df['Close'] - df['ma_20']) / df['ma_20']
    df['price_vs_ma50'] = (df['Close'] - df['ma_50']) / df['ma_50']
    
    # Fill NaN values and ensure all values are finite
    df = df.fillna(0)
    df = df.replace([np.inf, -np.inf], 0)
    
    return df

def create_news_features(news_data, market_dates):
    """Create news sentiment features for market dates."""
    print("📰 Creating news sentiment features...")
    
    news_features = {}
    
    for symbol, news_df in news_data.items():
        if news_df.empty:
            continue
            
        # Convert publishedAt to date
        news_df['date'] = pd.to_datetime(news_df['publishedAt']).dt.date
        
        # Group by date and calculate daily sentiment
        daily_sentiment = news_df.groupby('date').agg({
            'sentiment_compound': 'mean',
            'sentiment_positive': 'mean',
            'sentiment_negative': 'mean',
            'sentiment_neutral': 'mean'
        }).reset_index()
        
        # Create features for each market date
        symbol_features = []
        for date in market_dates:
            date_obj = date.date() if hasattr(date, 'date') else date
            
            # Find news for this date or nearby dates
            date_news = daily_sentiment[daily_sentiment['date'] == date_obj]
            
            if not date_news.empty:
                features = date_news.iloc[0].to_dict()
                del features['date']
            else:
                # Use average sentiment if no news for this date
                features = {
                    'sentiment_compound': 0,
                    'sentiment_positive': 0.5,
                    'sentiment_negative': 0.5,
                    'sentiment_neutral': 0.5
                }
            
            symbol_features.append(features)
        
        news_features[symbol] = pd.DataFrame(symbol_features)
        print(f"  ✅ {symbol}: {len(symbol_features)} news sentiment features")
    
    return news_features

def create_sequences_for_lstm(market_data, news_features, config):
    """Create sequences for LSTM training with real-time data."""
    print("🔧 Creating LSTM sequences from REAL-TIME data...")
    
    sequence_length = config['model']['sequence_length']
    temporal_features = config['model']['temporal_input_size']
    contextual_features = config['model']['contextual_input_size']
    
    all_temporal_sequences = []
    all_contextual_sequences = []
    all_targets = []
    
    for symbol, df in market_data.items():
        print(f"  Processing {symbol} with {len(df)} real data points...")
        
        # Calculate technical indicators
        df_with_indicators = calculate_technical_indicators(df)
        
        # Get news features for this symbol
        symbol_news = news_features.get(symbol, pd.DataFrame())
        
        # Create sequences
        for i in range(sequence_length, len(df_with_indicators)):
            # Temporal sequence (last sequence_length days)
            temporal_seq = []
            for j in range(i - sequence_length, i):
                # Technical features for this day
                tech_features = [
                    df_with_indicators['rsi'].iloc[j],
                    df_with_indicators['macd'].iloc[j],
                    df_with_indicators['macd_signal'].iloc[j],
                    df_with_indicators['bb_width'].iloc[j],
                    df_with_indicators['bb_position'].iloc[j],
                    df_with_indicators['volatility'].iloc[j],
                    df_with_indicators['volume_ratio'].iloc[j],
                    df_with_indicators['momentum_5'].iloc[j],
                    df_with_indicators['momentum_20'].iloc[j],
                    df_with_indicators['price_vs_ma20'].iloc[j],
                    df_with_indicators['Close'].iloc[j],
                    df_with_indicators['Open'].iloc[j],
                    df_with_indicators['High'].iloc[j],
                    df_with_indicators['Low'].iloc[j],
                    df_with_indicators['Volume'].iloc[j],
                    df_with_indicators['ma_5'].iloc[j],
                    df_with_indicators['ma_10'].iloc[j],
                    df_with_indicators['ma_20'].iloc[j],
                    df_with_indicators['ma_50'].iloc[j],
                    df_with_indicators['price_vs_ma50'].iloc[j]
                ]
                temporal_seq.append(tech_features[:temporal_features])
            
            # Contextual features (current day)
            if not symbol_news.empty and i < len(symbol_news):
                news_row = symbol_news.iloc[i]
                contextual_features_list = [
                    news_row.get('sentiment_compound', 0),
                    news_row.get('sentiment_positive', 0.5),
                    news_row.get('sentiment_negative', 0.5),
                    news_row.get('sentiment_neutral', 0.5),
                    0,  # Additional contextual features
                    0,
                    0,
                    0,
                    0,
                    0
                ]
            else:
                contextual_features_list = [0] * contextual_features
            
            # Target (next day's price)
            target_price = df_with_indicators['Close'].iloc[i]
            
            all_temporal_sequences.append(temporal_seq)
            all_contextual_sequences.append(contextual_features_list)
            all_targets.append(target_price)
    
    print(f"✅ Created {len(all_temporal_sequences)} sequences from REAL-TIME data")
    return np.array(all_temporal_sequences), np.array(all_contextual_sequences), np.array(all_targets)

def train_realtime_model(model, train_loader, val_loader, config, device):
    """Train the real-time LSTM model."""
    print("🧠 Training Real-time Bi-LSTM model...")
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['model']['learning_rate'], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    epochs = config['model']['epochs']
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 10
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for temporal_data, contextual_data, targets in train_loader:
            temporal_data = temporal_data.to(device)
            contextual_data = contextual_data.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(temporal_data, contextual_data)
            loss = criterion(outputs['price_prediction'].squeeze(), targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = train_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for temporal_data, contextual_data, targets in val_loader:
                temporal_data = temporal_data.to(device)
                contextual_data = contextual_data.to(device)
                targets = targets.to(device)
                
                outputs = model(temporal_data, contextual_data)
                loss = criterion(outputs['price_prediction'].squeeze(), targets)
                
                val_loss += loss.item()
                num_val_batches += 1
        
        avg_val_loss = val_loss / num_val_batches
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), '/home/srmist54/geopolitical_energy_forecasting/models/best_realtime_lstm_model.pth')
        else:
            patience_counter += 1
        
        if epoch % 5 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
        
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss
    }

def evaluate_realtime_model(model, test_loader, device):
    """Evaluate the trained real-time model."""
    print("📊 Evaluating real-time model...")
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for temporal_data, contextual_data, targets in test_loader:
            temporal_data = temporal_data.to(device)
            contextual_data = contextual_data.to(device)
            targets = targets.to(device)
            
            outputs = model(temporal_data, contextual_data)
            predictions = outputs['price_prediction'].squeeze()
            
            pred_list = predictions.cpu().numpy()
            target_list = targets.cpu().numpy()
            
            if pred_list.ndim == 0:
                pred_list = [pred_list.item()]
            else:
                pred_list = pred_list.tolist()
                
            if target_list.ndim == 0:
                target_list = [target_list.item()]
            else:
                target_list = target_list.tolist()
            
            all_predictions.extend(pred_list)
            all_targets.extend(target_list)
    
    # Calculate metrics
    mse = mean_squared_error(all_targets, all_predictions)
    mae = mean_absolute_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets, all_predictions)
    mape = np.mean(np.abs((np.array(all_targets) - np.array(all_predictions)) / np.array(all_targets))) * 100
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'predictions': all_predictions,
        'targets': all_targets
    }

def create_realtime_visualizations(results, history):
    """Create visualization plots for real-time model."""
    print("📊 Creating real-time visualizations...")
    
    # Create results directory
    results_dir = Path('/home/srmist54/geopolitical_energy_forecasting/results')
    results_dir.mkdir(exist_ok=True)
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # 1. Training History
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(history['train_losses'], label='Training Loss')
        ax1.plot(history['val_losses'], label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.set_title('Real-time LSTM Training History')
        ax1.legend()
        ax1.grid(True)
        
        # Predictions vs Actual
        ax2.scatter(results['targets'], results['predictions'], alpha=0.5)
        ax2.plot([min(results['targets']), max(results['targets'])], 
                [min(results['targets']), max(results['targets'])], 'r--', lw=2)
        ax2.set_xlabel('Actual Prices')
        ax2.set_ylabel('Predicted Prices')
        ax2.set_title('Real-time Predictions vs Actual Prices')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(results_dir / 'realtime_training_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Price Prediction Time Series
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot last 200 predictions for visualization
        n_plot = min(200, len(results['targets']))
        x_vals = range(n_plot)
        
        ax.plot(x_vals, results['targets'][:n_plot], label='Actual Prices', alpha=0.7, linewidth=2)
        ax.plot(x_vals, results['predictions'][:n_plot], label='Predicted Prices', alpha=0.7, linewidth=2)
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Price')
        ax.set_title('Real-time LSTM Price Prediction Time Series (Last 200 samples)')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(results_dir / 'realtime_price_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Real-time visualizations saved")
        
    except ImportError:
        print("⚠️  Matplotlib not available, skipping visualizations")

def save_realtime_results(results, history, config):
    """Save comprehensive real-time results."""
    print("💾 Saving real-time results...")
    
    # Create results directory
    results_dir = Path('/home/srmist54/geopolitical_energy_forecasting/results')
    results_dir.mkdir(exist_ok=True)
    
    results_data = {
        'timestamp': datetime.now().isoformat(),
        'model_type': 'Real-time Bi-LSTM (Real Data + News)',
        'metrics': {
            'mse': results['mse'],
            'mae': results['mae'],
            'rmse': results['rmse'],
            'r2': results['r2'],
            'mape': results['mape']
        },
        'training_history': history,
        'config': config,
        'data_info': {
            'total_samples': len(results['targets']),
            'sequence_length': config['model']['sequence_length'],
            'temporal_features': config['model']['temporal_input_size'],
            'contextual_features': config['model']['contextual_input_size']
        }
    }
    
    with open(results_dir / 'realtime_lstm_results.json', 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    print("✅ Real-time results saved")

def main():
    """Main training pipeline for real-time LSTM price forecasting."""
    print("🚀 REAL-TIME Bi-LSTM Price Forecasting with News Sentiment")
    print("=" * 80)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        if torch.cuda.device_count() > 1:
            print(f"   Multi-GPU: {torch.cuda.device_count()} GPUs available")
    
    # Step 1: Setup
    config = setup_environment()
    
    # Step 2: Fetch REAL-TIME market data
    print("📊 Fetching REAL-TIME market data...")
    market_data = fetch_realtime_data()
    if not market_data:
        print("❌ No real-time market data available. Exiting.")
        return
    
    # Step 3: Fetch REAL-TIME news data
    print("📰 Fetching REAL-TIME news data...")
    news_data = fetch_robust_news()
    
    # Step 4: Create news features
    all_market_dates = set()
    for df in market_data.values():
        all_market_dates.update(df.index)
    market_dates = sorted(list(all_market_dates))
    
    news_features = create_news_features(news_data, market_dates)
    
    # Step 5: Create LSTM sequences
    temporal_sequences, contextual_sequences, targets = create_sequences_for_lstm(market_data, news_features, config)
    
    if len(temporal_sequences) == 0:
        print("❌ No sequences created. Exiting.")
        return
    
    # Step 6: Prepare data loaders
    print("📦 Preparing real-time data loaders...")
    
    # Normalize features
    temporal_scaler = StandardScaler()
    contextual_scaler = StandardScaler()
    target_scaler = StandardScaler()
    
    # Reshape for scaling
    temporal_flat = temporal_sequences.reshape(-1, temporal_sequences.shape[-1])
    temporal_scaled = temporal_scaler.fit_transform(temporal_flat)
    temporal_sequences = temporal_scaled.reshape(temporal_sequences.shape)
    
    contextual_sequences = contextual_scaler.fit_transform(contextual_sequences)
    targets = target_scaler.fit_transform(targets.reshape(-1, 1)).flatten()
    
    # Split data
    from sklearn.model_selection import train_test_split
    
    X_temp, X_test, X_context_temp, X_context_test, y_temp, y_test = train_test_split(
        temporal_sequences, contextual_sequences, targets, test_size=0.2, random_state=42
    )
    
    X_train, X_val, X_context_train, X_context_val, y_train, y_val = train_test_split(
        X_temp, X_context_temp, y_temp, test_size=0.2, random_state=42
    )
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(X_context_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(X_context_val),
        torch.FloatTensor(y_val)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(X_context_test),
        torch.FloatTensor(y_test)
    )
    
    # Create data loaders
    batch_size = config['model']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"✅ Data loaders created: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Step 7: Create model
    print("🏗️  Creating Real-time Bi-LSTM model...")
    model = RealtimeBiLSTM(config).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Step 8: Train model
    history = train_realtime_model(model, train_loader, val_loader, config, device)
    
    # Step 9: Load best model and evaluate
    model.load_state_dict(torch.load('/home/srmist54/geopolitical_energy_forecasting/models/best_realtime_lstm_model.pth'))
    results = evaluate_realtime_model(model, test_loader, device)
    
    # Step 10: Create visualizations
    create_realtime_visualizations(results, history)
    
    # Step 11: Save results
    save_realtime_results(results, history, config)
    
    # Step 12: Display final results
    print("\n🎉 REAL-TIME LSTM Price Forecasting Completed Successfully!")
    print("=" * 80)
    print(f"📊 Final Performance:")
    print(f"   RMSE: {results['rmse']:.6f}")
    print(f"   MAE: {results['mae']:.6f}")
    print(f"   R²: {results['r2']:.4f}")
    print(f"   MAPE: {results['mape']:.2f}%")
    print(f"   Total Samples: {len(results['targets'])}")
    print(f"   Sequence Length: {config['model']['sequence_length']}")
    
    print(f"\n📁 Output Files:")
    print(f"   - results/realtime_lstm_results.json: Complete results and metrics")
    print(f"   - models/best_realtime_lstm_model.pth: Trained model weights")
    print(f"   - results/realtime_training_results.png: Training history and predictions")
    print(f"   - results/realtime_price_predictions.png: Price prediction time series")
    print(f"   - data/: Real-time market and news data")
    
    print(f"\n✅ Pipeline Summary:")
    print(f"   1. ✅ REAL-TIME market data collection")
    print(f"   2. ✅ REAL-TIME news sentiment analysis")
    print(f"   3. ✅ Technical indicator calculation")
    print(f"   4. ✅ LSTM sequence creation")
    print(f"   5. ✅ Real-time Bi-LSTM training")
    print(f"   6. ✅ Price forecasting evaluation")
    print(f"   7. ✅ Visualization generation")
    print(f"   8. ✅ Results saving")

if __name__ == "__main__":
    main()
