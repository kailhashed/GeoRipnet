"""
Simple test script for RippleNet-TFT setup
Tests core functionality without heavy dependencies
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_api_keys():
    """Test API key configuration"""
    logger.info("Testing API keys...")
    
    keys = {
        'NEWS_API_KEY': os.getenv('NEWS_API_KEY'),
        'EIA_API_KEY': os.getenv('EIA_API_KEY'),
        'WANDB_API_KEY': os.getenv('WANDB_API_KEY')
    }
    
    for key, value in keys.items():
        if value:
            logger.info(f"✅ {key}: Configured")
        else:
            logger.warning(f"❌ {key}: Missing")
    
    return all(keys.values())

def test_config():
    """Test configuration loading"""
    logger.info("Testing configuration...")
    
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info("✅ Configuration loaded successfully")
        logger.info(f"   Model hidden size: {config['model']['tft']['hidden_size']}")
        logger.info(f"   Training batch size: {config['training']['batch_size']}")
        logger.info(f"   Lookback window: {config['training']['lookback_window']}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Configuration error: {e}")
        return False

def test_data_creation():
    """Test synthetic data creation"""
    logger.info("Testing data creation...")
    
    try:
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        data = pd.DataFrame({
            'date': dates,
            'CL=F_close': 100 + np.cumsum(np.random.randn(100) * 0.02),
            'NG=F_close': 50 + np.cumsum(np.random.randn(100) * 0.03),
            'epu': np.random.normal(100, 20, 100),
            'gpr': np.random.normal(50, 15, 100)
        })
        
        logger.info(f"✅ Sample data created: {data.shape}")
        logger.info(f"   Date range: {data['date'].min()} to {data['date'].max()}")
        logger.info(f"   Price range: {data['CL=F_close'].min():.2f} to {data['CL=F_close'].max():.2f}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Data creation error: {e}")
        return False

def test_model_creation():
    """Test model creation (simplified)"""
    logger.info("Testing model creation...")
    
    try:
        import torch
        import torch.nn as nn
        
        # Simple test model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)
            
            def forward(self, x):
                return self.linear(x)
        
        model = SimpleModel()
        logger.info(f"✅ Model created: {sum(p.numel() for p in model.parameters())} parameters")
        
        # Test forward pass
        x = torch.randn(32, 10)
        y = model(x)
        logger.info(f"✅ Forward pass successful: {x.shape} -> {y.shape}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Model creation error: {e}")
        return False

def test_news_api():
    """Test NewsAPI connection"""
    logger.info("Testing NewsAPI connection...")
    
    try:
        import requests
        
        api_key = os.getenv('NEWS_API_KEY')
        if not api_key:
            logger.warning("❌ NewsAPI key not set")
            return False
        
        url = f'https://newsapi.org/v2/everything?q=energy&apiKey={api_key}&pageSize=1'
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ NewsAPI connection successful")
            logger.info(f"   Total articles: {data.get('totalResults', 'Unknown')}")
            return True
        else:
            logger.error(f"❌ NewsAPI error: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ NewsAPI error: {e}")
        return False

def test_eia_api():
    """Test EIA API connection"""
    logger.info("Testing EIA API connection...")
    
    try:
        import requests
        
        api_key = os.getenv('EIA_API_KEY')
        if not api_key:
            logger.warning("❌ EIA API key not set")
            return False
        
        url = f'https://api.eia.gov/v2/seriesid/PET.WCRFPUS2.W?api_key={api_key}'
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            logger.info(f"✅ EIA API connection successful")
            return True
        else:
            logger.error(f"❌ EIA API error: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ EIA API error: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🧪 Running RippleNet-TFT Setup Tests")
    logger.info("=" * 50)
    
    tests = [
        ("API Keys", test_api_keys),
        ("Configuration", test_config),
        ("Data Creation", test_data_creation),
        ("Model Creation", test_model_creation),
        ("NewsAPI", test_news_api),
        ("EIA API", test_eia_api)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n🔍 Testing {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    logger.info(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.info("\n🎉 ALL TESTS PASSED!")
        logger.info("🚀 RippleNet-TFT is ready for DGX A100 training!")
        logger.info("\nNext steps:")
        logger.info("1. Run: python train.py --data data/merged.csv")
        logger.info("2. Run: python evaluate.py --checkpoint checkpoints/best_model.pt")
    else:
        logger.warning(f"\n⚠️  {len(results) - passed} tests failed")
        logger.info("Please check the error messages above")

if __name__ == "__main__":
    main()
