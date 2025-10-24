#!/bin/bash
# Deployment script for NVIDIA DGX A100 server

echo "🚀 Deploying RippleNet-TFT to NVIDIA DGX A100 Server"
echo "=================================================="

# Set up environment
echo "📦 Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Set up API keys
echo "🔑 Setting up API keys..."
source api_keys.txt

# Test CUDA availability
echo "🔧 Testing CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')"

# Run data fetching
echo "📊 Fetching data..."
python data/data_fetcher.py

# Create training data
echo "🔄 Creating training dataset..."
python create_training_data.py

# Train model
echo "🤖 Training RippleNet-TFT model..."
python train.py --data data/merged.csv --device cuda

# Evaluate model
echo "📈 Evaluating model..."
python simple_evaluate.py

# Generate final summary
echo "📋 Generating final summary..."
python final_summary.py

echo "✅ Deployment completed successfully!"
echo "🎉 RippleNet-TFT is ready for production on DGX A100!"
