#!/bin/bash

# RippleNet-TFT Deployment Script for NVIDIA DGX A100
# This script sets up the environment and runs the model on the DGX server

echo "🚀 Deploying RippleNet-TFT to NVIDIA DGX A100 Server..."

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export EIA_API_KEY="yCimuv5dVdNf0WCafxh8TpV3AY23fwilz2JfDKJe"
export WANDB_API_KEY="cc761dd9562d7bbfb399c5788f2109d6bf1ea18f"
export WANDB_PROJECT="ripplenet-tft"
export WANDB_ENTITY="your_entity"

# Create directories
mkdir -p checkpoints
mkdir -p plots
mkdir -p results
mkdir -p logs

# Install dependencies (if not already installed)
echo "📦 Installing dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install transformers
pip install yfinance
pip install pandas numpy scikit-learn matplotlib seaborn
pip install statsmodels
pip install vaderSentiment
pip install wandb
pip install tqdm

# Set up CUDA environment
echo "🔧 Setting up CUDA environment..."
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Check GPU availability
echo "🖥️  Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

# Create training data with real data
echo "📊 Creating training dataset with real data..."
python create_training_data.py

# Train the model with optimized hyperparameters
echo "🏋️  Training RippleNet-TFT model..."
python train.py --data data/merged.csv --device cuda --epochs 200 --batch_size 64

# Evaluate the model
echo "📈 Evaluating model performance..."
python simple_evaluate.py

# Generate plots
echo "📊 Generating evaluation plots..."
python generate_plots.py

# Run real-time prediction setup
echo "⚡ Setting up real-time prediction..."
python realtime_data_fetcher.py

echo "✅ Deployment completed successfully!"
echo "📁 Results saved in:"
echo "   - checkpoints/ (model checkpoints)"
echo "   - plots/ (evaluation plots)"
echo "   - results/ (evaluation results)"
echo "   - logs/ (training logs)"