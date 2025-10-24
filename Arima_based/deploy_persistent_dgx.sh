#!/bin/bash

# RippleNet-TFT Persistent Deployment Script for NVIDIA DGX A100
# This script sets up persistent training that continues even when you close your computer

echo "🚀 Setting up RippleNet-TFT for persistent training on DGX A100..."

# Set environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Set PyTorch environment variables
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"
export FORCE_CUDA="1"
export CUDA_LAUNCH_BLOCKING="0"

# Set API keys
export EIA_API_KEY="yCimuv5dVdNf0WCafxh8TpV3AY23fwilz2JfDKJe"
export WANDB_API_KEY="cc761dd9562d7bbfb399c5788f2109d6bf1ea18f"
export WANDB_PROJECT="ripplenet-tft-persistent"
export WANDB_ENTITY="your_entity"

# Create project directories
echo "📁 Creating project directories..."
mkdir -p /workspace/ripplenet-tft-persistent
cd /workspace/ripplenet-tft-persistent

mkdir -p checkpoints
mkdir -p plots
mkdir -p results
mkdir -p logs
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models
mkdir -p monitoring

# Create Python virtual environment
echo "🐍 Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip and install core dependencies
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
echo "🔥 Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric
echo "🔗 Installing PyTorch Geometric..."
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# Install ML and data science packages
echo "📊 Installing ML packages..."
pip install pandas numpy scikit-learn matplotlib seaborn plotly
pip install scipy statsmodels
pip install transformers huggingface-hub
pip install yfinance
pip install vaderSentiment
pip install wandb
pip install tqdm
pip install pyyaml
pip install requests beautifulsoup4
pip install jupyter ipykernel

# Install monitoring packages
pip install prometheus-client psutil GPUtil

echo "✅ Environment setup completed!"
echo "📋 Next steps:"
echo "1. Copy your project files to /workspace/ripplenet-tft-persistent/"
echo "2. Run: ./start_persistent_training.sh"
echo "3. Monitor: tail -f logs/training.log"
