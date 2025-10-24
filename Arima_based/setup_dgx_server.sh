#!/bin/bash

# NVIDIA DGX A100 Server Setup Script for RippleNet-TFT
# This script prepares the server environment for optimal performance

echo "🚀 Setting up NVIDIA DGX A100 for RippleNet-TFT..."

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
export WANDB_PROJECT="ripplenet-tft"
export WANDB_ENTITY="your_entity"

# Create project directories
echo "📁 Creating project directories..."
mkdir -p /workspace/ripplenet-tft
cd /workspace/ripplenet-tft

mkdir -p checkpoints
mkdir -p plots
mkdir -p results
mkdir -p logs
mkdir -p data/raw
mkdir -p data/processed

# Install system dependencies
echo "📦 Installing system dependencies..."
apt-get update
apt-get install -y curl wget git build-essential
apt-get install -y python3-dev python3-pip
apt-get install -y libhdf5-dev libssl-dev libffi-dev

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

# Install additional dependencies
pip install requests beautifulsoup4
pip install jupyter ipykernel

# Check GPU availability and performance
echo "🖥️  Checking GPU setup..."
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
    print(f'  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB')
"

# Test CUDA performance
echo "⚡ Testing CUDA performance..."
python3 -c "
import torch
import time

# Test tensor operations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Create large tensors for performance test
x = torch.randn(10000, 10000, device=device)
y = torch.randn(10000, 10000, device=device)

start_time = time.time()
z = torch.matmul(x, y)
torch.cuda.synchronize()
end_time = time.time()

print(f'Matrix multiplication (10k x 10k): {end_time - start_time:.3f} seconds')
print(f'GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB')
print(f'GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB')
"

# Set up optimal CUDA settings
echo "⚙️  Configuring CUDA settings..."
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Create systemd service for continuous prediction
echo "🔧 Setting up systemd service..."
cat > /etc/systemd/system/ripplenet-tft.service << EOF
[Unit]
Description=RippleNet-TFT Real-time Predictor
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/workspace/ripplenet-tft
Environment=PATH=/workspace/ripplenet-tft/venv/bin
Environment=CUDA_VISIBLE_DEVICES=0,1,2,3
ExecStart=/workspace/ripplenet-tft/venv/bin/python realtime_predictor.py --continuous --interval 60
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable the service (but don't start yet)
systemctl daemon-reload
systemctl enable ripplenet-tft.service

echo "✅ NVIDIA DGX A100 setup completed!"
echo ""
echo "📋 Next steps:"
echo "1. Copy your project files to /workspace/ripplenet-tft/"
echo "2. Run: ./deploy_to_dgx.sh"
echo "3. Start real-time prediction: systemctl start ripplenet-tft"
echo "4. Monitor logs: journalctl -u ripplenet-tft -f"
echo ""
echo "🔧 Server configuration:"
echo "- GPUs: $(nvidia-smi --list-gpus | wc -l) available"
echo "- CUDA: $(nvcc --version | grep release)"
echo "- Python: $(python3 --version)"
echo "- PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
