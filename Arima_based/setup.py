"""
Setup script for RippleNet-TFT
Automated installation and configuration
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        logger.error("Python 3.10+ is required. Current version: {}.{}.{}".format(
            version.major, version.minor, version.micro))
        return False
    logger.info(f"Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"CUDA is available with {torch.cuda.device_count()} GPU(s)")
            for i in range(torch.cuda.device_count()):
                logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            logger.warning("CUDA is not available. Training will be slower on CPU.")
            return False
    except ImportError:
        logger.warning("PyTorch not installed yet. CUDA check will be done after installation.")
        return None

def install_requirements():
    """Install required packages"""
    logger.info("Installing requirements...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info("Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install requirements: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    logger.info("Creating project directories...")
    
    directories = [
        "data/raw",
        "data/processed", 
        "data/gdelt",
        "data/comtrade",
        "checkpoints",
        "results",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def create_env_file():
    """Create .env file if it doesn't exist"""
    env_file = Path(".env")
    env_example = Path("env_example.txt")
    
    if not env_file.exists() and env_example.exists():
        logger.info("Creating .env file from template...")
        with open(env_example, 'r') as f:
            content = f.read()
        
        with open(env_file, 'w') as f:
            f.write(content)
        
        logger.info("Created .env file. Please edit it with your API keys.")
    else:
        logger.info(".env file already exists")

def run_demo():
    """Run demo to test installation"""
    logger.info("Running demo to test installation...")
    
    try:
        subprocess.check_call([sys.executable, "demo.py"])
        logger.info("Demo completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Demo failed: {e}")
        return False

def print_next_steps():
    """Print next steps for the user"""
    logger.info("\n" + "="*60)
    logger.info("SETUP COMPLETED SUCCESSFULLY!")
    logger.info("="*60)
    logger.info("\nNext steps:")
    logger.info("1. Edit .env file with your API keys:")
    logger.info("   - NEWS_API_KEY (get from https://newsapi.org/)")
    logger.info("   - EIA_API_KEY (get from https://www.eia.gov/opendata/)")
    logger.info("\n2. Run the demo to test everything:")
    logger.info("   python demo.py")
    logger.info("\n3. Fetch real data (optional):")
    logger.info("   python data/data_fetcher.py")
    logger.info("\n4. Train the model:")
    logger.info("   python train.py --data data/merged.csv")
    logger.info("\n5. Evaluate the model:")
    logger.info("   python evaluate.py --checkpoint checkpoints/best_model.pt")
    logger.info("\nFor DGX A100 optimization:")
    logger.info("   python train.py --device cuda:0,1,2,3 --config config.yaml")
    logger.info("="*60)

def main():
    """Main setup function"""
    logger.info("Starting RippleNet-TFT setup...")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Create .env file
    create_env_file()
    
    # Install requirements
    if not install_requirements():
        logger.error("Setup failed during requirements installation")
        sys.exit(1)
    
    # Check CUDA after installation
    cuda_available = check_cuda()
    if cuda_available is True:
        logger.info("CUDA setup is optimal for DGX A100 training")
    elif cuda_available is False:
        logger.warning("CUDA not available - training will be slower")
    
    # Run demo
    if run_demo():
        print_next_steps()
    else:
        logger.error("Setup completed but demo failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
