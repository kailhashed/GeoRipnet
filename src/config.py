"""
Central configuration for GeoRipNet.
All paths, constants, and hyperparameters live here.
"""
import torch
from pathlib import Path

ROOT = Path(__file__).parent.parent

# ── Directories ───────────────────────────────────────────────────────────────
DATA_DIR        = ROOT / "data"
PRICE_DIR       = DATA_DIR / "price"
COMTRADE_DIR    = DATA_DIR / "uncomtrade"
GDELT_DIR       = DATA_DIR / "gdelt_data"
RAW_CACHE_DIR   = GDELT_DIR / "raw_cache"
CHECKPOINT_DIR  = ROOT / "checkpoints"
RESULTS_DIR     = ROOT / "results"

# Pre-split data directories
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR   = DATA_DIR / "val"
TEST_DIR  = DATA_DIR / "test"

for d in [CHECKPOINT_DIR, RESULTS_DIR, RAW_CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Price files (normalized, YYYY-MM-DD ascending) ───────────────────────────
PRICE_FILES = {
    "WTI":    PRICE_DIR / "wti_daily.csv",
    "Brent":  PRICE_DIR / "brent_daily.csv",
    "OPEC":   PRICE_DIR / "opec_daily.csv",
    "ESPO":   PRICE_DIR / "urals_daily.csv",
    "Indian": PRICE_DIR / "indian_basket_daily.csv",
}

# Node order must match price files and Comtrade adjacency matrix
NODES = ["WTI", "Brent", "OPEC", "ESPO", "Indian"]
NODE_COUNTRIES = {
    "WTI":    ["USA"],
    "Brent":  ["GBR", "NOR"],
    "OPEC":   ["SAU"],
    "ESPO":   ["RUS"],
    "Indian": ["IND"],
}
N_NODES = 5

# ── Comtrade ──────────────────────────────────────────────────────────────────
ADJACENCY_FILE = COMTRADE_DIR / "adjacency_monthly.parquet"

# ── GDELT ─────────────────────────────────────────────────────────────────────
GDELT_TENSOR_FILE = GDELT_DIR / "daily_gdelt_tensor.parquet"
GDELT_CAMEO_CODES = ["13", "17", "18", "19", "20"]
GDELT_CHANNELS    = ["GoldsteinScale", "AvgTone", "NumMentions"]
N_CHANNELS        = 3

# ── Dataset split dates ──────────────────────────────────────────────────────
TRAIN_START = "2010-01-01"
TRAIN_END   = "2019-12-31"
VAL_START   = "2020-01-01"
VAL_END     = "2021-12-31"
TEST_START  = "2022-01-01"

# ── Model hyperparameters ─────────────────────────────────────────────────────
LOOKBACK_WINDOW = 30        # k — updated to 30 for better structural anchoring
HORIZONS        = [1, 7, 14, 30] 
D_MODEL         = 128       # increased capacity for multi-horizon
N_HEADS_GAT     = 4
N_TRANSFORMER_LAYERS = 4    # deeper temporal reasoning
DROPOUT         = 0.1

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE      = 64
LR              = 1e-3
WEIGHT_DECAY    = 1e-5
EPOCHS          = 200
PATIENCE        = 10        # early stopping patience

# ── Device ────────────────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()
