from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

EXPERIMENTS_DIR = ROOT_DIR / "experiments"
CONFIG_DIR = EXPERIMENTS_DIR / "configs"
MODELS_DIR = EXPERIMENTS_DIR / "models"
