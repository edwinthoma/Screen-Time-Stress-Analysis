import os
from pathlib import Path


PROJECT_TITLE = "Screen Time, Sleep & Stress Analysis"

# Raw dataset path provided by user (absolute)
RAW_DATA_PATH = Path(r"C:\Users\edwin\Desktop\PYspiders\ML\ML_PROOJECT\data\Smartphone_Usage_Productivity_Dataset.csv")

# Default in-project dataset path
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_DATASET_PATH = DATA_DIR / "dataset.csv"

# Column configuration (may be auto-corrected if invalid)
TEXT_COLUMN_CONFIG = "ser_ID,Age,Gender,Occupation,Device_Type,Daily_Phone_Hours,Social_Media_Hours,Work_Productivity_Score,Sleep_Hours,,App_Usage_Count,Caffeine_Intake_Cups,Weekend_Screen_Time_Hours"
TARGET_COLUMN = "Stress_Level"

MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"


def ensure_project_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_dataset_path() -> Path:
    """
    Prefer in-project dataset; fall back to raw path if needed.
    """
    if DEFAULT_DATASET_PATH.exists():
        return DEFAULT_DATASET_PATH
    if RAW_DATA_PATH.exists():
        return RAW_DATA_PATH
    env_path = os.getenv("DATASET_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
    return DEFAULT_DATASET_PATH

