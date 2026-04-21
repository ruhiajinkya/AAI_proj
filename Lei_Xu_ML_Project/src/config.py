from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = SRC_DIR / "artifacts"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
PLOTS_DIR = ARTIFACTS_DIR / "plots"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2019-01-01"
END_DATE = "2025-12-31"

DAILY_FILE = RAW_DIR / "daily_sp100_2019_2025.csv"
MONTHLY_FILE = PROCESSED_DIR / "monthly_sp100_2019_2025.csv"
MONTHLY_FEATURE_FILE = PROCESSED_DIR / "monthly_features_sp100_2019_2025.csv"
ML_FILE = PROCESSED_DIR / "sp100_monthly_ml_dataset.csv"
LEGACY_ML_FILE = BASE_DIR / "sp100_monthly_ml_dataset.csv"

if not ML_FILE.exists() and LEGACY_ML_FILE.exists():
    ML_FILE = LEGACY_ML_FILE
