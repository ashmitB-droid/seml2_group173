from pathlib import Path

# ==========================================================
# PROJECT ROOT
# ==========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ==========================================================
# DATA PATHS
# ==========================================================
DATA_DIR = PROJECT_ROOT / "dataset"
TEST_DATA_PATH = DATA_DIR / "test.csv"

# ==========================================================
# API ENDPOINT SETTINGS
# ==========================================================
BASE_URL = "https://health-insurance-cross-sell-prediction.onrender.com"
METRICS_URL = BASE_URL + "/metrics"
SINGLE_PREDICTION_URL = BASE_URL + "/predict"
BATCH_PREDICTION_URL = BASE_URL + "/batch-predict"