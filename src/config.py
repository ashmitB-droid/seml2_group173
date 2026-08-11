"""
config.py

Central configuration file for the
Health Insurance Cross Sell Prediction project.
"""

from pathlib import Path

# ==========================================================
# PROJECT ROOT
# ==========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ==========================================================
# DATA PATHS
# ==========================================================
DATA_DIR = PROJECT_ROOT / "dataset"
TRAIN_DATA_PATH = DATA_DIR / "train.csv"

# ==========================================================
# MODEL DIRECTORY
# ==========================================================
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ==========================================================
# MODEL FILES
# ==========================================================

MODEL_PATH = MODEL_DIR / "xgb_model.pkl"
ENCODER_PATH = MODEL_DIR / "encoder.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"
SHAP_EXPLAINER_PATH = MODEL_DIR / "shap_explainer.pkl"

# ==========================================================
# LOG DIRECTORY
# ==========================================================
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
TRAIN_LOG = LOG_DIR / "training.log"
PREDICTION_LOG = LOG_DIR / "prediction.log"
API_LOG = LOG_DIR / "api.log"

# ==========================================================
# TARGET COLUMN
# ==========================================================
TARGET = "Response"

# ==========================================================
# RANDOM STATE
# ==========================================================
RANDOM_STATE = 42

# ==========================================================
# TRAIN TEST SPLIT
#
# ==========================================================
TEST_SIZE = 0.10

# ==========================================================
# CROSS VALIDATION
# ==========================================================
N_SPLITS = 5

# ==========================================================
# XGBOOST DEFAULT PARAMETERS
# ==========================================================
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# ==========================================================
# OPTUNA SETTINGS
# ==========================================================
OPTUNA_TRIALS = 50

# ==========================================================
# SHAP SETTINGS
# ==========================================================
BACKGROUND_SAMPLES = 1000

# ==========================================================
# API SETTINGS
# ==========================================================
API_TITLE = "Health Insurance Cross Sell API"
API_VERSION = "1.0.0"
API_DESCRIPTION = "Predict whether a customer is likely " "to purchase vehicle insurance."

# ==========================================================
# STREAMLIT SETTINGS
# ==========================================================
PAGE_TITLE = "Health Insurance Cross Sell"
PAGE_ICON = "🚗"
LAYOUT = "wide"

# ==========================================================
# FEATURE LIST
# ==========================================================
FEATURE_COLUMNS = [
    "Gender",
    "Age",
    "Driving_License",
    "Region_Code",
    "Previously_Insured",
    "Vehicle_Age",
    "Vehicle_Damage",
    "Annual_Premium",
    "Policy_Sales_Channel",
    "Vintage",
    "Age_Group",
    "Premium_Per_Age",
    "Vintage_Years",
    "High_Premium",
    "Young_Driver",
    "Senior_Citizen",
    "Damage_Not_Insured",
    "Premium_Category",
    "Age_VehicleAge",
    "Premium_Vintage",
    "Log_Premium",
]

# ==========================================================
# DECISION THRESHOLD
# ==========================================================
# The probability above which a customer is labelled "Interested".
#
# This is a BUSINESS decision, not a modelling default. The campaign is
# cheap per contact but a missed interested customer is lost revenue, so we
# deliberately trade precision for recall and sit below 0.50.
#
# It lives here because training and inference must use the SAME value.
# Previously train.py evaluated at 0.40 while predict.py used
# model.predict(), which hard-codes 0.50 -- so the reported recall (0.8827)
# described a decision rule the deployed API never applied (0.7874).
DECISION_THRESHOLD = 0.40

# ==========================================================
# RAW DATA SCHEMA CONTRACT
# ==========================================================
# Declarative description of a valid raw record, used by both the training
# pipeline and the inference path (Objective 2, Req. 8b).
REQUIRED_COLUMNS = [
    "Gender",
    "Age",
    "Driving_License",
    "Region_Code",
    "Previously_Insured",
    "Vehicle_Age",
    "Vehicle_Damage",
    "Annual_Premium",
    "Policy_Sales_Channel",
    "Vintage",
]

NUMERIC_COLUMNS = [
    "Age",
    "Driving_License",
    "Region_Code",
    "Previously_Insured",
    "Annual_Premium",
    "Policy_Sales_Channel",
    "Vintage",
]

ALLOWED_CATEGORIES = {
    "Gender": ["Male", "Female"],
    "Vehicle_Age": ["< 1 Year", "1-2 Year", "> 2 Years"],
    "Vehicle_Damage": ["Yes", "No"],
}

# Maximum tolerated fraction of missing values in any single column.
MAX_MISSING_FRACTION = 0.05

# PSI drift bands (population stability index).
PSI_MODERATE_DRIFT = 0.10
PSI_SIGNIFICANT_DRIFT = 0.25

# ==========================================================
# AGE GROUP BINS
# ==========================================================
# NOTE: bins are used with include_lowest=True. Without it pd.cut excludes
# the left edge, so Age == 18 fell outside every bin, became NaN, and the
# OrdinalEncoder raised "Found unknown categories [nan]" -- for an age the
# API schema explicitly accepts (Age >= 18).
AGE_BINS = [18, 25, 35, 45, 55, 65, 100]
AGE_LABELS = ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
