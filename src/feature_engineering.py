"""
feature_engineering.py Feature Engineering for Health Insurance Cross Sell Prediction
"""

import numpy as np
import pandas as pd


# ==========================================================
# PREMIUM PER AGE
# ==========================================================
def premium_per_age(df: pd.DataFrame):

    df["Premium_Per_Age"] = df["Annual_Premium"] / df["Age"].replace(0, 1)
    return df


# ==========================================================
# CUSTOMER TENURE IN YEARS
# ==========================================================
def vintage_years(df: pd.DataFrame):
    df["Vintage_Years"] = (df["Vintage"] / 365).round(2)
    return df


# ==========================================================
# LOG TRANSFORM PREMIUM
# ==========================================================
def log_premium(df: pd.DataFrame):
    df["Log_Premium"] = np.log1p(df["Annual_Premium"])
    return df


# ==========================================================
# AGE FLAGS
# ==========================================================
def age_flags(df: pd.DataFrame):
    df["Young_Driver"] = (df["Age"] < 30).astype(int)
    df["Senior_Citizen"] = (df["Age"] >= 60).astype(int)
    return df


# ==========================================================
# HIGH PREMIUM FLAG
# Use a FIXED threshold obtained from training dataset.
# ==========================================================
HIGH_PREMIUM_THRESHOLD = 41000


def high_premium(df: pd.DataFrame):
    df["High_Premium"] = (df["Annual_Premium"] >= HIGH_PREMIUM_THRESHOLD).astype(int)
    return df


# ==========================================================
# VEHICLE DAMAGE + NOT PREVIOUSLY INSURED
# ==========================================================
def damage_not_insured(df: pd.DataFrame):
    df["Damage_Not_Insured"] = (
        (df["Vehicle_Damage"] == 1) & (df["Previously_Insured"] == 0)
    ).astype(int)
    return df


# ==========================================================
# AGE × VEHICLE AGE
# ==========================================================
def age_vehicle_interaction(df: pd.DataFrame):
    df["Age_VehicleAge"] = df["Age"] * df["Vehicle_Age"]
    return df


# ==========================================================
# PREMIUM × VINTAGE
# ==========================================================
def premium_vintage(df: pd.DataFrame):
    df["Premium_Vintage"] = df["Annual_Premium"] * df["Vintage"]
    return df


# ==========================================================
# LICENSE × PREVIOUSLY INSURED
# ==========================================================
def insurance_interaction(df: pd.DataFrame):
    df["License_Insured"] = df["Driving_License"] * df["Previously_Insured"]
    return df


# ==========================================================
# AGE × PREMIUM
# ==========================================================
def age_premium(df: pd.DataFrame):
    df["Age_Premium"] = df["Age"] * df["Annual_Premium"]
    return df


# ==========================================================
# COMPLETE FEATURE ENGINEERING
# ==========================================================
def feature_engineering(df: pd.DataFrame):
    df = premium_per_age(df)
    df = vintage_years(df)
    df = log_premium(df)
    df = age_flags(df)
    df = high_premium(df)
    df = damage_not_insured(df)
    df = age_vehicle_interaction(df)
    df = premium_vintage(df)
    df = insurance_interaction(df)
    df = age_premium(df)
    return df
