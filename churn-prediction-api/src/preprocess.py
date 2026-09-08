import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

COLUMNS_TO_DROP = [
    "CustomerID", "Count", "Country", "State", "City",
    "Zip Code", "Lat Long", "Latitude", "Longitude",
    "Churn Label", "Churn Score", "CLTV", "Churn Reason"
]

TARGET = "Churn Value"

BINARY_COLUMNS = [
    "Senior Citizen", "Partner", "Dependents",
    "Phone Service", "Paperless Billing"
]

CATEGORICAL_COLUMNS = [
    "Gender", "Multiple Lines", "Internet Service", "Online Security",
    "Online Backup", "Device Protection", "Tech Support",
    "Streaming TV", "Streaming Movies", "Contract", "Payment Method"
]

NUMERICAL_COLUMNS = ["Tenure Months", "Monthly Charges", "Total Charges"]

# Maps API snake_case field names to dataset column names
API_TO_DATASET_COLUMNS = {
    "gender": "Gender",
    "senior_citizen": "Senior Citizen",
    "partner": "Partner",
    "dependents": "Dependents",
    "tenure_months": "Tenure Months",
    "phone_service": "Phone Service",
    "multiple_lines": "Multiple Lines",
    "internet_service": "Internet Service",
    "online_security": "Online Security",
    "online_backup": "Online Backup",
    "device_protection": "Device Protection",
    "tech_support": "Tech Support",
    "streaming_tv": "Streaming TV",
    "streaming_movies": "Streaming Movies",
    "contract": "Contract",
    "paperless_billing": "Paperless Billing",
    "payment_method": "Payment Method",
    "monthly_charges": "Monthly Charges",
    "total_charges": "Total Charges",
}


def load_data(path: str) -> pd.DataFrame:
    if path.endswith(".xlsx"):
        return pd.read_excel(path, engine="openpyxl")
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=COLUMNS_TO_DROP)

    # Total Charges may contain blank strings or already be numeric
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
    df = df.dropna(subset=["Total Charges"])
    df = df.reset_index(drop=True)

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    binary_cols_present = [col for col in BINARY_COLUMNS if col in df.columns]
    for col in binary_cols_present:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=True)

    return df


def split_data(df: pd.DataFrame):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test):
    scaler = StandardScaler()

    X_train[NUMERICAL_COLUMNS] = scaler.fit_transform(X_train[NUMERICAL_COLUMNS])
    X_val[NUMERICAL_COLUMNS] = scaler.transform(X_val[NUMERICAL_COLUMNS])
    X_test[NUMERICAL_COLUMNS] = scaler.transform(X_test[NUMERICAL_COLUMNS])

    return X_train, X_val, X_test, scaler


def save_feature_names(feature_names: list, path: str = "models/feature_names.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(feature_names, f)


def load_feature_names(path: str = "models/feature_names.json") -> list:
    with open(path, "r") as f:
        return json.load(f)


def preprocess_single(customer_dict: dict, scaler: StandardScaler, feature_names: list) -> np.ndarray:
    renamed = {API_TO_DATASET_COLUMNS[k]: v for k, v in customer_dict.items()}
    df = pd.DataFrame([renamed])

    binary_cols_present = [col for col in BINARY_COLUMNS if col in df.columns]
    for col in binary_cols_present:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=True)

    # Align to training feature set - fill any missing one-hot columns with 0
    df = df.reindex(columns=feature_names, fill_value=0)

    df[NUMERICAL_COLUMNS] = scaler.transform(df[NUMERICAL_COLUMNS])

    return df.values
