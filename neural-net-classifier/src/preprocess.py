import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TARGET = "Loan_Status"

COLUMNS_TO_DROP = ["Loan_ID"]

NUMERICAL_COLUMNS = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]

CATEGORICAL_COLUMNS = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]

BINARY_COLUMNS = ["Credit_History"]

FEATURE_NAMES_PATH = "models/feature_names.json"


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=COLUMNS_TO_DROP, errors="ignore")
    df[TARGET] = df[TARGET].map({"Y": 1, "N": 0})
    for col in NUMERICAL_COLUMNS:
        df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].fillna(df[col].mode()[0])
    df["Credit_History"] = df["Credit_History"].fillna(df["Credit_History"].mode()[0])
    return df.reset_index(drop=True)


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
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
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()
    X_train[NUMERICAL_COLUMNS] = scaler.fit_transform(X_train[NUMERICAL_COLUMNS])
    X_val[NUMERICAL_COLUMNS] = scaler.transform(X_val[NUMERICAL_COLUMNS])
    X_test[NUMERICAL_COLUMNS] = scaler.transform(X_test[NUMERICAL_COLUMNS])
    return X_train, X_val, X_test, scaler


def save_feature_names(feature_names: list, path: str = FEATURE_NAMES_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(feature_names, f)


def load_feature_names(path: str = FEATURE_NAMES_PATH) -> list:
    with open(path) as f:
        return json.load(f)


def preprocess_single(data: dict, scaler: StandardScaler, feature_names: list) -> np.ndarray:
    df = pd.DataFrame([data])
    df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=True)
    df = df.reindex(columns=feature_names, fill_value=0)
    df[NUMERICAL_COLUMNS] = scaler.transform(df[NUMERICAL_COLUMNS])
    return df.values.astype(np.float32)
