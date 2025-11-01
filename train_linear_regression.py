
# SOFE3370 Final Project – Linear Regression for Battery SOH


import os
import re
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# File paths 

RAW_FILE = "PulseBat Dataset (1).xlsx"
SAVE_DIR = "results"
os.makedirs(SAVE_DIR, exist_ok=True)


# Data Preprocessing & Aggregation 


def find_voltage_columns(df):
    """Identify columns named U1–U21 in the dataset."""
    pattern = re.compile(r"u(\d{1,2})", re.IGNORECASE)
    voltage_cols = [c for c in df.columns if pattern.fullmatch(c.strip())]
    if len(voltage_cols) < 10:
        raise ValueError("Not enough U1–U21 columns found!")
    return sorted(voltage_cols, key=lambda c: int(re.findall(r"\d+", c)[0]))

def create_pack_soh(df, voltage_cols):
    """Compute pack SOH by normalizing U1–U21 and averaging them."""
    X = df[voltage_cols].astype(float)
    if X.median().median() > 1.0:  # treat as voltages
        row_min = X.min(axis=1)
        row_max = X.max(axis=1)
        diff = (row_max - row_min).replace(0, np.nan)
        norm = (X.sub(row_min, axis=0)).div(diff, axis=0)
        soh = norm.mean(axis=1)
    else:
        soh = X.mean(axis=1)
    return soh.clip(0, 1)

def preprocess_dataset(df):
    """Prepare cleaned data and ensure SOH target is available."""
    volt_cols = find_voltage_columns(df)
    target = None
    for c in df.columns:
        if str(c).lower() in ["soh", "pack_soh", "target", "battery_soh"]:
            target = c
            break

    if target is None:
        print("→ No SOH column found, computing Pack SOH automatically.")
        df["Pack_SOH"] = create_pack_soh(df, volt_cols)
        target = "Pack_SOH"

    data = df[volt_cols + [target]].copy()
    data.rename(columns={target: "SOH"}, inplace=True)
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    X = data[volt_cols].astype(float)
    y = data["SOH"].astype(float)

    clean_file = os.path.join(SAVE_DIR, "preprocessed_dataset.csv")
    data.to_csv(clean_file, index=False)
    print(f"Data cleaned → {len(data)} valid samples saved to '{clean_file}'")
    return X, y, volt_cols


# Model Training & Evaluation


def train_regression(X, y):
    """Train and evaluate Linear Regression model for SOH prediction."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Save trained model for chatbot integration
    model_path = os.path.join(SAVE_DIR, "linear_regression_model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Evaluate model
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Save metrics
    with open(os.path.join(SAVE_DIR, "model_metrics.txt"), "w") as f:
        f.write(f"R^2: {r2:.4f}\nMSE: {mse:.6f}\nMAE: {mae:.6f}\n")

    # Create and save plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel("Actual SOH")
    plt.ylabel("Predicted SOH")
    plt.title("Battery Pack SOH: Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "soh_prediction_plot.png"))
    plt.close()

    print("\nModel training completed successfully!")
    print(f"R^2 = {r2:.4f} | MSE = {mse:.6f} | MAE = {mae:.6f}")
    print("Results saved in folder: 'results/'")


# EXECUTION

if __name__ == "__main__":
    df = pd.read_excel(RAW_FILE)
    X, y, u_columns = preprocess_dataset(df)
    train_regression(X, y)
