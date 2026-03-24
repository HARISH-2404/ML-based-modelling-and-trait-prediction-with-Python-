# ==========================================================
# COMPLETE ML PIPELINE FOR TRAIT PREDICTION
# ==========================================================

import os
import pandas as pd
import numpy as np
import logging
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ==========================================================
# STEP 1: CREATE FOLDERS
# ==========================================================

def create_folders():
    folders = [
        "data/raw",
        "data/processed",
        "data/external",
        "results/metrics",
        "results/plots",
        "results/models"
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)


# ==========================================================
# STEP 2: GENERATE DATA
# ==========================================================

def generate_data(n=100):
    np.random.seed(42)

    df = pd.DataFrame({
        "Genotype": [f"G{i}" for i in range(1, n+1)],
        "PlantHeight": np.random.normal(180, 15, n),
        "Sympodia": np.random.normal(16, 3, n),
        "Monopodia": np.random.randint(1, 3, n),
        "Bolls": np.random.normal(30, 5, n),
        "Yield": np.random.normal(220, 30, n),
        "GOT": np.random.normal(35, 2, n),
    })

    df["LintYield"] = (df["Yield"] * df["GOT"]) / 100
    df["SeedYield"] = df["Yield"] - df["LintYield"]

    df.to_csv("data/raw/data.csv", index=False)
    return df


# ==========================================================
# STEP 3: PREPROCESS
# ==========================================================

def preprocess(df):

    df = df.drop_duplicates()
    df = df.fillna(df.mean(numeric_only=True))

    X = df.drop(columns=["Genotype", "Yield", "LintYield", "SeedYield"])
    y = df[["Yield", "LintYield", "SeedYield"]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, "results/models/scaler.pkl")

    return X_scaled, y


# ==========================================================
# STEP 4: TRAIN MODEL
# ==========================================================

def train_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=200))
    model.fit(X_train, y_train)

    return model, X_test, y_test


# ==========================================================
# STEP 5: EVALUATION
# ==========================================================

def evaluate(model, X_test, y_test):

    y_pred = model.predict(X_test)

    results = {}

    for i, col in enumerate(y_test.columns):

        r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i]))
        mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])

        results[col] = {"R2": r2, "RMSE": rmse, "MAE": mae}

    df_metrics = pd.DataFrame(results).T
    df_metrics.to_csv("results/metrics/model_performance.csv")

    return y_pred, df_metrics


# ==========================================================
# STEP 6: PLOTS
# ==========================================================

def generate_plots(df, y_test, y_pred):

    # Correlation heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True)
    plt.savefig("results/plots/correlation_heatmap.png")
    plt.close()

    # Prediction plots
    for i, col in enumerate(y_test.columns):
        plt.figure()
        plt.scatter(y_test.iloc[:, i], y_pred[:, i])
        plt.xlabel("Observed")
        plt.ylabel("Predicted")
        plt.title(col)

        plt.savefig(f"results/plots/{col}_prediction.png")
        plt.close()


# ==========================================================
# STEP 7: SAVE MODEL
# ==========================================================

def save_model(model):
    joblib.dump(model, "results/models/model.pkl")


# ==========================================================
# MAIN PIPELINE
# ==========================================================

def run_pipeline():

    logging.info("Starting pipeline...")

    create_folders()

    df = generate_data()

    X, y = preprocess(df)

    model, X_test, y_test = train_model(X, y)

    y_pred, metrics = evaluate(model, X_test, y_test)

    generate_plots(df, y_test, y_pred)

    save_model(model)

    logging.info("Pipeline completed successfully 🚀")
    print(metrics)


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    run_pipeline()
