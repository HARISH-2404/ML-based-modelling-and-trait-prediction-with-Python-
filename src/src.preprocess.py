# =========================================
# ADVANCED DATA PREPROCESSING MODULE
# =========================================

import pandas as pd
import numpy as np
import os
import logging
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO)

class DataPreprocessor:
    def __init__(self,
                 target_cols,
                 scaling_method="standard",
                 apply_pca=False,
                 n_components=5):

        self.target_cols = target_cols
        self.scaling_method = scaling_method
        self.apply_pca = apply_pca
        self.n_components = n_components

        self.scaler = StandardScaler() if scaling_method == "standard" else MinMaxScaler()
        self.imputer = SimpleImputer(strategy="mean")
        self.pca = PCA(n_components=n_components) if apply_pca else None

    def load_data(self, file_path):
        logging.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        logging.info(f"Shape: {df.shape}")
        return df

    def explore_data(self, df):
        logging.info("Exploring dataset...")
        print(df.head())
        print("\nMissing values:\n", df.isnull().sum())
        print("\nSummary stats:\n", df.describe())

    def split_features_targets(self, df):
        X = df.drop(columns=self.target_cols)
        y = df[self.target_cols]
        return X, y

    def handle_missing(self, X):
        logging.info("Handling missing values...")
        X_imputed = self.imputer.fit_transform(X)
        return pd.DataFrame(X_imputed, columns=X.columns)

    def scale_features(self, X):
        logging.info("Scaling features...")
        X_scaled = self.scaler.fit_transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns)

    def apply_pca_transform(self, X):
        if self.apply_pca:
            logging.info("Applying PCA...")
            X_pca = self.pca.fit_transform(X)
            cols = [f"PC{i+1}" for i in range(self.n_components)]
            return pd.DataFrame(X_pca, columns=cols)
        return X

    def correlation_filter(self, X, threshold=0.9):
        logging.info("Removing highly correlated features...")
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        drop_cols = [column for column in upper.columns if any(upper[column] > threshold)]
        X_filtered = X.drop(columns=drop_cols)

        logging.info(f"Dropped columns: {drop_cols}")
        return X_filtered

    def save_objects(self, output_dir="models"):
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(self.scaler, f"{output_dir}/scaler.pkl")
        joblib.dump(self.imputer, f"{output_dir}/imputer.pkl")
        if self.pca:
            joblib.dump(self.pca, f"{output_dir}/pca.pkl")

    def full_pipeline(self, file_path):
        df = self.load_data(file_path)
        self.explore_data(df)

        X, y = self.split_features_targets(df)
        X = self.handle_missing(X)
        X = self.scale_features(X)
        X = self.correlation_filter(X)
        X = self.apply_pca_transform(X)

        return X, y
