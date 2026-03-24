# ==========================================================
# MODEL TRAINING MODULE (MULTI-TRAIT ML)
# ==========================================================

import logging
import joblib
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputRegressor

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR

logging.basicConfig(level=logging.INFO)

class TraitModel:

    def __init__(self, model_type="rf"):
        self.model_type = model_type
        self.model = None

    def build(self):
        if self.model_type == "rf":
            base = RandomForestRegressor(n_estimators=200, random_state=42)

        elif self.model_type == "gboost":
            base = GradientBoostingRegressor()

        elif self.model_type == "ridge":
            base = Ridge()

        elif self.model_type == "lasso":
            base = Lasso()

        elif self.model_type == "svr":
            base = SVR()

        else:
            raise ValueError("Invalid model")

        self.model = MultiOutputRegressor(base)
        return self.model

    # -----------------------------------
    def train(self, X, y):
        logging.info("Splitting dataset...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = self.build()

        logging.info("Training model...")
        model.fit(X_train, y_train)

        return model, X_test, y_test

    # -----------------------------------
    def tune(self, X, y):
        logging.info("Hyperparameter tuning...")

        params = {
            "estimator__n_estimators": [100, 200],
            "estimator__max_depth": [5, 10]
        }

        model = MultiOutputRegressor(RandomForestRegressor())

        grid = GridSearchCV(model, params, cv=3)
        grid.fit(X, y)

        return grid.best_estimator_

    # -----------------------------------
    def save(self, model, path="models"):
        os.makedirs(path, exist_ok=True)
        joblib.dump(model, f"{path}/model.pkl")
