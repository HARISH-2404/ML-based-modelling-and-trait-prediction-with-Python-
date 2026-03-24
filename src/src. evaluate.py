# ==========================================================
# MODEL EVALUATION MODULE
# ==========================================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

class Evaluator:

    def __init__(self, model):
        self.model = model

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)

        results = {}

        for i, col in enumerate(y_test.columns):

            r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
            rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i]))
            mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])

            results[col] = {"R2": r2, "RMSE": rmse, "MAE": mae}

            print(f"{col} -> R2={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}")

        return y_pred, results

    def plot(self, y_test, y_pred):
        for i, col in enumerate(y_test.columns):

            plt.figure()
            plt.scatter(y_test.iloc[:, i], y_pred[:, i])
            plt.xlabel("Observed")
            plt.ylabel("Predicted")
            plt.title(col)
            plt.show()

    def residuals(self, y_test, y_pred):
        for i, col in enumerate(y_test.columns):

            res = y_test.iloc[:, i] - y_pred[:, i]

            plt.figure()
            plt.hist(res, bins=20)
            plt.title(f"{col} Residuals")
            plt.show()
