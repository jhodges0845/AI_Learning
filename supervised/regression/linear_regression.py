"""
supervised/regression/linear_regression.py
===========================================
A clean wrapper around scikit-learn's LinearRegression.

Covers:
  - Synthetic data generation
  - Train/test splitting
  - Model training
  - Evaluation (R², MSE, RMSE)
  - Plotting
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_regression


class LinearRegressionModel:
    """
    Simple Linear Regression wrapper.

    Usage:
        model = LinearRegressionModel()
        X, y = model.generate_data()
        X_train, X_test, y_train, y_test = model.split(X, y)
        model.train(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)
        model.plot(X_test, y_test)
    """

    def __init__(self):
        self.model = LinearRegression()
        self.is_trained = False

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def generate_data(
        self,
        n_samples: int = 100,
        n_features: int = 1,
        noise: float = 10.0,
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate synthetic regression data using sklearn's make_regression."""
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=noise,
            random_state=random_state,
        )
        return X, y

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> tuple:
        """Split data into train and test sets."""
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the model to training data."""
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions for X."""
        self._check_trained()
        return self.model.predict(X)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Return a dict with R², MSE, and RMSE."""
        self._check_trained()
        y_pred = self.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        return {
            "r2": r2_score(y_test, y_pred),
            "mse": mse,
            "rmse": np.sqrt(mse),
        }

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot(
        self,
        X: np.ndarray,
        y: np.ndarray,
        title: str = "Linear Regression",
    ) -> None:
        """Scatter plot of actual vs predicted values (single-feature only)."""
        self._check_trained()

        if X.shape[1] != 1:
            print("plot() only supports single-feature data. Skipping.")
            return

        y_pred = self.predict(X)

        plt.figure(figsize=(8, 5))
        plt.scatter(X, y, color="steelblue", alpha=0.7, label="Actual")
        plt.plot(X, y_pred, color="tomato", linewidth=2, label="Predicted")
        plt.title(title)
        plt.xlabel("Feature")
        plt.ylabel("Target")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Convenience properties (mirror sklearn's API)
    # ------------------------------------------------------------------

    @property
    def coef_(self):
        self._check_trained()
        return self.model.coef_

    @property
    def intercept_(self):
        self._check_trained()
        return self.model.intercept_

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _check_trained(self):
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet. Call train() first.")