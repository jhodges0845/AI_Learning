"""
supervised/regression/polynomial_regression.py
================================================
A clean wrapper around scikit-learn polynomial regression.

Covers:
  - Synthetic curved data generation
  - Train/test splitting
  - Polynomial feature transformation
  - Model training
  - Evaluation (R², MSE, RMSE)
  - Plotting
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score


class PolynomialRegressionModel:
    """Simple polynomial regression wrapper."""

    def __init__(self, degree: int = 2):
        self.degree = degree
        self.polynomial_features = PolynomialFeatures(
            degree=degree,
            include_bias=False,
        )
        self.model = LinearRegression()
        self.is_trained = False

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def generate_data(
        self,
        n_samples: int = 100,
        noise: float = 10.0,
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate synthetic data with a curved quadratic relationship."""
        rng = np.random.default_rng(random_state)
        X = rng.uniform(-5, 5, size=(n_samples, 1))

        # Known relationship used for learning/demo purposes:
        # y = 3x² + 2x + 5 + random noise
        y = (
            3 * X[:, 0] ** 2
            + 2 * X[:, 0]
            + 5
            + rng.normal(0, noise, n_samples)
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
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Transform the features and fit the regression model."""
        X_poly = self.polynomial_features.fit_transform(X_train)
        self.model.fit(X_poly, y_train)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Transform X into polynomial features and return predictions."""
        self._check_trained()
        X_poly = self.polynomial_features.transform(X)
        return self.model.predict(X_poly)

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
        title: str = "Polynomial Regression",
    ) -> None:
        """Scatter plot of actual values and the predicted curve."""
        self._check_trained()

        if X.shape[1] != 1:
            print("plot() only supports single-feature data. Skipping.")
            return

        # Sort X so matplotlib draws the prediction as a smooth curve.
        sort_indices = np.argsort(X[:, 0])
        X_sorted = X[sort_indices]
        y_pred = self.predict(X_sorted)

        plt.figure(figsize=(8, 5))
        plt.scatter(X, y, alpha=0.7, label="Actual")
        plt.plot(X_sorted, y_pred, linewidth=2, label="Predicted")
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
