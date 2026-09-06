import numpy as np
import pytest

from supervised.regression.linear_regression import LinearRegressionModel


def test_generate_data_returns_expected_shapes():
    model = LinearRegressionModel()

    X, y = model.generate_data(n_samples=50, n_features=2, random_state=42)

    assert X.shape == (50, 2)
    assert y.shape == (50,)


def test_generate_data_is_reproducible_with_random_state():
    model = LinearRegressionModel()

    X1, y1 = model.generate_data(n_samples=25, random_state=7)
    X2, y2 = model.generate_data(n_samples=25, random_state=7)

    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_split_preserves_all_samples():
    model = LinearRegressionModel()
    X, y = model.generate_data(n_samples=100, random_state=42)

    X_train, X_test, y_train, y_test = model.split(
        X, y, test_size=0.25, random_state=42
    )

    assert len(X_train) == 75
    assert len(X_test) == 25
    assert len(y_train) == 75
    assert len(y_test) == 25


def test_predict_before_training_raises_runtime_error():
    model = LinearRegressionModel()
    X, _ = model.generate_data(n_samples=10)

    with pytest.raises(RuntimeError, match="Model has not been trained yet"):
        model.predict(X)


def test_model_trains_and_predicts():
    model = LinearRegressionModel()
    X, y = model.generate_data(n_samples=100, noise=5, random_state=42)
    X_train, X_test, y_train, _ = model.split(X, y, random_state=42)

    model.train(X_train, y_train)
    predictions = model.predict(X_test)

    assert model.is_trained is True
    assert predictions.shape == (len(X_test),)


def test_evaluate_returns_expected_metrics():
    model = LinearRegressionModel()
    X, y = model.generate_data(n_samples=200, noise=5, random_state=42)
    X_train, X_test, y_train, y_test = model.split(X, y, random_state=42)

    model.train(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)

    assert set(metrics) == {"r2", "mse", "rmse"}
    assert metrics["r2"] <= 1.0
    assert metrics["mse"] >= 0.0
    assert metrics["rmse"] >= 0.0
    assert metrics["rmse"] == pytest.approx(np.sqrt(metrics["mse"]))


def test_coefficients_are_available_after_training():
    model = LinearRegressionModel()
    X, y = model.generate_data(n_samples=100, n_features=2, random_state=42)

    model.train(X, y)

    assert model.coef_.shape == (2,)
    assert np.isscalar(model.intercept_)


def test_coefficients_before_training_raise_runtime_error():
    model = LinearRegressionModel()

    with pytest.raises(RuntimeError, match="Model has not been trained yet"):
        _ = model.coef_

    with pytest.raises(RuntimeError, match="Model has not been trained yet"):
        _ = model.intercept_
