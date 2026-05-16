from supervised.regression.linear_regression import LinearRegressionModel
 
 
def main():
    print("=" * 50)
    print("  ML Learning Repo")
    print("=" * 50)
 
    # --- Linear Regression Demo ---
    print("\n[ Linear Regression ]\n")
 
    model = LinearRegressionModel()
 
    # Generate synthetic data and run the full pipeline
    X, y = model.generate_data(n_samples=100, noise=15, random_state=42)
    X_train, X_test, y_train, y_test = model.split(X, y)
 
    model.train(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)
 
    print(f"  Coefficients : {model.coef_}")
    print(f"  Intercept    : {model.intercept_:.4f}")
    print(f"  R² Score     : {metrics['r2']:.4f}")
    print(f"  MSE          : {metrics['mse']:.4f}")
    print(f"  RMSE         : {metrics['rmse']:.4f}")
 
    model.plot(X_test, y_test, title="Linear Regression - Test Set")
 
 
if __name__ == "__main__":
    main()
 
