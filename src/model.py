import numpy as np

class LinearRegressionNormal:

    def __init__(self):
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        Xb = np.c_[np.ones((X.shape[0], 1)), X]
        A = np.linalg.inv(Xb.T @ Xb) @ Xb.T @ y

        self.intercept_ = A[0]
        self.coef_ = A[1:]

    def predict(self, X):
        X = np.array(X, dtype=float)
        return X @ self.coef_ + self.intercept_