import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from src.model import LinearRegressionNormal

# ── Load Data ─────────────────────────────────────────────────────────
insurance = pd.read_csv('data/insurance.csv')

# ── Preprocessing ─────────────────────────────────────────────────────
insurance["log_charges"] = np.log2(insurance["charges"])
insurance["is_smoker"] = (insurance["smoker"] == "yes")

X = insurance[["age", "bmi", "is_smoker"]]
y = insurance["log_charges"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ── Baseline Model (sklearn) ──────────────────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)

# ── Evaluation ────────────────────────────────────────────────────────
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)

train_mse = mean_squared_error(y_train, train_pred)
test_mse = mean_squared_error(y_test, test_pred)

print(f"Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
print(f"Train MSE: {train_mse:.4f} | Test MSE: {test_mse:.4f}")

# ── Custom Normal Equation Model ──────────────────────────────────────
custom_model = LinearRegressionNormal()
custom_model.fit(X_train, y_train)

train_pred_custom = custom_model.predict(X_train)
test_pred_custom = custom_model.predict(X_test)

train_r2_custom = r2_score(y_train, train_pred_custom)
test_r2_custom = r2_score(y_test, test_pred_custom)

print(f"Sklearn  → Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
print(f"Custom   → Train R²: {train_r2_custom:.4f} | Test R²: {test_r2_custom:.4f}")

# ── Diagnostic Plots ──────────────────────────────────────────────────
plot_df = pd.DataFrame({
    'predictions': train_pred,
    'actual': y_train,
    'is_smoker': X_train['is_smoker'],
    'age': X_train['age'],
    'bmi': X_train['bmi'],
    'residuals': y_train - train_pred,
})

plt.figure(figsize=(10, 6))
sns.scatterplot(x='predictions', y='actual', data=plot_df, alpha=0.7)
plt.plot([11, 16], [11, 16], 'k--', alpha=0.5)
plt.xlabel('Predicted log_charges')
plt.ylabel('Actual log_charges')
plt.title('Predicted vs Actual')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='predictions', y='residuals', data=plot_df, alpha=0.7)
plt.axhline(0, color='k', linestyle='--', alpha=0.5)
plt.xlabel('Predictions')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()