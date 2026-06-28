import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

insurance = pd.read_csv('data/insurance.csv')

insurance["log_charges"] = np.log2(insurance["charges"])
insurance["is_smoker"] = (insurance["smoker"] == "yes")

X = insurance[["age", "bmi", "is_smoker"]]
y = insurance["log_charges"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)