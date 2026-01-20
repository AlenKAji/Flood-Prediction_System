import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -----------------------------
# Load dataset
# -----------------------------
data = pd.read_csv('flood[1].csv')
print("Data Loaded")

# Add Latitude and Longitude (Kerala-based dummy values)
np.random.seed(42)
data['Latitude'] = np.random.uniform(8.0, 12.5, size=len(data))
data['Longitude'] = np.random.uniform(74.8, 77.5, size=len(data))

# Drop missing values
data.dropna(inplace=True)

# -----------------------------
# Separate features and target
# -----------------------------
X = data.drop(['FloodProbability', 'Latitude', 'Longitude'], axis=1)
y = data['FloodProbability']
print("Data Preprocessing Done")

# -----------------------------
# Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Base model for RFE
# -----------------------------
base_model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42
)

# -----------------------------
# Apply Recursive Feature Elimination
# -----------------------------
rfe = RFE(
    estimator=base_model,
    n_features_to_select=10,   
    step=1
)

X_rfe = rfe.fit_transform(X_scaled, y)

selected_features = X.columns[rfe.support_]
print("Selected Features after RFE:")
print(list(selected_features))

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_rfe, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train final model
# -----------------------------
final_model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42
)

final_model.fit(X_train, y_train)
print("Model Training Complete")

# -----------------------------
# Prediction & Evaluation
# -----------------------------
y_pred = final_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nGradient Boosting + RFE Model Performance:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# -----------------------------
# Feature Importance Plot (after RFE)
# -----------------------------
importances = final_model.feature_importances_
sorted_idx = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), importances[sorted_idx])
plt.yticks(range(len(sorted_idx)), selected_features[sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Feature Importance After RFE")
plt.tight_layout()
plt.show()
