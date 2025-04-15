import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

#Load dataset
data = pd.read_csv('flood[1].csv')
print("Data Loaded")

# Add Latitude and Longitude (Kerala-based dummy values)
np.random.seed(42)
data['Latitude'] = np.random.uniform(8.0, 12.5, size=len(data))
data['Longitude'] = np.random.uniform(74.8, 77.5, size=len(data))

# Drop missing values
data = data.dropna()

# Separate features and target
X = data.drop(['FloodProbability', 'Latitude', 'Longitude'], axis=1)
y = data['FloodProbability']
print("Data Preprocessing Done")

# Normalize features (optional but often helps)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Gradient Boosting with tuned hyperparameters
model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42
)
model.fit(X_train, y_train)
print("Model Training Complete")

# Prediction & Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Gradient Boosting Model Performance (Improved):")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

import matplotlib.pyplot as plt

feature_importance = model.feature_importances_
feature_names = X.columns
sorted_idx = np.argsort(feature_importance)

plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Gradient Boosting Feature Importance")
plt.tight_layout()
plt.show()
