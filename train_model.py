
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# -----------------------------
# Step 1: Create Dataset
# -----------------------------
data = {
    'area': [
        600, 800, 1000, 1200, 1500,
        1800, 2000, 2200, 2500, 2800,
        3000, 3200, 3500, 3800, 4000,
        4200, 4500, 4800
    ],
    'bedrooms': [
        1, 2, 2, 2, 3, 3, 3, 3, 4,
        4, 4, 4, 5, 5, 5, 5, 6, 6
    ],
    'floors': [
        1, 1, 1, 1, 1, 1, 2, 2, 2,
        2, 2, 2, 2, 2, 3, 3, 3, 3
    ],
    'age': [
        30, 25, 20, 18, 15, 12, 10, 9, 8,
        7, 6, 5, 5, 4, 4, 3, 3, 2
    ],
    'location': [
        0, 0, 0, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1
    ],
    'water': [
        0, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1
    ],
    'parking': [
        0, 0, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1
    ],
    'nearby_facilities': [
        0, 0, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1
    ],
    'price': [
        900000, 1200000, 1600000, 2000000,
        2600000, 3200000, 3600000, 4000000,
        4600000, 5200000, 5800000, 6200000,
        6800000, 7200000, 7800000, 8200000,
        9000000, 9800000
    ]
}

# -----------------------------
# Step 2: Convert to DataFrame
# -----------------------------
df = pd.DataFrame(data)

# Scale area consistently with app.py
df['area'] = df['area'] / 1000

# -----------------------------
# Step 3: Select Features and Target
# -----------------------------
X = df[
    [
        'area',
        'bedrooms',
        'floors',
        'age',
        'location',
        'water',
        'parking',
        'nearby_facilities'
    ]
]

y = df['price']

# -----------------------------
# Step 4: Split Data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Step 5: Train Model for Evaluation
# -----------------------------
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predict prices for test data
y_pred = model.predict(X_test)

# -----------------------------
# Step 6: Calculate Evaluation Metrics
# -----------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n===== Model Evaluation Metrics =====")
print(f"MAE       : Rs. {mae:,.2f}")
print(f"MSE       : {mse:,.2f}")
print(f"RMSE      : Rs. {rmse:,.2f}")
print(f"R2 Score  : {r2:.4f}")

# -----------------------------
# Step 7: Train Final Model
# -----------------------------
# Retrain using all 18 samples for the application
model.fit(X, y)

# Save the final model
joblib.dump(model, 'model.pkl')

print("\nFinal model trained successfully.")
print("model.pkl saved successfully.")
