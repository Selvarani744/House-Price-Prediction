import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

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
        1,2,2,2,3,
        3,3,3,4,4,
        4,4,5,5,5,
        5,6,6
    ],

    'floors': [
        1,1,1,1,1,
        1,2,2,2,2,
        2,2,2,2,3,
        3,3,3
    ],

    'age': [
        30,25,20,18,15,
        12,10,9,8,7,
        6,5,5,4,4,
        3,3,2
    ],

    'location': [
        0,0,0,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1
    ],

    'water': [
        0,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1
    ],

    'parking': [
        0,0,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1
    ],

    'nearby_facilities': [
        0,0,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1
    ],

    'price': [
        900000,   # 600
        1200000,  # 800
        1600000,  # 1000
        2000000,  # 1200
        2600000,  # 1500
        3200000,  # 1800
        3600000,  # 2000
        4000000,  # 2200
        4600000,  # 2500
        5200000,  # 2800
        5800000,  # 3000
        6200000,  # 3200
        6800000,  # 3500
        7200000,  # 3800
        7800000,  # 4000
        8200000,  # 4200
        9000000,  # 4500
        9800000   # 4800
    ]
}
# -----------------------------
# Step 2: Convert to DataFrame
# -----------------------------
df = pd.DataFrame(data)
df['area'] = df['area'] / 1000

# -----------------------------
# Step 3: Select Features & Target
# -----------------------------
X = df[['area', 'bedrooms', 'floors', 'age', 'location', 'water', 'parking', 'nearby_facilities']]
y = df['price']

# -----------------------------
# Step 4: Train the Model
# -----------------------------
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# -----------------------------
# Step 5: Save the Trained Model
# -----------------------------
joblib.dump(model, 'model.pkl')

print("Model trained successfully and model.pkl saved")
