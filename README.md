<img width="677" height="1280" alt="image" src="https://github.com/user-attachments/assets/6e02c254-6829-4184-b6be-329eb8d65e9a" /># Smart-Traffic-Insights
AI-powered traffic insights for Tech-o-Tsav 2025
1. ENVIRONMENTAL SETUP-
# (optional) create virtualenv
python -m venv venv
# activate:
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# install packages
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter

CREATE requirements.txt
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
jupyter

2) TRAFFIC PREDICTION DATASET --48K ENTRIES, CLEAN AND SIMPLE.
3) About Dataset
Context
Traffic congestion is rising in cities around the world. Contributing factors include expanding urban populations, aging infrastructure, inefficient and uncoordinated traffic signal timing and a lack of real-time data.

The impacts are significant. Traffic data and analytics company INRIX estimates that traffic congestion cost U.S. commuters $305 billion in 2017 due to wasted fuel, lost time and the increased cost of transporting goods through congested areas. Given the physical and financial limitations around building additional roads, cities must use new strategies and technologies to improve traffic conditions.

Content
This dataset contains 48.1k (48120) observations of the number of vehicles each hour in four different junctions:
1) DateTime
2) Juction
3) Vehicles
4) ID

About the data
The sensors on each of these junctions were collecting data at different times, hence you will see traffic data from different time periods. Some of the junctions have provided limited or sparse data requiring thoughtfulness when creating future projections.

Source
(Confidential Source) - Use only for educational purposes
If you use this dataset in your research, please credit the author.


4)  Cell 1 — imports & settings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Optional: display settings
pd.set_option('display.max_columns', 50)


Cell 2 — load dataset (use nrows to speed up if needed)

# If slow, use nrows=5000 to sample
data = pd.read_csv("train.csv")   # or pd.read_csv("train.csv", nrows=5000)
print("Rows, cols:", data.shape)
data.head()


Cell 3 — quick EDA

print(data.info())
print("\nMissing values:\n", data.isnull().sum())
print("\nTrip duration stats:")
display(data['trip_duration'].describe())


Cell 4 — basic preprocessing & time features

# Drop ID & vendor columns which are not useful for quick model
df = data.copy()
if 'id' in df.columns:
    df = df.drop(['id'], axis=1)
if 'vendor_id' in df.columns:
    df = df.drop(['vendor_id'], axis=1)

# Convert pickup datetime
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df['pickup_hour'] = df['pickup_datetime'].dt.hour
df['pickup_day'] = df['pickup_datetime'].dt.dayofweek
df['pickup_month'] = df['pickup_datetime'].dt.month

# Optional: if dropoff_datetime exists, you can drop or ignore
# Remove extreme long trips (outliers)
df = df[df['trip_duration'] < 10000]  # < ~2.7 hours; adjust if necessary

# Show new columns
df[['pickup_datetime','pickup_hour','pickup_day','pickup_month','trip_duration']].head()


Optional Cell 5 — add haversine distance (if you want location feature)
(only if dataset has lat/lon columns: pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude — names may vary)

import math

def haversine(lon1, lat1, lon2, lat2):
    # convert degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km

if {'pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'}.issubset(df.columns):
    df['haversine_km'] = haversine(df['pickup_longitude'], df['pickup_latitude'],
                                   df['dropoff_longitude'], df['dropoff_latitude'])
    print("Added haversine_km. Sample:\n", df[['haversine_km']].head())
else:
    print("Location columns not present or named differently; skipping haversine.")


Cell 6 — choose features & target (quick robust set)

# Choose features for quick model. Add haversine_km if available.
features = ['passenger_count','pickup_hour','pickup_day']
if 'haversine_km' in df.columns:
    features.append('haversine_km')

X = df[features]
y = df['trip_duration']

print("Feature columns:", X.columns.tolist())


Cell 7 — train/test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train size:", X_train.shape, "Test size:", X_test.shape)


Cell 8 — baseline linear regression

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = mean_squared_error(y_test, y_pred_lr, squared=False)
print("Linear Regression MAE:", round(mae_lr,2), "RMSE:", round(rmse_lr,2))


Cell 9 — Random Forest (stronger model)

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)
print("Random Forest MAE:", round(mae_rf,2), "RMSE:", round(rmse_rf,2))


Cell 10 — simple evaluation plots

# Residual distribution (RF)
residuals = y_test - y_pred_rf
plt.figure(figsize=(6,4))
sns.histplot(residuals, bins=50, kde=True)
plt.title("Residuals (y_test - y_pred_rf)")
plt.show()

# True vs Pred
plt.figure(figsize=(6,6))
plt.scatter(y_test.sample(1000, random_state=42), y_pred_rf[:1000], alpha=0.3)
plt.xlabel("True trip duration (s)")
plt.ylabel("Predicted trip duration (s)")
plt.title("True vs Predicted (sample 1000)")
plt.show()


Cell 11 — save model & scaler if used (we didn't scale here, but we save model)

joblib.dump(rf, "rf_model.joblib")
# If you used a scaler: joblib.dump(scaler, "scaler.joblib")
print("Model saved: rf_model.joblib")


Cell 12 — prediction helper function

def predict_trip_duration(passenger_count, pickup_hour, pickup_day, hav_km=None):
    row = [passenger_count, pickup_hour, pickup_day]
    if 'haversine_km' in X.columns:
        if hav_km is None:
            print("Please provide haversine_km or compute coordinates.")
            return
        row.append(hav_km)
    sample = np.array(row).reshape(1, -1)
    pred = rf.predict(sample)[0]
    return int(pred), f"{int(pred/60)} minutes"

# Example
print("Example prediction (2 passengers, 9 AM, Tuesday):", predict_trip_duration(2, 9, 1))


Cell 13 — save notebook outputs (optional)

# Save a small CSV with sample predictions for demonstration
sample_df = X_test.copy()
sample_df['true_duration'] = y_test
sample_df['pred_rf'] = y_pred_rf
sample_df[['passenger_count','pickup_hour','pickup_day','true_duration','pred_rf']].head(20).to_csv('sample_predictions.csv', index=False)
print("Saved sample_predictions.csv")
============================================================================================================================================
 # 🚦 Smart Traffic Insights (NYC Taxi Trip Duration)

**Team:** [AIgnition]  
**Event:** Tech-o-Tsav 2025 – AI-Fusion (Round 2)
============================================================================================================================================
## OVERVIEW--
Predict trip duration using NYC Taxi dataset to help route optimization and delivery planning.

## Files
- `notebook.ipynb` — Jupyter notebook with preprocessing, models, and evaluation.
- `rf_model.joblib` — Saved Random Forest model.
- `report.pdf` — Documentation (or `report.md`).
- `workflow.png` — Flowchart image.
- `sample_predictions.csv` — Example predictions.

## How to run
1. Clone:
   ```bash
   git clone https://github.com/<your-username>/Smart-Traffic-Insights.git
   cd Smart-Traffic-Insights

   CONTACT:
   Team members: A, B, C


Replace `<your-username>` and team details.

---

# 📝 5) `report.md` (documentation) — paste & convert to PDF
Use the earlier documentation template; here is a compact version to paste into `report.md`:

```markdown
# Smart Traffic Insights for Indian Cities
**Team:** [AIgnition]  
**Event:** Tech-o-Tsav 2025 – AI-Fusion

## 1. Problem Statement
Predict trip duration to help citizens, delivery services and city planners optimize routes and reduce delays.

## 2. Dataset
NYC Taxi Trip Duration (train.csv). ~55k rows. Key features: pickup_datetime, passenger_count, trip_duration, pickup/dropoff coords.

## 3. Approach
- Preprocess: datetime → hour, day, month; remove outliers.
- Features: passenger_count, pickup_hour, pickup_day, (optional haversine_km).
- Models: Linear Regression (baseline), Random Forest (final).
- Evaluation: MAE, RMSE.

## 4. Results
- Random Forest MAE: [fill value]
- Random Forest RMSE: [fill value]
(Include charts / sample predictions)

## 5. Business Impact
- ETA prediction for deliveries, better scheduling for delivery firms, traffic-aware planning for city officials.

## 6. Future Work
- Add real-time data, weather, traffic density. Use deep learning/time-series models for better temporal modeling.

## 7. Deliverables
- Jupyter Notebook, flowchart, report, demo video, GitHub repo link.

FLOWCHART--


Dataset

→ Data Cleaning

→ Feature Engineering (hour/day/passenger_count, haversine)

→ Model Training (LinearReg, RandomForest)

→ Evaluation (MAE, RMSE, plots)

→ Output: Predictions / sample API / CSV / Business insights

