# === LIBRARY IMPORTS ===
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import optuna
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge

# === LOAD DATASET ===
file_path = r"C:\Users\ronny\Documents\kaggle\metering_nec_20kw - cleaned.csv"
df = pd.read_csv(file_path)

# === FITUR & TARGET ===
all_features = df.columns.tolist()
all_features.remove("TRANSMITTER_FWD_POWER")
X_full = df[all_features]
y = df["TRANSMITTER_FWD_POWER"]

# === 1. FEATURE SELECTION BERDASARKAN KORELASI TERHADAP TARGET ===
correlation_with_target = df.corr(numeric_only=True)["TRANSMITTER_FWD_POWER"].abs().sort_values(ascending=False)
top_corr_features = correlation_with_target[1:40].index.tolist()  # Ambil 40 fitur teratas selain target

X = df[top_corr_features]
print("\n=== TOP 40 FEATURES BERDASARKAN KORELASI ===")
print(correlation_with_target[1:40])

# === SPLIT DATA ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# === SCALING ===
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

# === 2. FITUR SELECTION LANJUTAN DENGAN SHAP ===
xgb_base = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=150, random_state=42)
xgb_base.fit(X_train_scaled, y_train)

explainer = shap.Explainer(xgb_base)
shap_values = explainer(X_train_scaled)
shap_importance = np.abs(shap_values.values).mean(axis=0)

# Ambil 20 fitur SHAP tertinggi dari hasil korelasi awal
shap_df = pd.DataFrame({
    "Feature": X.columns,
    "SHAP Importance": shap_importance
}).sort_values(by="SHAP Importance", ascending=False)

top_shap_features = shap_df["Feature"][:20].tolist()

import matplotlib.pyplot as plt

correlation_with_target[1:21].plot(kind='barh')
plt.title("Top 20 Features by Correlation")
plt.gca().invert_yaxis()
plt.show()

shap.summary_plot(shap_values, X_train_scaled, feature_names=X.columns)

# === FINAL FEATURE SET ===
X_train_selected = X_train[top_shap_features]
X_valid_selected = X_valid[top_shap_features]
X_test_selected = X_test[top_shap_features]

# Scaling ulang untuk fitur terpilih
X_train_scaled = scaler.fit_transform(X_train_selected)
X_valid_scaled = scaler.transform(X_valid_selected)
X_test_scaled = scaler.transform(X_test_selected)

# === 3. HYPERPARAMETER TUNING OPTUNA UNTUK XGBOOST ===
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.1),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0)
    }
    model = xgb.XGBRegressor(objective='reg:squarederror', **params, random_state=42)
    model.fit(X_train_scaled, y_train, eval_set=[(X_valid_scaled, y_valid)], verbose=False)
    pred = model.predict(X_valid_scaled)
    return mean_absolute_error(y_valid, pred)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=75)

best_params = study.best_params
print("Best hyperparameters:", best_params)

# === 4. TRAIN FINAL XGBOOST MODEL ===
best_xgb = xgb.XGBRegressor(objective='reg:squarederror', **best_params, random_state=42)
best_xgb.fit(X_train_scaled, y_train)

# === 5. STACKING MODEL DENGAN RIDGE SEBAGAI META-LEARNER ===
stacking_model = StackingRegressor(
    estimators=[
        ('rf', RandomForestRegressor(n_estimators=250, random_state=42)),
        ('xgb', best_xgb)
    ],
    final_estimator=Ridge(alpha=1.0)
)
stacking_model.fit(X_train_scaled, y_train)

# === 6. EVALUASI PADA TEST SET ===
y_pred = stacking_model.predict(X_test_scaled)

print("=== Evaluation Metrics ===")
print("MAE :", mean_absolute_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
print("R²  :", r2_score(y_test, y_pred))
print("MAPE:", mean_absolute_percentage_error(y_test, y_pred))

# === 7. SAVE MODEL DAN SCALER ===
joblib.dump(stacking_model, "stacked_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(top_shap_features, "selected_features.pkl")
