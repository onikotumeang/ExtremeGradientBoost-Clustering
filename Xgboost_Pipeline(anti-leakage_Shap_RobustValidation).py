# ============================================================
# IMPROVED PIPELINE: ANTI-LEAKAGE + SHAP + K-FOLD VALIDATION
# ============================================================
# Perbaikan utama:
# 1. Anti data leakage (feature selection hanya di train)
# 2. Kombinasi Correlation + SHAP
# 3. Robust validation menggunakan K-Fold
# 4. Pipeline lebih clean & production ready
# ============================================================

# === LIBRARY IMPORTS ===
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import optuna
import joblib

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge

# ============================================================
# === LOAD DATASET ===
# ============================================================
file_path = r"C:\Users\ronny\Documents\kaggle\metering_nec_20kw - cleaned.csv"
df = pd.read_csv(file_path)

# ============================================================
# === DEFINE FEATURE & TARGET ===
# ============================================================
all_features = df.columns.tolist()
all_features.remove("TRANSMITTER_FWD_POWER")

X_full = df[all_features]
y = df["TRANSMITTER_FWD_POWER"]

# ============================================================
# === SPLIT DATA (ANTI-LEAKAGE STEP) ===
# ============================================================
# Feature selection WAJIB dilakukan setelah split
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)

# ============================================================
# === 1. CORRELATION FEATURE SELECTION (TRAIN ONLY) ===
# ============================================================
train_df = X_train_full.copy()
train_df["TARGET"] = y_train_full

correlation_with_target = train_df.corr(numeric_only=True)["TARGET"].abs().sort_values(ascending=False)

# Ambil top 40 dari TRAIN ONLY (anti leakage)
top_corr_features = correlation_with_target[1:40].index.tolist()

print("\n=== TOP 40 FEATURES (TRAIN ONLY - NO LEAKAGE) ===")
print(correlation_with_target[1:40])

# Filter dataset
X_train_corr = X_train_full[top_corr_features]
X_test_corr = X_test[top_corr_features]

# ============================================================
# === 2. SHAP FEATURE SELECTION (TRAIN ONLY) ===
# ============================================================
scaler_temp = RobustScaler()
X_train_scaled_temp = scaler_temp.fit_transform(X_train_corr)

xgb_base = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, random_state=42)
xgb_base.fit(X_train_scaled_temp, y_train_full)

explainer = shap.Explainer(xgb_base)
shap_values = explainer(X_train_scaled_temp)

shap_importance = np.abs(shap_values.values).mean(axis=0)

shap_df = pd.DataFrame({
    "Feature": X_train_corr.columns,
    "SHAP Importance": shap_importance
}).sort_values(by="SHAP Importance", ascending=False)

# Ambil top 20 SHAP
shap_top20 = shap_df["Feature"].head(20).tolist()

# ============================================================
# === 3. COMBINE CORRELATION + SHAP ===
# ============================================================
# Strategi: Intersection (lebih robust)
final_features = list(set(top_corr_features).intersection(set(shap_top20)))

print("\n=== FINAL FEATURES (CORR INTERSECT SHAP) ===")
print(final_features)

# Apply ke dataset
X_train = X_train_full[final_features]
X_test = X_test[final_features]

# ============================================================
# === 4. SCALING FINAL DATA ===
# ============================================================
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# === 5. ROBUST VALIDATION (K-FOLD + OPTUNA) ===
# ============================================================
# Tidak pakai single validation lagi -> lebih stabil

kf = KFold(n_splits=5, shuffle=True, random_state=42)


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

    scores = []

    # K-FOLD LOOP
    for train_idx, val_idx in kf.split(X_train_scaled):
        X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_tr, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

        model = xgb.XGBRegressor(objective='reg:squarederror', **params, random_state=42)
        model.fit(X_tr, y_tr, verbose=False)

        pred = model.predict(X_val)
        score = mean_absolute_error(y_val, pred)
        scores.append(score)

    return np.mean(scores)


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

best_params = study.best_params
print("\nBest hyperparameters:", best_params)

# ============================================================
# === 6. FINAL MODEL TRAINING ===
# ============================================================
best_xgb = xgb.XGBRegressor(objective='reg:squarederror', **best_params, random_state=42)
best_xgb.fit(X_train_scaled, y_train_full)

# ============================================================
# === 7. STACKING MODEL ===
# ============================================================
stacking_model = StackingRegressor(
    estimators=[
        ('rf', RandomForestRegressor(n_estimators=250, random_state=42)),
        ('xgb', best_xgb)
    ],
    final_estimator=Ridge(alpha=1.0)
)

stacking_model.fit(X_train_scaled, y_train_full)

# ============================================================
# === 8. EVALUATION (TEST SET) ===
# ============================================================
y_pred = stacking_model.predict(X_test_scaled)

print("\n=== FINAL EVALUATION ===")
print("MAE :", mean_absolute_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
print("R²  :", r2_score(y_test, y_pred))
print("MAPE:", mean_absolute_percentage_error(y_test, y_pred))

# ============================================================
# === 9. SAVE ARTIFACT ===
# ============================================================
joblib.dump(stacking_model, "stacked_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(final_features, "selected_features.pkl")

# ============================================================
# CATATAN PENTING:
# - Pipeline ini sudah ANTI-LEAKAGE
# - Feature selection hanya dari TRAIN
# - Validasi pakai K-Fold (lebih stabil dari single split)
# - SHAP + Correlation digabung (lebih robust)
# ============================================================
