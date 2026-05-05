# ============================================================

# INFERENCE SCRIPT (LOAD MODEL + PREDICT NEW DATA)

# SESUAI PIPELINE TRAINING TERBARU (ANTI-LEAKAGE + SHAP + K-FOLD)

# ============================================================

# ============================================================

# === LIBRARY IMPORTS ===

# ============================================================

import pandas as pd          # Untuk membaca dan memproses data
import joblib               # Untuk load model .pkl
import numpy as np          # Untuk operasi numerik

# ============================================================

# === 1. LOAD ARTIFACT HASIL TRAINING ===

# ============================================================

# Load stacking model (hasil akhir training)

model = joblib.load("stacked_model.pkl")

# Load scaler (WAJIB sama dengan training)

scaler = joblib.load("scaler.pkl")

# Load fitur hasil selection (CORR ∩ SHAP)

selected_features = joblib.load("selected_features.pkl")

print("[INFO] Model, scaler, dan fitur berhasil di-load")
print(f"[INFO] Total fitur digunakan: {len(selected_features)}")

# ============================================================

# === 2. LOAD DATA BARU (NEW_DATA.csv) ===

# ============================================================

file_path = r"C:\Users\ronny\Documents\kaggle\NEW_DATA.csv"

# Load dataset baru

df_new = pd.read_csv(file_path)

print(f"[INFO] Data baru berhasil di-load dengan shape: {df_new.shape}")

# ============================================================

# === 3. VALIDASI STRUKTUR DATA ===

# ============================================================

# Cek apakah ada fitur penting yang hilang

missing_cols = set(selected_features) - set(df_new.columns)

# Cek apakah ada kolom tambahan

extra_cols = set(df_new.columns) - set(selected_features)

# Jika ada kolom penting hilang → STOP

if missing_cols:
raise ValueError(f"[ERROR] Kolom penting hilang: {missing_cols}")

# Jika ada kolom tambahan → warning saja

if extra_cols:
print(f"[WARNING] Kolom tambahan akan diabaikan: {extra_cols}")

# ============================================================

# === 4. AMBIL FITUR SESUAI MODEL ===

# ============================================================

# Ambil hanya fitur yang digunakan saat training

X_new = df_new[selected_features].copy()

print("[INFO] Fitur berhasil disesuaikan dengan model")

# ============================================================

# === 5. HANDLE DATA QUALITY ===

# ============================================================

# Handle missing values (NaN)

if X_new.isnull().sum().sum() > 0:
print("[WARNING] Missing values ditemukan → diisi median")
X_new = X_new.fillna(X_new.median())

# Handle infinite values

if np.isinf(X_new.values).any():
print("[WARNING] Infinite values ditemukan → dibersihkan")
X_new = X_new.replace([np.inf, -np.inf], np.nan)
X_new = X_new.fillna(X_new.median())

# ============================================================

# === 6. SCALING DATA ===

# ============================================================

# WAJIB pakai scaler dari training (TIDAK boleh fit ulang)

X_scaled = scaler.transform(X_new)

print("[INFO] Scaling selesai")

# ============================================================

# === 7. PREDIKSI ===

# ============================================================

# Jalankan model untuk prediksi

predictions = model.predict(X_scaled)

# Statistik hasil prediksi (debug)

print(f"[INFO] Prediction stats → min: {predictions.min()}, max: {predictions.max()}, mean: {predictions.mean()}")

# ============================================================

# === 8. SIMPAN HASIL PREDIKSI ===

# ============================================================

# Tambahkan hasil prediksi ke dataframe

# NOTE: ini adalah output target yang diprediksi

df_new["PREDICTED_TRANSMITTER_FWD_POWER"] = predictions

# Path output file

output_path = r"C:\Users\ronny\Documents\kaggle\HASIL_PREDIKSI.csv"

# Simpan ke CSV

df_new.to_csv(output_path, index=False)

print("\n[SUCCESS] Prediksi berhasil disimpan ke HASIL_PREDIKSI.csv")

# ============================================================

# === CATATAN PENTING ===

# ============================================================

# 1. File NEW_DATA.csv TIDAK perlu memiliki kolom target (TRANSMITTER_FWD_POWER)

# 2. Harus memiliki SEMUA selected_features

# 3. Boleh ada kolom tambahan → akan diabaikan

# 4. Urutan fitur dijaga otomatis oleh selected_features

# 5. Jangan pernah fit scaler ulang saat inference

# ============================================================

# END SCRIPT

# ============================================================
