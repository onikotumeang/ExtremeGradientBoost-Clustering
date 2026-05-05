# ============================================================
# CLUSTERING + ANOMALY PIPELINE (IMPROVED - PRO LEVEL)
# ============================================================
# FINAL IMPROVEMENTS:
# 1. Better cluster naming (fixed logic)
# 2. Reduced false CRITICAL
# 3. Drift alert detection
# 4. Top anomaly logging
# ============================================================

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest

# ============================================================
# LOAD DATA
# ============================================================
file_path = r"C:\Users\ronny\Documents\kaggle\metering_nec_20kw - cleaned.csv"
df = pd.read_csv(file_path)

# ============================================================
# FEATURE GROUPING
# ============================================================
pa_fwd_cols = [col for col in df.columns if "FWD_POWER" in col and "PA" in col]
pa_temp_cols = [col for col in df.columns if "TEMP" in col and "PA" in col]
pa_dc_cols = [col for col in df.columns if "_DC" in col and "PA" in col]

# ============================================================
# FEATURE ENGINEERING
# ============================================================
df['PA_FWD_MEAN'] = df[pa_fwd_cols].mean(axis=1)
df['PA_FWD_STD'] = df[pa_fwd_cols].std(axis=1)

df['PA_FWD_RANGE'] = df[pa_fwd_cols].max(axis=1) - df[pa_fwd_cols].min(axis=1)
df['PA_TEMP_RANGE'] = df[pa_temp_cols].max(axis=1) - df[pa_temp_cols].min(axis=1)
df['PA_IMBALANCE'] = df['PA_FWD_STD'] / (df['PA_FWD_MEAN'] + 1e-6)

# ============================================================
# TIME SERIES FEATURE
# ============================================================
df['rolling_mean'] = df['PA_FWD_MEAN'].rolling(window=10, min_periods=1).mean()
df['deviation'] = abs(df['PA_FWD_MEAN'] - df['rolling_mean'])

# ============================================================
# FEATURE SET
# ============================================================
features = pa_fwd_cols + pa_temp_cols + pa_dc_cols + [
    'PA_FWD_MEAN','PA_FWD_STD','PA_FWD_RANGE','PA_TEMP_RANGE','PA_IMBALANCE','deviation'
]

X = df[features]

# ============================================================
# SCALING
# ============================================================
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# DRIFT BASELINE
# ============================================================
drift_baseline = {
    "mean": np.mean(X_scaled, axis=0),
    "std": np.std(X_scaled, axis=0)
}
joblib.dump(drift_baseline, "drift_baseline_v1.pkl")

# ============================================================
# PCA
# ============================================================
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

print(f"Original dim: {X_scaled.shape[1]}")
print(f"Reduced dim (PCA): {X_pca.shape[1]}")

# ============================================================
# CLUSTER SEARCH
# ============================================================
print("\n=== SILHOUETTE SEARCH ===")
best_k = 2
best_score = -1

for k in range(2, 8):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    print(f"k={k}, score={score}")

    if score > best_score:
        best_score = score
        best_k = k

print(f"\nBest k: {best_k}")

# ============================================================
# FINAL KMEANS
# ============================================================
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['CLUSTER'] = kmeans.fit_predict(X_pca)

# ============================================================
# DBSCAN TUNING
# ============================================================
print("\n=== DBSCAN TUNING ===")
best_eps = None
best_noise_ratio = 1.0

for eps in [2, 3, 4, 5, 6]:
    db = DBSCAN(eps=eps, min_samples=10)
    labels = db.fit_predict(X_pca)

    noise_ratio = np.mean(labels == -1)
    print(f"eps={eps} -> noise_ratio={noise_ratio:.4f}")

    if 0.001 < noise_ratio < best_noise_ratio:
        best_noise_ratio = noise_ratio
        best_eps = eps

if best_eps is None:
    best_eps = 3

print(f"\nSelected eps: {best_eps}")

# FINAL DBSCAN
dbscan = DBSCAN(eps=best_eps, min_samples=10)
df['ANOMALY_DBSCAN'] = dbscan.fit_predict(X_pca)

# ============================================================
# NOISE RATIO
# ============================================================
final_noise_ratio = np.mean(df['ANOMALY_DBSCAN'] == -1)
print(f"\nFinal Noise Ratio: {final_noise_ratio:.4f}")

# ============================================================
# ISOLATION FOREST
# ============================================================
iso = IsolationForest(contamination=0.02, random_state=42)
df['ANOMALY_IF'] = iso.fit_predict(X_scaled)

# ============================================================
# LABELING (FIXED)
# ============================================================
imbalance_warning_th = df['PA_IMBALANCE'].quantile(0.90)
imbalance_critical_th = df['PA_IMBALANCE'].quantile(0.995)  # FIX
deviation_warning_th = df['deviation'].quantile(0.90)
deviation_critical_th = df['deviation'].quantile(0.98)


def assign_label(row):
    if row['ANOMALY_DBSCAN'] == -1 or row['ANOMALY_IF'] == -1:
        return "CRITICAL"

    if row['PA_IMBALANCE'] > imbalance_critical_th:
        return "CRITICAL"

    if row['deviation'] > deviation_critical_th:
        return "CRITICAL"

    if row['PA_IMBALANCE'] > imbalance_warning_th:
        return "WARNING"

    if row['deviation'] > deviation_warning_th:
        return "WARNING"

    return "NORMAL"


df['STATUS'] = df.apply(assign_label, axis=1)

# ============================================================
# CLUSTER ANALYSIS
# ============================================================
print("\n=== CLUSTER ANALYSIS (MEAN VALUES) ===")
cluster_summary = df.groupby('CLUSTER')[pa_fwd_cols + pa_temp_cols + ['PA_FWD_MEAN','PA_IMBALANCE']].mean()
print(cluster_summary)

# ============================================================
# CLUSTER NAMING (FIXED RULE)
# ============================================================
cluster_names = {}
low_power_threshold = df['PA_FWD_MEAN'].quantile(0.25)

for cluster_id, row in cluster_summary.iterrows():
    mean_power = row['PA_FWD_MEAN']
    imbalance = row['PA_IMBALANCE']

    if imbalance > 0.08:
        cluster_names[cluster_id] = "UNSTABLE"
    elif mean_power < low_power_threshold:
        cluster_names[cluster_id] = "LOW_OUTPUT"
    else:
        cluster_names[cluster_id] = "HEALTHY"


df['CLUSTER_NAME'] = df['CLUSTER'].map(cluster_names)

print("\n=== CLUSTER NAMING ===")
for k, v in cluster_names.items():
    print(f"Cluster {k}: {v}")

# ============================================================
# DRIFT CHECK
# ============================================================
current_mean = np.mean(X_scaled, axis=0)
drift_score = np.mean(np.abs(current_mean - drift_baseline["mean"]))
print(f"\nDrift Score (mean shift): {drift_score:.6f}")

if drift_score > 0.1:
    print("[DRIFT ALERT] Distribution shift detected")

# ============================================================
# ROOT CAUSE ANALYSIS PER ROW (VERY POWERFUL)
# ============================================================
# Tujuan: Menentukan PA mana yang paling berkontribusi terhadap imbalance

# --- kontribusi relatif tiap PA terhadap deviasi dari mean ---
# (|PA_i - mean| / sum |PA_j - mean|) -> proporsi kontribusi
pa_deviation = (df[pa_fwd_cols].sub(df['PA_FWD_MEAN'], axis=0)).abs()
dev_sum = pa_deviation.sum(axis=1) + 1e-6  # avoid division by zero
pa_contrib = pa_deviation.div(dev_sum, axis=0)

# --- ambil TOP-3 PA penyumbang terbesar per baris ---
def top_k_features(row, cols, k=3):
    # row: series of contributions for a single sample
    idx = np.argsort(-row.values)[:k]
    return ','.join([cols[i] for i in idx])

# simpan nama PA yang paling berkontribusi
contrib_cols = pa_fwd_cols
pa_contrib_df = pa_contrib.copy()
pa_contrib_df.columns = [c + '_CONTRIB' for c in pa_contrib_df.columns]

# join ke df utama
df = pd.concat([df, pa_contrib_df], axis=1)

# kolom top contributor
df['TOP3_PA_ROOT_CAUSE'] = pa_contrib.apply(lambda r: top_k_features(r, pa_fwd_cols, k=3), axis=1)

# ============================================================
# TIME-SERIES ANOMALY ESCALATION (TREND BASED)
# ============================================================
# Ide: jika WARNING berulang -> escalate menjadi CRITICAL

# encode status ke numeric untuk rolling
status_map = {'NORMAL': 0, 'WARNING': 1, 'CRITICAL': 2}
df['STATUS_NUM'] = df['STATUS'].map(status_map)

# rolling severity (window bisa dituning)
df['ROLLING_SEVERITY'] = df['STATUS_NUM'].rolling(window=5, min_periods=1).mean()

# escalation rule
# jika rata-rata severity tinggi -> escalate

def escalate_status(row):
    if row['STATUS'] == 'CRITICAL':
        return 'CRITICAL'

    # jika trend buruk (banyak WARNING berturut-turut)
    if row['ROLLING_SEVERITY'] >= 1.2:
        return 'CRITICAL'

    if row['ROLLING_SEVERITY'] >= 0.6:
        return 'WARNING'

    return row['STATUS']


df['STATUS_ESCALATED'] = df.apply(escalate_status, axis=1)

# ============================================================
# TOP ANOMALY LOGGING (ENHANCED)
# ============================================================
print("\n=== TOP IMBALANCE ANOMALIES ===")
cols_show = ['Timestamp','PA_FWD_MEAN','PA_IMBALANCE','STATUS','STATUS_ESCALATED','CLUSTER_NAME','TOP3_PA_ROOT_CAUSE']
print(df.sort_values('PA_IMBALANCE', ascending=False)[cols_show].head(10))

# ============================================================
# SAVE MODELS
# ============================================================
joblib.dump(kmeans, "kmeans_model_v3.pkl")
joblib.dump(dbscan, "dbscan_model_v3.pkl")
joblib.dump(iso, "isolation_forest_v3.pkl")
joblib.dump(scaler, "scaler_clustering_v3.pkl")
joblib.dump(pca, "pca_model_v3.pkl")

# ============================================================
# SAVE RESULT
# ============================================================
output_path = r"C:\Users\ronny\Documents\kaggle\CLUSTERING_HASIL_V3.csv"
df.to_csv(output_path, index=False)

# ============================================================
# ALERT SYSTEM
# ============================================================
status_dist = df['STATUS'].value_counts(normalize=True)

critical_ratio = status_dist.get('CRITICAL', 0)
warning_ratio = status_dist.get('WARNING', 0)

print("\nSTATUS DISTRIBUTION:")
print(status_dist)

if critical_ratio > 0.05:
    print("[ALERT] SYSTEM ALERT: High CRITICAL anomaly rate")
elif warning_ratio > 0.15:
    print("[WARNING] System showing instability trend")
else:
    print("[OK] System operating normally")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\nPipeline selesai (LEVEL PRO MAX):")
print("- Feature imbalance aktif")
print("- Time-series anomaly aktif")
print("- PCA optimization aktif")
print("- Auto DBSCAN tuning")
print("- Auto labeling (NORMAL/WARNING/CRITICAL)")
print("- Drift detection aktif")
print("- Top anomaly logging aktif")
print("- Anomaly detection: DBSCAN + Isolation Forest")
