# 🚀 Extreme Gradient Boost + Clustering Pipeline

Project ini merupakan pipeline Machine Learning end-to-end untuk:

- Prediksi `TRANSMITTER_FWD_POWER`
- Feature Selection (Correlation + SHAP)
- Model Optimization (Optuna + K-Fold)
- Ensemble Learning (Stacking Model)
- Clustering & Anomaly Detection

---

## 📌 Features Utama

✅ Anti Data Leakage Pipeline  
✅ Feature Selection: Correlation + SHAP  
✅ Hyperparameter Tuning (Optuna)  
✅ Robust Validation (K-Fold)  
✅ Stacking Model (XGBoost + RandomForest + Ridge)  
✅ Production-ready Inference Script  
✅ Clustering + Anomaly Detection (DBSCAN, Isolation Forest, PCA)

---

## 📂 Struktur Project
├── ClusteringAnomalyPipeline.py
├── Xgboost_Pipeline(anti-leakage_Shap_RobustValidation).py
├── predict_transmitter.py
├── xgboost_featureselection.py
├── load_csv.py
├── requirements.txt
├── README.md