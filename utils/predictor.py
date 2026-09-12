import joblib
import shap
import numpy as np
from tensorflow.keras.models import load_model

# ===========================
# Load Models
# ===========================

rf = joblib.load("models/random_forest.pkl")
xgb = joblib.load("models/xgboost.pkl")
autoencoder = load_model("models/autoencoder.keras")

# SHAP Explainer
explainer = shap.TreeExplainer(rf)

print("RF expects:", rf.n_features_in_)
print("XGB expects:", xgb.n_features_in_)
print("Random Forest expects:", rf.n_features_in_, "features")


def predict_all(data, threshold):

    # ===========================
    # Random Forest
    # ===========================

    rf_pred = int(rf.predict(data)[0])
    rf_prob = float(rf.predict_proba(data)[0][1])

    # ===========================
    # XGBoost
    # ===========================

    xgb_pred = int(xgb.predict(data)[0])
    xgb_prob = float(xgb.predict_proba(data)[0][1])

    # ===========================
    # Autoencoder
    # ===========================

    reconstructed = autoencoder.predict(data, verbose=0)

    error = np.mean(np.square(data - reconstructed))

    ae_pred = 1 if error > threshold else 0

    # ===========================
    # SHAP
    # ===========================    

    feature_names = [
        "transaction_id",
        "transaction_time",
        "customer_id",
        "merchant_id",
        "account_age_days",
        "credit_score_band",
        "kyc_level",
        "avg_monthly_spend",
        "merchant_risk_score",
        "transaction_amount",
        "payment_channel",
        "device_type",
        "is_international",
        "ip_risk_score",
        "txn_count_1h",
        "txn_count_24h",
        "failed_txn_count_24h",
        "geo_distance_from_last_txn",
        "amount_deviation_from_user_mean",
    ]

    # Works for both old and new SHAP versions
    # Works for both old and new SHAP versions
    # Handle different SHAP versions
    shap_values = explainer.shap_values(data)
    sample = shap_values[0][:, 1]
    
    print(type(shap_values))
    print(np.array(shap_values).shape)


    shap_data = []

    for feature, value in zip(feature_names, sample):

        shap_data.append({
            "feature": feature,
            "importance": round(abs(float(value)), 4),
            "contribution": round(float(value), 4)
        })

    shap_data = sorted(
        shap_data,
        key=lambda x: x["importance"],
        reverse=True
    )

    # ===========================
    # Console Output
    # ===========================

    print("\n===== MODEL OUTPUT =====")
    print("RF Prediction :", rf_pred)
    print("RF Confidence :", rf_prob)

    print("XGB Prediction:", xgb_pred)
    print("XGB Confidence:", xgb_prob)

    print("AE Error      :", error)
    print("Threshold     :", threshold)
    print("AE Prediction :", ae_pred)

    print("\nTop SHAP Features:")

    for item in shap_data[:5]:
        print(item)

    print("========================\n")

    # ===========================
    # API Response
    # ===========================

    return {

    "random_forest": {
        "prediction": rf_pred,
        "confidence": round(rf_prob * 100, 2)
    },

    "xgboost": {
        "prediction": xgb_pred,
        "confidence": round(xgb_prob * 100, 2)
    },

    "autoencoder": {
        "prediction": ae_pred,
        "threshold": float(threshold),
        "reconstruction_error": float(error)
    },

    "shap": shap_data[:10]

}