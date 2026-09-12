import pandas as pd

def generate_features(data):

    # Convert datetime to Unix timestamp
    time = data.get("transaction_time", "")

    if time:
        transaction_time = pd.to_datetime(time).timestamp()
    else:
        transaction_time = 0

    # Convert dropdowns to numeric values
    credit_score_map = {
        "Poor": 0,
        "Fair": 1,
        "Good": 2,
        "Excellent": 3
    }

    kyc_map = {
        "Minimal": 0,
        "Partial": 1,
        "Full": 2
    }

    payment_map = {
        "UPI": 0,
        "Credit Card": 1,
        "Debit Card": 2,
        "Wallet": 3,
        "Net Banking": 4
    }

    device_map = {
        "Android": 0,
        "iPhone": 1,
        "Laptop": 2,
        "Tablet": 3
    }

    features = {
        "transaction_id": int(data["transaction_id"]),
        "transaction_time": transaction_time,
        "customer_id": int(data["customer_id"]),
        "merchant_id": int(data["merchant_id"]),

        "account_age_days": int(data["account_age_days"]),
        "credit_score_band": credit_score_map[data["credit_score_band"]],
        "kyc_level": kyc_map[data["kyc_level"]],

        "avg_monthly_spend": float(data["avg_monthly_spend"]),
        "merchant_risk_score": float(data["merchant_risk_score"]),
        "transaction_amount": float(data["transaction_amount"]),

        "payment_channel": payment_map[data["payment_channel"]],
        "device_type": device_map[data["device_type"]],
        "is_international": 1 if data["is_international"] == "Yes" else 0,

        "ip_risk_score": float(data["ip_risk_score"]),
        "txn_count_1h": int(data["txn_count_1h"]),
        "txn_count_24h": int(data["txn_count_24h"]),
        "failed_txn_count_24h": int(data["failed_txn_count_24h"]),
        "geo_distance_from_last_txn": float(data["geo_distance_from_last_txn"]),
        "amount_deviation_from_user_mean": float(data["amount_deviation_from_user_mean"])
    }

    return pd.DataFrame([features])