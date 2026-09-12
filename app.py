from flask import Flask, request, jsonify
from flask_cors import CORS
from utils.gemini import generate_explanation
from utils.feature_generator import generate_features
from utils.preprocess import preprocess
from utils.predictor import predict_all
import os
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load Autoencoder Threshold
with open("models/threshold.txt", "r") as f:
    threshold = float(f.read().strip())

print("Threshold:", threshold)


@app.route("/")
def home():
    return {"message": "UPI Fraud Detection API Running"}


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    # Generate Features
    df = generate_features(data)

    print(df.columns.tolist())
    print(df.shape)

    # Preprocess
    processed = preprocess(df)

    # Predict
    result = predict_all(processed, threshold)
    ai_explanation = generate_explanation(result)

    result["gemini_explanation"] = ai_explanation



    os.makedirs("data", exist_ok=True)

    history_file = os.path.abspath("data/history.csv")

    print("Saving to:", history_file)

    row = {
    "Transaction ID": data["transaction_id"],
    "Amount": data["transaction_amount"],
    "Customer": data["customer_id"],
    "Merchant": data["merchant_id"],
    "Prediction": "Fraud" if result["random_forest"]["prediction"] == 1 else "Legitimate",
    "Confidence": result["random_forest"]["confidence"]
}

    df_history = pd.DataFrame([row])
    print("Saving transaction to history...")
    print(row)

    os.makedirs("data", exist_ok=True)

    if os.path.exists(history_file):
        df_history.to_csv(history_file, mode="a", header=False, index=False)
    else:
        df_history.to_csv(history_file, index=False)
    print("History saved successfully!")
    print("File exists:", os.path.exists(history_file))
    print("File size:", os.path.getsize(history_file))

    print("========== RESULT ==========")
    print(result)
    print("============================")
   

    return jsonify(result)

@app.route("/history")
def history():

    import pandas as pd

    history_file = os.path.abspath("data/history.csv")
    print("Saving to:", history_file)

    if not os.path.exists(history_file):
        return jsonify([])

    df = pd.read_csv(history_file)

    return jsonify(df.to_dict(orient="records"))


if __name__ == "__main__":
    app.run(debug=True)