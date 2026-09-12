import joblib

scaler = joblib.load("models/scaler.pkl")

def preprocess(df):

    # No encoder needed because the model was trained on numeric data
    processed = scaler.transform(df)

    return processed