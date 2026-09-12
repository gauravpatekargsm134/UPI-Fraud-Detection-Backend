import os
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

# Create Gemini client
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def generate_explanation(prediction_result):

    rf = prediction_result["random_forest"]
    xgb = prediction_result["xgboost"]
    ae = prediction_result["autoencoder"]
    shap = prediction_result["shap"][:5]

    shap_text = ""

    for item in shap:
        direction = (
            "increased fraud risk"
            if item["contribution"] > 0
            else "reduced fraud risk"
        )

        shap_text += (
            f"- {item['feature']} "
            f"(Contribution: {item['contribution']:.4f}) "
            f"{direction}\n"
        )

    prompt = f"""
You are an Explainable AI assistant for a UPI Fraud Detection System.

Model Outputs:

Random Forest:
{rf}

XGBoost:
{xgb}

Autoencoder:
{ae}

Top SHAP Features:
{shap_text}

Generate:

1. A short explanation (4-5 lines)
2. Main reasons
3. Risk Level (Low / Medium / High)
4. Recommendation

Return the response in proper markdown.
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )

    return response.text