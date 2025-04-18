from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
import google.generativeai as genai

app = Flask(__name__)

# Load ML model and data
model = joblib.load("crop_prob_model.pkl")
label_mapping = joblib.load("crop_label_mapping.pkl")
fertilizer_df = pd.read_csv("fertilizer.csv").set_index("Crop")

# Gemini config
os.environ["GOOGLE_API_KEY"] = "AIzaSyDD8QW1BggDVVMLteDygHCHrD6Ff9Dy0e8"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


def validate_inputs(n, p, k, temp, humidity, ph, rainfall):
    return (
        3.5 <= ph <= 9 and 0 <= rainfall <= 500
    )


def gemini_fertilizer_advice(crop, n, p, k, ideal_n, ideal_p, ideal_k):
    def compare(val, ideal, name):
        if val < ideal:
            return f"{name} is low ({val} < {ideal})"
        elif val > ideal:
            return f"{name} is high ({val} > {ideal})"
        else:
            return f"{name} is optimal ({val} = {ideal})"

    n_status = compare(n, ideal_n, "Nitrogen")
    p_status = compare(p, ideal_p, "Phosphorus")
    k_status = compare(k, ideal_k, "Potassium")

    prompt = f"""
    A farmer wants to grow {crop}.
    Ideal NPK values: Nitrogen: {ideal_n}, Phosphorus: {ideal_p}, Potassium: {ideal_k}.
    Current soil values: Nitrogen: {n} → {n_status}, Phosphorus: {p} → {p_status}, Potassium: {k} → {k_status}.
    Based on this, suggest what nutrients are deficient or excessive and recommend suitable fertilizers.
    Keep it concise within 50 words. Use Indian agricultural context. No headings or markdown.
    """

    model = genai.GenerativeModel("gemini-2.0-flash")
    return model.generate_content(prompt).text.strip()


@app.route('/predict_crops', methods=['POST'])
def predict_crops():
    data = request.get_json()
    try:
        n = float(data["n"])
        p = float(data["p"])
        k = float(data["k"])
        temp = float(data["temp"])
        humidity = float(data["humidity"])
        ph = float(data["ph"])
        rainfall = float(data["rainfall"])
    except (KeyError, ValueError):
        return jsonify({"error": "Missing or invalid input values"}), 400

    if not validate_inputs(n, p, k, temp, humidity, ph, rainfall):
        return jsonify({"error": "❌ Invalid input values. Check ranges."}), 400

    features = np.array([[n, p, k, temp, humidity, ph, rainfall]])
    probs = model.predict_proba(features)[0]
    top5_indices = np.argsort(probs)[::-1][:5]
    top5 = [
        {"crop": label_mapping[i], "confidence": round(probs[i] * 100, 2)}
        for i in top5_indices
    ]

    return jsonify({"predictions": top5}), 200

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API is running"}), 200


@app.route('/fertilizer_advice', methods=['POST'])
def fertilizer_advice():
    data = request.get_json()
    try:
        crop = data["crop"]
        n = float(data["n"])
        p = float(data["p"])
        k = float(data["k"])
    except (KeyError, ValueError):
        return jsonify({"error": "Missing or invalid input values"}), 400

    try:
        ideal = fertilizer_df.loc[crop]
        advice = gemini_fertilizer_advice(crop, n, p, k, ideal["N"], ideal["P"], ideal["K"])
        return jsonify({"crop": crop, "advice": advice}), 200
    except KeyError:
        return jsonify({"error": f"No fertilizer data available for '{crop}'"}), 404


if __name__ == '__main__':
    app.run(debug=True)
