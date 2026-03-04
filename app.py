from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("HR_Attrition.pkl")

@app.route("/")
def home():
    return "HR Attrition Prediction API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_data = pd.DataFrame(data)

        prediction_prob = model.predict_proba(input_data)[0][1]
        prediction_class = int(model.predict(input_data)[0])

        return jsonify({
            "attrition_risk_score": round(float(prediction_prob), 4),
            "prediction": "Will Leave" if prediction_class == 1 else "Will Stay",
            "status": "Success"
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)