from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load dict from pickle
with open("model.pkl", "rb") as file:
    data = pickle.load(file)

# Extract model and scaler
model = data["model"]
scaler = data["scaler"]

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get features from form
        float_features = [float(x) for x in request.form.values()]
        features = np.array([float_features])   

        # Apply scaling
        features_scaled = scaler.transform(features)

        # Predict with XGBoost model
        prediction = model.predict(features_scaled)
        output = prediction[0]

        return render_template(
            "index.html",
            prediction_text=f"Predicted Concrete Strength: {output:.2f} MPa"
        )

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")



if __name__ == "__main__":
    app.run(debug=True)