from flask import Flask, render_template, request
import numpy as np
import pickle

# Initialize Flask App
app = Flask(__name__)

# Load Model (Ensure model.pkl exists in the same directory)
try:
    model = pickle.load(open("model.pkl", "rb"))
except Exception as e:
    print("Error loading model:", e)
    model = None

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            # Get Form Data
            temp = float(request.form["temperature"])
            rainfall = float(request.form["rainfall"])
            fertilizer = float(request.form["fertilizer"])

            # Check if Model is Loaded
            if model:
                # Reshape Input & Predict
                input_features = np.array([[temp, rainfall, fertilizer]])
                prediction = round(model.predict(input_features)[0], 2)
            else:
                error = "⚠️ Model is not loaded."

        except Exception as e:
            error = f"⚠️ Error: {str(e)}"

    return render_template("index.html", prediction=prediction, error=error)

# Run the App
if __name__ == "__main__":
    app.run(debug=True)
