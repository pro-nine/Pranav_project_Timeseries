from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import threading
from solar_forecast_backend import SolarForecastPipeline

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

pipeline = None
train_progress = {"VAR": False, "RF": False, "GB": False, "LSTM": False}
train_logs = []
training_lock = threading.Lock()

# --------------------------------------------------------------
#  HEALTH CHECK (Render needs this)
# --------------------------------------------------------------
@app.route("/health")
def health():
    return "OK", 200

@app.route("/")
def index():
    return render_template("index.html")

# --------------------------------------------------------------
#  FILE UPLOAD
# --------------------------------------------------------------
@app.route("/api/upload", methods=["POST"])
def upload_file():
    try:
        file = request.files.get("file")
        if file is None:
            return jsonify({"error": "No file uploaded"}), 400

        if not file.filename.endswith(".csv"):
            return jsonify({"error": "Only CSV allowed"}), 400

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], "data.csv")
        file.save(filepath)

        return jsonify({"success": True, "message": "File uploaded"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------------------------------------------------------------
#  TRAIN
# --------------------------------------------------------------
@app.route("/api/train", methods=["POST"])
def train_models():
    global pipeline, train_progress, train_logs

    with training_lock:
        try:
            data = request.json
            selected_models = data.get("models", ["VAR", "RF", "GB", "LSTM"])
            train_end = data.get("trainEnd", "2017-11-30 10:00:00")
            test_end = data.get("testEnd", "2017-12-31 23:00:00")

            filepath = os.path.join(app.config["UPLOAD_FOLDER"], "data.csv")
            if not os.path.exists(filepath):
                return jsonify({"error": "Upload file first"}), 400

            # Reset progress
            train_progress = {"VAR": False, "RF": False, "GB": False, "LSTM": False}
            train_logs.clear()

            pipeline = SolarForecastPipeline(filepath)
            train_logs.append("Data loaded")

            pipeline.load_and_inspect_data()
            pipeline.preprocess_data(correlation_threshold=0.7)
            pipeline.split_data(train_end=train_end, test_end=test_end)
            pipeline.prepare_ml_data(test_size=0.2, random_state=42)

            if "VAR" in selected_models:
                pipeline.train_var_model(max_lag=50)
                train_progress["VAR"] = True
                train_logs.append("VAR done")

            if "RF" in selected_models or "GB" in selected_models:
                pipeline.train_rf_gb_models(use_tuned=True, targets=["GHI", "DHI", "DNI"])
                if "RF" in selected_models:
                    train_progress["RF"] = True
                if "GB" in selected_models:
                    train_progress["GB"] = True
                train_logs.append("RF/GB done")

            if "LSTM" in selected_models:
                pipeline.train_lstm_model(epochs=80, batch_size=48, patience=5, targets=["GHI", "DHI", "DNI"])
                train_progress["LSTM"] = True
                train_logs.append("LSTM done")

            pipeline.generate_forecasts()
            pipeline.evaluate_models()
            train_logs.append("Forecast ready")

            return jsonify({"success": True})
        except Exception as e:
            train_logs.append(f"Error: {str(e)}")
            return jsonify({"error": str(e)}), 500

# --------------------------------------------------------------
#  TRAIN STATUS + LOGS
# --------------------------------------------------------------
@app.route("/api/train_status", methods=["GET"])
def train_status():
    return jsonify({"progress": train_progress})

@app.route("/api/train_logs", methods=["GET"])
def get_logs():
    return jsonify({"logs": train_logs})

# --------------------------------------------------------------
#  DATA OVERVIEW
# --------------------------------------------------------------
@app.route("/api/data_overview", methods=["GET"])
def data_overview():
    try:
        if pipeline is None:
            return jsonify({"error": "Train first"}), 400

        target = request.args.get("target", "GHI")
        view_type = request.args.get("type", "raw")

        image = pipeline.get_data_overview(target, view_type)
        return jsonify({"success": True, "image": image})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------------------------------------------------------------
#  FORECAST WINDOW
# --------------------------------------------------------------
@app.route("/api/forecast", methods=["GET"])
def forecast():
    try:
        if pipeline is None or not pipeline.forecasts:
            return jsonify({"error": "Train first"}), 400

        target = request.args.get("target", "GHI")
        date = request.args.get("date")
        window = int(request.args.get("window", 6))

        image = pipeline.generate_forecast_window(target, date, window)
        return jsonify({"success": True, "image": image})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------------------------------------------------------------
#  MODEL COMPARISON
# --------------------------------------------------------------
@app.route("/api/model_comparison", methods=["GET"])
def model_comparison():
    try:
        if pipeline is None or not pipeline.forecasts:
            return jsonify({"error": "Train first"}), 400

        target = request.args.get("target", "GHI")
        date = request.args.get("date")

        image = pipeline.compare_models_for_date(target, date)
        return jsonify({"success": True, "image": image})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------------------------------------------------------------
#  TRAIN METRICS
# --------------------------------------------------------------
@app.route("/api/get_training_metrics", methods=["GET"])
def training_metrics():
    try:
        image = pipeline.get_training_metrics()
        return jsonify({"success": True, "image": image})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------------------------------------------------------------
#  PERFORMANCE
# --------------------------------------------------------------
@app.route("/api/get_performance_comparison", methods=["GET"])
def perf_metrics():
    try:
        image = pipeline.compare_model_performance()
        return jsonify({"success": True, "image": image})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
