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
#  HEALTH CHECK 
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
# (In your flask_app.py file, replace the existing /api/train function)

@app.route("/api/train", methods=["POST"])
def train_models():
    global pipeline
    
    # 1. Prevent multiple training runs
    if training_lock.locked():
        return jsonify({"error": "Training is already in progress"}), 409
        
    try:
        # Define the function that contains the long-running ML process
        def run_training():
            global pipeline
            
            try:
                # Reset progress flags and logs
                for key in train_progress:
                    train_progress[key] = False
                train_logs.clear()
                train_logs.append("Training initiated in background...")
                
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], "data.csv")
                
                if not os.path.exists(filepath):
                    train_logs.append("ERROR: Data file not found.")
                    return

                # --- START OF LONG-RUNNING LOGIC ---
                pipeline = SolarForecastPipeline(filepath)
                train_logs.append("Pipeline initialized and data loading...")

                pipeline.load_and_inspect_data()
                pipeline.preprocess_data(correlation_threshold=0.7)
                pipeline.split_data(train_end='2017-11-30 10:00:00', test_end='2017-12-31 23:00:00')
                pipeline.prepare_ml_data(test_size=0.2, random_state=42)
                train_logs.append("Data preprocessed and prepared.")
                
                # VAR Model
                pipeline.train_var_model(max_lag=50)
                train_progress["VAR"] = True
                train_logs.append("VAR model trained.")
                
                # RF & GB Models
                pipeline.train_rf_gb_models(use_tuned=True)
                train_progress["RF"] = True
                train_progress["GB"] = True
                train_logs.append("RF and GB models trained.")

                # LSTM Model
                pipeline.train_lstm_model(epochs=80, batch_size=48, patience=5)
                train_progress["LSTM"] = True
                train_logs.append("LSTM model trained.")

                # Final steps
                pipeline.generate_forecasts()
                pipeline.evaluate_models()
                train_logs.append("All forecasts generated and evaluated. Process complete.")
                # --- END OF LONG-RUNNING LOGIC ---

            except Exception as e:
                # Log any error that happens during training
                error_msg = f"FATAL TRAINING ERROR: {str(e)}"
                train_logs.append(error_msg)
            finally:
                # Release the lock regardless of success or failure
                training_lock.release()

        # 2. Acquire lock (to show status is 'running') and start thread immediately
        training_lock.acquire()
        thread = threading.Thread(target=run_training)
        thread.start()

        # 3. Return success IMMEDIATELY (prevents Gunicorn timeout)
        return jsonify({"success": True, "message": "Training started in background. Poll /api/train_status for progress."})

    except Exception as e:
        # Handle exceptions that occur before the thread is started (e.g., file upload path error)
        if training_lock.locked():
             training_lock.release()
        return jsonify({"error": str(e)}), 500

# --------------------------------------------------------------
#  TRAIN STATUS + LOGS
# --------------------------------------------------------------
@app.route("/api/train_status", methods=["GET"])
def train_status():
    if training_lock.locked():
        status = "running"
    elif any(train_progress.values()):
        # Lock is released AND models ran: SUCCESS
        status = "completed"
    elif any("ERROR" in log.upper() for log in train_logs):
        # Lock is released AND logs show an error: FAILURE
        status = "failed"
    else:
        # Lock is released, no progress, no clear error in logs: IDLE/Not Started
        status = "idle" 
        
    return jsonify({
        "status": status,
        "progress": train_progress,
        "logs": train_logs
    })

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
