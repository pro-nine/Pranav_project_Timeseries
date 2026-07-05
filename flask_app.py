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

# Directory holding models trained OFFLINE via train_and_save.py and
# committed to the repo. The deployed app loads from here on boot -
# it does not train on a live request. See train_and_save.py.
MODEL_DIR = os.environ.get('MODEL_DIR', 'saved_models')

pipeline = None
train_progress = {"VAR": False, "RF": False, "GB": False, "LSTM": False}
train_logs = []
training_lock = threading.Lock()


def _backfill_forecasts_in_background(p):
    """
    Only used for saved_models/ produced before forecast-persistence was
    added to save_models(). Runs the recursive forecast pass in a
    background thread so it never blocks app boot / health checks -
    for a full month at 15-min resolution this can take on the order of
    20 minutes. Once done, re-saves so this never has to run again.
    """
    def _run():
        train_logs.append(
            "artifacts.joblib predates forecast persistence - regenerating "
            "forecasts once in the background (this can take ~20 min for a "
            "full month of 15-min data). Forecast/comparison endpoints will "
            "return 'not ready' until this finishes."
        )
        try:
            p.generate_forecasts()
            p.evaluate_models()
            p.save_models(MODEL_DIR)
            for key in train_progress:
                train_progress[key] = True
            train_logs.append("Forecasts regenerated and re-saved - future boots will be fast.")
        except Exception as e:
            train_logs.append(f"FATAL: background forecast regeneration failed: {e}")

    threading.Thread(target=_run, daemon=True).start()


def load_pretrained_pipeline():
    """
    Runs once at process startup (module import time). Loads the
    pretrained models + precomputed forecasts from MODEL_DIR. Boot is
    fast (deserializing arrays/DataFrames, not retraining or
    re-forecasting) as long as saved_models/ was produced by the current
    save_models(), which persists self.forecasts alongside the models.
    """
    global pipeline
    artifacts_path = os.path.join(MODEL_DIR, 'artifacts.joblib')
    if not os.path.exists(artifacts_path):
        train_logs.append(
            f"No pretrained models found in '{MODEL_DIR}/'. "
            f"Run train_and_save.py locally and commit the output, "
            f"or POST to /api/upload + /api/train to train live."
        )
        return

    try:
        p = SolarForecastPipeline(data_path=None)
        p.load_models(MODEL_DIR)
        pipeline = p

        if p.forecasts:
            for key in train_progress:
                train_progress[key] = True
            train_logs.append(f"Loaded pretrained models + forecasts from '{MODEL_DIR}/' on startup.")
        else:
            _backfill_forecasts_in_background(p)
    except Exception as e:
        train_logs.append(f"FATAL: failed to load pretrained models: {e}")


load_pretrained_pipeline()

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
#  TRAIN (manual retrain path - offline train_and_save.py is preferred;
#  this exists for retraining on a freshly uploaded CSV without a
#  redeploy. NOTE: with the recursive multi-step forecast now used for
#  RF/GB/LSTM, a full retrain including forecast generation over a
#  month-long test window can take well over the ~20 min measured for
#  generate_forecasts() alone, plus LSTM/RF/GB fit time. It's
#  backgrounded so it won't block the request or the worker, but don't
#  expect "a few minutes" anymore.)
# --------------------------------------------------------------
@app.route("/api/train", methods=["POST"])
def train_models():
    global pipeline

    if training_lock.locked():
        return jsonify({"error": "Training is already in progress"}), 409

    try:
        def run_training():
            global pipeline

            try:
                for key in train_progress:
                    train_progress[key] = False
                train_logs.clear()
                train_logs.append("Training initiated in background...")

                filepath = os.path.join(app.config["UPLOAD_FOLDER"], "data.csv")

                if not os.path.exists(filepath):
                    train_logs.append("ERROR: Data file not found.")
                    return

                new_pipeline = SolarForecastPipeline(filepath)
                train_logs.append("Pipeline initialized, loading data...")

                new_pipeline.load_and_inspect_data()
                new_pipeline.compute_clear_sky_index()
                new_pipeline.preprocess_data()
                new_pipeline.split_data(train_end='2017-11-30 10:00:00', test_end='2017-12-31 23:00:00')
                new_pipeline.prepare_ml_data()
                train_logs.append("Clear-sky index computed, features prepared.")

                new_pipeline.train_var_model(max_lag=50)
                train_progress["VAR"] = True
                train_logs.append("VAR model trained (kt_GHI/kt_DHI/kt_DNI).")

                new_pipeline.train_rf_gb_models(use_tuned=True)
                train_progress["RF"] = True
                train_progress["GB"] = True
                train_logs.append("RF and HistGradientBoosting models trained.")

                new_pipeline.train_lstm_model(epochs=80, batch_size=48, patience=5)
                train_progress["LSTM"] = True
                train_logs.append("LSTM model trained.")

                train_logs.append("Generating recursive multi-step forecasts "
                                   "(this is the slow step - can take ~20 min "
                                   "for a full month at 15-min resolution)...")
                new_pipeline.generate_forecasts()
                new_pipeline.evaluate_models()
                train_logs.append("All forecasts generated and evaluated. Process complete.")

                # NOTE: on ephemeral-filesystem hosts (Render/Railway/Heroku
                # free-hobby tiers) this directory is wiped on the next
                # deploy/restart - it is NOT a substitute for committing
                # saved_models/ to git via train_and_save.py.
                new_pipeline.save_models(MODEL_DIR)
                train_logs.append(f"Models + forecasts saved to '{MODEL_DIR}/'.")

                pipeline = new_pipeline

            except Exception as e:
                error_msg = f"FATAL TRAINING ERROR: {str(e)}"
                train_logs.append(error_msg)
            finally:
                training_lock.release()

        training_lock.acquire()
        thread = threading.Thread(target=run_training, daemon=True)
        thread.start()

        return jsonify({"success": True, "message": "Training started in background. Poll /api/train_status for progress."})

    except Exception as e:
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
        status = "completed"
    elif any("ERROR" in log.upper() for log in train_logs):
        status = "failed"
    else:
        status = "idle"

    return jsonify({
        "status": status,
        "progress": train_progress,
        "logs": train_logs
    })

@app.route("/api/train_logs", methods=["GET"])
def get_logs():
    return jsonify({"logs": train_logs})

@app.route("/api/model_info", methods=["GET"])
def model_info():
    return jsonify({
        "pretrained_loaded": pipeline is not None and bool(pipeline.forecasts),
        "model_dir": MODEL_DIR,
        "metrics": pipeline.metrics if pipeline is not None else {}
    })

# --------------------------------------------------------------
#  DATA OVERVIEW
# --------------------------------------------------------------
@app.route("/api/data_overview", methods=["GET"])
def data_overview():
    try:
        if pipeline is None:
            return jsonify({"error": "No data loaded yet"}), 400

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
            return jsonify({"error": "Forecasts not ready yet - training or backfilling in the background"}), 400

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
            return jsonify({"error": "Forecasts not ready yet - training or backfilling in the background"}), 400

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
        if pipeline is None or not pipeline.metrics:
            return jsonify({"error": "Metrics not ready yet"}), 400
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
        if pipeline is None or not pipeline.metrics:
            return jsonify({"error": "Metrics not ready yet"}), 400
        image = pipeline.compare_model_performance()
        return jsonify({"success": True, "image": image})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
