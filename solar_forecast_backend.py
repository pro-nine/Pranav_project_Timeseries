import os
import json
from collections import deque

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from statsmodels.tsa.api import VAR
import statsmodels.api as sm

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from datetime import timedelta
from io import BytesIO
import base64


class SolarForecastPipeline:
    """
    Solar irradiance forecasting pipeline: VAR (classical multivariate),
    Random Forest, Histogram Gradient Boosting, and LSTM (deep sequence),
    all trained and evaluated on the SAME clear-sky-index target, the SAME
    chronological train/test split, and the SAME forecast horizon so that
    comparing their RMSE/MAE is a fair like-for-like comparison.

    Key modeling decisions (see accompanying write-up for full rationale):
      - Targets are the clear-sky index kt = raw / Clearsky, not raw
        GHI/DHI/DNI. This removes the deterministic diurnal/seasonal solar
        geometry signal, leaving only the stochastic cloud-attenuation
        component that is actually hard to forecast.
      - RF/GB/LSTM forecast the test period recursively (autoregressively),
        using only their OWN prior predictions as lag inputs - never ground
        truth from the test period. This matches what VAR does by
        construction and is required for the comparison to be valid.
      - Forecasts are converted back to physical units (W/m^2) via
        raw_pred = kt_pred * Clearsky before being stored in self.forecasts,
        so downstream plotting/evaluation code is unchanged.
    """

    def __init__(self, data_path, skiprows=2):
        self.data_path = data_path
        self.skiprows = skiprows
        self.s = 96  # samples/day at 15-minute resolution

        self.targets = ['GHI', 'DHI', 'DNI']
        self.clearsky_map = {'GHI': 'Clearsky GHI', 'DHI': 'Clearsky DHI', 'DNI': 'Clearsky DNI'}
        self.kt_clip = 1.5          # clear-sky index can slightly exceed 1 (cloud-edge enhancement)
        self.clearsky_floor = 10.0  # W/m^2; below this, treat as night and force kt = 0
        self.lstm_window = 8        # real sequence length fed to the LSTM (was 1 - not recurrent - before)

        self.data = None
        self.new_data = None
        self.train_data = None
        self.test_data = None

        self.feature_order = None
        self.scaler_ts = MinMaxScaler()

        self.X_train_scaled = None
        self.y_train_ts = {}
        self.X_train_lstm = None
        self.y_train_lstm = {}
        self.y_test_ts = {}

        self.var_model_fit = None
        self.rf_models = {}
        self.gb_models = {}   # HistGradientBoostingRegressor now, kept under the "GB" name/color for continuity
        self.lstm_models = {}

        self.forecasts = {}
        self.metrics = {}
        self.cv_report = None

    # ------------------------------------------------------------------
    # Loading & feature engineering
    # ------------------------------------------------------------------
    def load_and_inspect_data(self):
        self.data = pd.read_csv(self.data_path, skiprows=self.skiprows)

        features_to_keep = [
            'Year', 'Month', 'Day', 'Hour', 'Minute', 'Temperature',
            'Clearsky DHI', 'Clearsky DNI', 'Clearsky GHI', 'DHI', 'DNI', 'GHI'
        ]
        columns_to_drop = [c for c in self.data.columns if 'Unnamed' in c or c not in features_to_keep]
        self.data = self.data.drop(columns=columns_to_drop)

        self.data['Timestamp'] = pd.to_datetime(self.data[['Year', 'Month', 'Day', 'Hour', 'Minute']])
        self.data.set_index('Timestamp', inplace=True)
        self.data.drop(columns=['Year', 'Month', 'Day', 'Hour', 'Minute'], inplace=True)
        return self.data

    def compute_clear_sky_index(self):
        """kt = raw / Clearsky, forced to 0 at night, clipped to [0, kt_clip]."""
        for tgt in self.targets:
            csky_col = self.clearsky_map[tgt]
            kt = self.data[tgt] / self.data[csky_col].replace(0, np.nan)
            kt = kt.where(self.data[csky_col] > self.clearsky_floor, 0.0)
            kt = kt.clip(lower=0.0, upper=self.kt_clip)
            self.data[f'kt_{tgt}'] = kt
        return self.data

    def preprocess_data(self, correlation_threshold=None):
        """
        Builds lag/rolling/cyclical features on the clear-sky index series.
        `correlation_threshold` is accepted (unused) for backward
        compatibility with older callers - correlation-based feature
        selection was removed because it was gating VAR's endogenous
        variable set with a heuristic threshold rather than a deliberate
        choice, and (worse) could admit Clearsky GHI/DNI/DHI themselves as
        endogenous VAR variables, when they are perfectly known in advance
        and belong on the exogenous side, not forecast alongside GHI/DHI/DNI.
        VAR here operates explicitly on kt_GHI/kt_DHI/kt_DNI only.
        """
        if 'kt_GHI' not in self.data.columns:
            self.compute_clear_sky_index()

        df = self.data.copy()
        for tgt in self.targets:
            col = f'kt_{tgt}'
            shifted = df[col].shift(1)  # strictly-past value, so rolling3 below never includes "now"
            df[f'{col}_lag1'] = shifted
            df[f'{col}_lag3'] = df[col].shift(3)
            df[f'{col}_rolling3'] = shifted.rolling(window=3).mean()

        doy = df.index.dayofyear.values.astype(float)
        df['doy_sin'] = np.sin(2 * np.pi * doy / 365.25)
        df['doy_cos'] = np.cos(2 * np.pi * doy / 365.25)
        hour_frac = (df.index.hour + df.index.minute / 60).values.astype(float)
        df['hour_sin'] = np.sin(2 * np.pi * hour_frac / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour_frac / 24)

        self.new_data = df
        self.feature_order = []
        for tgt in self.targets:
            self.feature_order += [f'kt_{tgt}_lag1', f'kt_{tgt}_lag3', f'kt_{tgt}_rolling3']
        self.feature_order += ['doy_sin', 'doy_cos', 'hour_sin', 'hour_cos']
        return self.new_data

    # ------------------------------------------------------------------
    # Splitting & ML data prep
    # ------------------------------------------------------------------
    def split_data(self, train_end='2017-11-30 10:00:00', test_end='2017-12-31 23:00:00'):
        """
        ONE chronological split, used by every model (VAR, RF, GB, LSTM).
        Previously VAR used this date-cutoff split while RF/GB/LSTM used an
        unrelated 80/20 split of a different length - that mismatch is what
        made the model-comparison chart invalid.
        """
        df = self.new_data.dropna()
        train_end_dt = pd.to_datetime(train_end)
        test_end_dt = pd.to_datetime(test_end)

        self.train_data = df.loc[:train_end_dt]
        self.test_data = df.loc[train_end_dt + timedelta(minutes=15):test_end_dt]

        if len(self.train_data) <= self.lstm_window:
            raise ValueError(
                f"Training window has only {len(self.train_data)} rows after "
                f"dropna(), which is <= lstm_window={self.lstm_window}. "
                f"Use an earlier test_end/train_end or a smaller lstm_window."
            )
        if len(self.test_data) == 0:
            raise ValueError("Test window is empty - check train_end/test_end.")

        self.y_test_ts = {tgt: self.test_data[tgt] for tgt in self.targets}
        return self.train_data, self.test_data

    def prepare_ml_data(self):
        X_train = self.train_data[self.feature_order]
        self.scaler_ts.fit(X_train)
        self.X_train_scaled = self.scaler_ts.transform(X_train)
        self.y_train_ts = {tgt: self.train_data[f'kt_{tgt}'] for tgt in self.targets}

        T = self.lstm_window
        n_feat = len(self.feature_order)
        n = self.X_train_scaled.shape[0]

        X_lstm = np.zeros((n - T, T, n_feat))
        y_lstm = {tgt: np.zeros(n - T) for tgt in self.targets}
        for i in range(T, n):
            X_lstm[i - T] = self.X_train_scaled[i - T:i]
            for tgt in self.targets:
                y_lstm[tgt][i - T] = self.y_train_ts[tgt].iloc[i]

        self.X_train_lstm = X_lstm
        self.y_train_lstm = y_lstm

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train_var_model(self, max_lag=50):
        kt_cols = [f'kt_{t}' for t in self.targets]
        model = VAR(self.train_data[kt_cols])
        max_lag_order = max(1, min(max_lag, len(self.train_data) // 4))

        # select_order computes AIC/BIC/FPE/HQIC in one pass instead of
        # manually re-fitting the model in a Python loop for every candidate lag.
        selected = model.select_order(max_lag_order)
        best_lag = selected.aic or 1
        best_lag = max(1, best_lag)

        self.var_model_fit = model.fit(maxlags=best_lag)
        return self.var_model_fit

    def train_rf_gb_models(self, use_tuned=True, targets=None):
        targets = targets or self.targets
        for tgt in targets:
            if use_tuned:
                rf = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=10,
                                            random_state=42, n_jobs=-1)
                gb = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=5,
                                                    l2_regularization=0.1, random_state=42)
            else:
                rf = RandomForestRegressor(random_state=42, n_jobs=-1)
                gb = HistGradientBoostingRegressor(random_state=42)

            rf.fit(self.X_train_scaled, self.y_train_ts[tgt])
            gb.fit(self.X_train_scaled, self.y_train_ts[tgt])
            self.rf_models[tgt] = rf
            self.gb_models[tgt] = gb
        return self.rf_models, self.gb_models

    def train_lstm_model(self, epochs=80, batch_size=48, patience=5, targets=None):
        targets = targets or self.targets
        T = self.lstm_window
        n_feat = self.X_train_lstm.shape[2]

        for tgt in targets:
            m = Sequential([
                LSTM(50, return_sequences=True, input_shape=(T, n_feat)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(1)
            ])
            m.compile(optimizer='adam', loss='mean_squared_error')
            early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
            m.fit(self.X_train_lstm, self.y_train_lstm[tgt], epochs=epochs, batch_size=batch_size,
                  validation_split=0.2, verbose=0, callbacks=[early_stop])
            self.lstm_models[tgt] = m
        return self.lstm_models

    # ------------------------------------------------------------------
    # Forecasting (recursive / autoregressive for RF, GB, LSTM)
    # ------------------------------------------------------------------
    @staticmethod
    def _lag_feats(buf):
        lag1 = buf[-1]
        lag3 = buf[0] if len(buf) == 3 else buf[-1]
        roll3 = float(np.mean(buf))
        return lag1, lag3, roll3

    def _recursive_forecast_ml(self):
        """
        Walks forward through test_data.index ONCE. At every step, each of
        RF/GB/LSTM predicts using only its OWN prior predictions as lag
        inputs (three independent autoregressive histories) - never the
        actual test-period values. This is what makes the forecast genuinely
        multi-step-ahead, matching VAR, instead of a repeated one-step
        prediction with the answers for the recent past supplied for free.
        """
        T = self.lstm_window
        kt_state = {
            kind: {tgt: deque(self.train_data[f'kt_{tgt}'].values[-3:], maxlen=3) for tgt in self.targets}
            for kind in ['RF', 'GB', 'LSTM']
        }
        lstm_buf = deque(self.X_train_scaled[-T:], maxlen=T)
        raw_preds = {kind: {tgt: [] for tgt in self.targets} for kind in ['RF', 'GB', 'LSTM']}

        for ts in self.test_data.index:
            doy = ts.dayofyear
            hour_frac = ts.hour + ts.minute / 60
            cyc = [np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
                   np.sin(2 * np.pi * hour_frac / 24), np.cos(2 * np.pi * hour_frac / 24)]

            for kind in ['RF', 'GB']:
                feat = []
                for tgt in self.targets:
                    feat += list(self._lag_feats(kt_state[kind][tgt]))
                feat += cyc
                x_scaled = self.scaler_ts.transform(pd.DataFrame([feat], columns=self.feature_order))
                bank = self.rf_models if kind == 'RF' else self.gb_models
                for tgt in self.targets:
                    pred_kt = float(bank[tgt].predict(x_scaled)[0])
                    raw_preds[kind][tgt].append(pred_kt)
                    kt_state[kind][tgt].append(pred_kt)

            feat = []
            for tgt in self.targets:
                feat += list(self._lag_feats(kt_state['LSTM'][tgt]))
            feat += cyc
            x_scaled = self.scaler_ts.transform(pd.DataFrame([feat], columns=self.feature_order))[0]
            lstm_buf.append(x_scaled)
            window = np.array(lstm_buf).reshape(1, T, len(self.feature_order))
            for tgt in self.targets:
                pred_kt = float(self.lstm_models[tgt].predict(window, verbose=0)[0, 0])
                raw_preds['LSTM'][tgt].append(pred_kt)
                kt_state['LSTM'][tgt].append(pred_kt)

        for kind in ['RF', 'GB', 'LSTM']:
            df = pd.DataFrame(index=self.test_data.index)
            for tgt in self.targets:
                kt_series = pd.Series(raw_preds[kind][tgt], index=self.test_data.index)
                csky = self.test_data[self.clearsky_map[tgt]]
                df[tgt] = (kt_series * csky).clip(lower=0)
            self.forecasts[kind] = df

    def generate_forecasts(self, alpha=0.05):
        self.forecasts = {}
        kt_cols = [f'kt_{t}' for t in self.targets]

        forecast_steps = len(self.test_data)
        last_k = self.train_data[kt_cols].iloc[-self.var_model_fit.k_ar:]
        f_mean, f_low, f_up = self.var_model_fit.forecast_interval(
            y=last_k.values, steps=forecast_steps, alpha=alpha
        )
        kt_var = pd.DataFrame(f_mean, index=self.test_data.index, columns=kt_cols)

        var_df = pd.DataFrame(index=self.test_data.index)
        for tgt in self.targets:
            csky = self.test_data[self.clearsky_map[tgt]]
            var_df[tgt] = (kt_var[f'kt_{tgt}'] * csky).clip(lower=0)
        self.forecasts['VAR'] = var_df

        self._recursive_forecast_ml()
        return self.forecasts

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate_models(self):
        self.metrics = {}
        for model_name in ['VAR', 'RF', 'GB', 'LSTM']:
            if model_name not in self.forecasts:
                continue
            for tgt in self.targets:
                if tgt not in self.forecasts[model_name].columns:
                    continue
                actual = self.test_data[tgt]
                pred = self.forecasts[model_name][tgt]
                actual_aligned, pred_aligned = actual.align(pred, join='inner')
                key = f'{model_name}_{tgt}'
                self.metrics[key] = {
                    'MAE': float(mean_absolute_error(actual_aligned, pred_aligned)),
                    'MSE': float(mean_squared_error(actual_aligned, pred_aligned)),
                    'RMSE': float(np.sqrt(mean_squared_error(actual_aligned, pred_aligned)))
                }
        return self.metrics

    # ------------------------------------------------------------------
    # Rolling-origin cross-validation (methodology validation, not deployment)
    # ------------------------------------------------------------------
    def run_time_series_cv(self, n_splits=5, max_lag=50, rf_gb_kwargs=None,
                            lstm_epochs=80, lstm_batch_size=48, lstm_patience=5, verbose=True):
        """
        Retrains all 4 models from scratch on n_splits expanding-window
        folds (sklearn.TimeSeriesSplit) and reports per-fold + aggregated
        RMSE/MAE. A single winter train/test cut doesn't tell you how the
        model behaves in other seasons; this does.

        This is a validation REPORT for write-ups, not the artifact you
        deploy - call save_models() after a normal single-split
        train_and_save.py run for that. Calling this mutates
        self.train_data/test_data/rf_models/etc. as a side effect of
        retraining each fold, so call it BEFORE your final production fit,
        not after, or refit the production split again afterward.
        """
        tss = TimeSeriesSplit(n_splits=n_splits)
        full = self.new_data.dropna()
        idx = full.index

        fold_rows = []
        for fold, (train_idx, test_idx) in enumerate(tss.split(full), start=1):
            train_end = idx[train_idx[-1]]
            test_end = idx[test_idx[-1]]
            if verbose:
                print(f"[CV fold {fold}/{n_splits}] train_end={train_end}  test_end={test_end}  "
                      f"n_train={len(train_idx)}  n_test={len(test_idx)}")

            self.rf_models, self.gb_models, self.lstm_models = {}, {}, {}
            self.var_model_fit = None

            self.split_data(train_end=str(train_end), test_end=str(test_end))
            self.prepare_ml_data()
            self.train_var_model(max_lag=max_lag)
            self.train_rf_gb_models(**(rf_gb_kwargs or {}))
            self.train_lstm_model(epochs=lstm_epochs, batch_size=lstm_batch_size, patience=lstm_patience)
            self.generate_forecasts()
            fold_metrics = self.evaluate_models()

            for key, m in fold_metrics.items():
                fold_rows.append({'fold': fold, 'model_target': key, **m})

        cv_df = pd.DataFrame(fold_rows)
        summary = cv_df.groupby('model_target')[['MAE', 'MSE', 'RMSE']].agg(['mean', 'std'])
        self.cv_report = {'folds': cv_df, 'summary': summary}
        return self.cv_report

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_models(self, model_dir='saved_models'):
        os.makedirs(model_dir, exist_ok=True)

        if self.var_model_fit is not None:
            self.var_model_fit.save(os.path.join(model_dir, 'var_model.pkl'))

        joblib.dump(self.rf_models, os.path.join(model_dir, 'rf_models.joblib'))
        joblib.dump(self.gb_models, os.path.join(model_dir, 'gb_models.joblib'))
        joblib.dump(self.scaler_ts, os.path.join(model_dir, 'scaler_ts.joblib'))

        for tgt, model in self.lstm_models.items():
            model.save(os.path.join(model_dir, f'lstm_{tgt}.keras'))

        joblib.dump({
            'data': self.data,
            'new_data': self.new_data,
            'train_data': self.train_data,
            'test_data': self.test_data,
            'y_test_ts': self.y_test_ts,
            'X_train_scaled': self.X_train_scaled,
            'forecasts': self.forecasts,
        }, os.path.join(model_dir, 'artifacts.joblib'))

        config = {
            'targets': self.targets, 'clearsky_map': self.clearsky_map,
            'feature_order': self.feature_order, 'lstm_window': self.lstm_window,
            'kt_clip': self.kt_clip, 'clearsky_floor': self.clearsky_floor, 's': self.s,
        }
        with open(os.path.join(model_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

        if self.metrics:
            with open(os.path.join(model_dir, 'metrics.json'), 'w') as f:
                json.dump(self.metrics, f, indent=2)

    def load_models(self, model_dir='saved_models'):
        required = ['rf_models.joblib', 'gb_models.joblib', 'scaler_ts.joblib',
                    'artifacts.joblib', 'config.json']
        missing = [f for f in required if not os.path.exists(os.path.join(model_dir, f))]
        if missing:
            raise FileNotFoundError(
                f"Missing model artifacts in '{model_dir}': {missing}. "
                f"Run train_and_save.py first to populate this directory."
            )

        with open(os.path.join(model_dir, 'config.json')) as f:
            config = json.load(f)
        self.targets = config['targets']
        self.clearsky_map = config['clearsky_map']
        self.feature_order = config['feature_order']
        self.lstm_window = config['lstm_window']
        self.kt_clip = config['kt_clip']
        self.clearsky_floor = config['clearsky_floor']
        self.s = config['s']

        var_path = os.path.join(model_dir, 'var_model.pkl')
        if os.path.exists(var_path):
            self.var_model_fit = sm.load(var_path)

        self.rf_models = joblib.load(os.path.join(model_dir, 'rf_models.joblib'))
        self.gb_models = joblib.load(os.path.join(model_dir, 'gb_models.joblib'))
        self.scaler_ts = joblib.load(os.path.join(model_dir, 'scaler_ts.joblib'))

        self.lstm_models = {}
        for tgt in self.targets:
            p = os.path.join(model_dir, f'lstm_{tgt}.keras')
            if os.path.exists(p):
                self.lstm_models[tgt] = load_model(p)

        artifacts = joblib.load(os.path.join(model_dir, 'artifacts.joblib'))
        self.data = artifacts['data']
        self.new_data = artifacts['new_data']
        self.train_data = artifacts['train_data']
        self.test_data = artifacts['test_data']
        self.y_test_ts = artifacts['y_test_ts']
        self.X_train_scaled = artifacts['X_train_scaled']
        # Older artifacts.joblib (saved before forecast-persistence was added)
        # won't have this key - forecasts will be empty and the caller needs
        # to regenerate once (slow: a full recursive pass over the test period).
        self.forecasts = artifacts.get('forecasts', {})

        metrics_path = os.path.join(model_dir, 'metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                self.metrics = json.load(f)
        return self

    # ------------------------------------------------------------------
    # Plotting (interface unchanged - all operate on raw W/m^2 units)
    # ------------------------------------------------------------------
    def _fig_to_base64(self, fig):
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=100, facecolor='#2D3142', bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        return image_base64

    def get_data_overview(self, target='GHI', view_type='raw'):
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor('#2D3142')
        ax.set_facecolor('#4F5D75')

        if target not in self.data.columns:
            ax.text(0.5, 0.5, f'{target} not found', ha='center', va='center', color='#FFFFFF')
            return self._fig_to_base64(fig)

        if view_type == 'raw':
            ax.plot(self.data.index, self.data[target], color='#0D9276', linewidth=1)
            ax.set_title(f'{target} Raw Data', color='#FFF6E9', fontweight='bold')
        elif view_type == 'daily':
            hourly = self.data.groupby(self.data.index.hour)[target].mean()
            ax.plot(hourly.index, hourly.values, color='#0D9276', linewidth=0.8, marker='o')
            ax.set_title(f'{target} Daily Cycle', color='#FFF6E9', fontweight='bold')
        elif view_type == 'monthly':
            monthly = self.data[target].resample('M').mean()
            ax.bar(monthly.index, monthly.values, color='#0D9276', width=10)
            ax.set_title(f'{target} Monthly Average', color='#FFF6E9', fontweight='bold')
        elif view_type == 'yearly':
            daily = self.data[target].resample('D').mean()
            ax.plot(daily.index, daily.values, color='#0D9276', linewidth=1, marker='^')
            ax.set_title(f'{target} Yearly Cycle', color='#FFF6E9', fontweight='bold')
            ax.set_xlabel('Hour of Day', color='#FFFFFF')

        ax.set_ylabel(target, color='#FFFFFF')
        ax.tick_params(colors='#BFC0C0')
        ax.grid(alpha=0.2, color='#BFC0C0')
        for spine in ax.spines.values():
            spine.set_color('#BFC0C0')

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def generate_forecast_window(self, target='GHI', date='2017-11-30', hours=6):
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor('#2D3142')
        ax.set_facecolor('#4F5D75')

        start_dt = pd.to_datetime(date)
        end_dt = start_dt + timedelta(hours=hours - 1)

        actual = self.test_data[target]
        actual_window = actual.loc[start_dt:end_dt]
        ax.plot(actual_window.index, actual_window.values, label='Actual', color='#FFFFFF', linewidth=1.8)

        colors = {'VAR': '#EF8354', 'RF': '#66bb6a', 'GB': '#4fc3f7', 'LSTM': '#ffa726'}
        for model_name in ['VAR', 'RF', 'GB', 'LSTM']:
            if model_name not in self.forecasts or target not in self.forecasts[model_name].columns:
                continue
            pred_window = self.forecasts[model_name][target].loc[start_dt:end_dt]
            ax.plot(pred_window.index, pred_window.values, label=model_name, color=colors[model_name], linewidth=1.8)

        ax.set_title(f'{target} Forecast - {hours}h Window', color='#EF8354', fontweight='bold')
        ax.set_xlabel('Time', color='#FFFFFF')
        ax.set_ylabel(target, color='#FFFFFF')
        ax.legend(facecolor='#4F5D75', edgecolor='#BFC0C0', labelcolor='#FFFFFF')
        ax.tick_params(colors='#BFC0C0')
        ax.grid(alpha=0.2, color='#BFC0C0')
        for spine in ax.spines.values():
            spine.set_color('#BFC0C0')

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def compare_models_for_date(self, target='GHI', date='2017-11-30'):
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor('#2D3142')
        ax.set_facecolor('#4F5D75')

        date_dt = pd.to_datetime(date)
        end_dt = date_dt + timedelta(days=1)

        if target not in self.test_data.columns:
            raise ValueError(f"Target '{target}' not present in test_data")

        actual = self.test_data[target].loc[date_dt:end_dt]
        if len(actual) == 0:
            raise ValueError("No actual data found for selected date")
        ax.plot(actual.index, actual.values, label='Actual', color='#FFFFFF', linewidth=1.2)

        colors = {'VAR': '#EF8354', 'RF': '#66bb6a', 'GB': '#4fc3f7', 'LSTM': '#ffa726'}
        for model_name, color in colors.items():
            if model_name not in self.forecasts or target not in self.forecasts[model_name]:
                continue
            pred_day = self.forecasts[model_name][target].loc[date_dt:end_dt]
            if len(pred_day) == 0:
                continue
            ax.plot(pred_day.index, pred_day.values, linestyle='-', color=color, linewidth=1.1, label=model_name)

        ax.set_title(f'{target} Model Comparison - {date}', color='#EF8354', fontweight='bold')
        ax.set_xlabel('Time', color='#FFFFFF')
        ax.set_ylabel(target, color='#FFFFFF')
        ax.tick_params(colors='#BFC0C0')
        ax.grid(alpha=0.25, color='#BFC0C0')
        for spine in ax.spines.values():
            spine.set_color('#BFC0C0')
        ax.legend(facecolor='#4F5D75', edgecolor='#BFC0C0', labelcolor='#FFFFFF')

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def get_preprocessing_correlation(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('#2D3142')

        corr = self.train_data.corr()  # train-only, avoids leaking test-period correlations
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn', center=0, square=True,
                    linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)

        ax.set_title('Correlation Heatmap (train only)', color='#EF8354', fontweight='bold')
        plt.xticks(color='#FFFFFF', rotation=45, ha='right')
        plt.yticks(color='#FFFFFF', rotation=0)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(colors='#FFFFFF')

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def get_training_metrics(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        fig.patch.set_facecolor('#2D3142')

        models, rmse, mae, colors_list = [], [], [], []
        color_map = {'VAR': '#EF8354', 'RF': '#66bb6a', 'GB': '#4fc3f7', 'LSTM': '#ffa726'}

        for key, met in self.metrics.items():
            model_name, tgt = key.split("_")
            if tgt not in self.targets:
                continue
            models.append(key)
            rmse.append(met['RMSE'])
            mae.append(met['MAE'])
            colors_list.append(color_map[model_name])

        ax1.bar(models, rmse, color=colors_list, edgecolor='#BFC0C0')
        ax1.set_title("RMSE", color='#EF8354')
        ax1.tick_params(colors='#BFC0C0')
        ax1.set_facecolor('#4F5D75')
        ax1.grid(axis='y', alpha=0.2, color='#BFC0C0')
        for s in ax1.spines.values():
            s.set_color('#BFC0C0')

        ax2.bar(models, mae, color=colors_list, edgecolor='#BFC0C0')
        ax2.set_title("MAE", color='#EF8354')
        ax2.tick_params(colors='#BFC0C0')
        ax2.set_facecolor('#4F5D75')
        ax2.grid(axis='y', alpha=0.2, color='#BFC0C0')
        for s in ax2.spines.values():
            s.set_color('#BFC0C0')

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def compare_model_performance(self):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.patch.set_facecolor('#2D3142')

        colors_map = {'VAR': '#EF8354', 'RF': '#66bb6a', 'GB': '#4fc3f7', 'LSTM': '#ffa726'}

        for idx, component in enumerate(self.targets):
            ax = axes[idx]
            ax.set_facecolor('#4F5D75')

            models, rmse_vals, bar_colors = [], [], []
            for key, metrics in self.metrics.items():
                if key.endswith('_' + component):
                    model_name = key.split('_')[0]
                    models.append(model_name)
                    rmse_vals.append(metrics['RMSE'])
                    bar_colors.append(colors_map.get(model_name, '#EF8354'))

            if models:
                ax.bar(models, rmse_vals, color=bar_colors, edgecolor='#BFC0C0')

            ax.set_ylabel('RMSE', color='#FFFFFF', fontweight='bold')
            ax.set_title(f'{component}', color='#EF8354', fontweight='bold')
            ax.tick_params(colors='#BFC0C0')
            ax.grid(axis='y', alpha=0.2, color='#BFC0C0')
            for spine in ax.spines.values():
                spine.set_color('#BFC0C0')

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def plot_forecast_comparison(self, models=('VAR', 'RF', 'GB', 'LSTM'), component='GHI',
                                  start_time=None, end_time=None):
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor('#2D3142')
        ax.set_facecolor('#4F5D75')

        actual_data = self.test_data[component]
        if start_time and end_time:
            start_dt, end_dt = pd.to_datetime(start_time), pd.to_datetime(end_time)
            actual_data = actual_data.loc[start_dt:end_dt]

        ax.plot(actual_data.index, actual_data.values, label=f'Actual {component}',
                linewidth=2, alpha=0.8, color='#FFFFFF')

        colors = {'VAR': '#EF8354', 'RF': '#66bb6a', 'GB': '#4fc3f7', 'LSTM': '#ffa726'}
        linestyles = {'VAR': '--', 'RF': '-.', 'GB': ':', 'LSTM': '-'}

        for model_name in models:
            if model_name not in self.forecasts:
                continue
            pred_data = self.forecasts[model_name][component]
            if start_time and end_time:
                pred_data = pred_data.loc[start_dt:end_dt]
            ax.plot(pred_data.index, pred_data.values, label=model_name,
                    linestyle=linestyles.get(model_name, '-'), linewidth=2, alpha=0.8,
                    color=colors.get(model_name, '#EF8354'))

        ax.set_title(f'Model Predictions vs Actual {component}', color='#EF8354', fontweight='bold')
        ax.set_xlabel('Time', color='#FFFFFF')
        ax.set_ylabel(f'{component} (W/m^2)', color='#FFFFFF')
        ax.legend(facecolor='#4F5D75', edgecolor='#BFC0C0', labelcolor='#FFFFFF')
        ax.grid(alpha=0.2, color='#BFC0C0')
        ax.tick_params(colors='#BFC0C0')
        for spine in ax.spines.values():
            spine.set_color('#BFC0C0')

        plt.tight_layout()
        return self._fig_to_base64(fig)
