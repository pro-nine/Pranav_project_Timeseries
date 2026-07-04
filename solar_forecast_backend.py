import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, kpss, pacf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime, timedelta
import joblib
import os
from io import BytesIO
import base64


class SolarForecastPipeline:
    def __init__(self, data_path, skiprows=2):
        self.data_path = data_path
        self.skiprows = skiprows
        self.s = 96
        
        self.data = None
        self.new_data = None
        self.train_data = None
        self.test_data = None
        
        self.X_train_nts_scaled = None
        self.X_test_nts_scaled = None
        self.y_train_nts = {}
        self.y_test_nts = {}
        
        self.X_train_ts_scaled = None
        self.X_test_ts_scaled = None
        self.y_train_ts = {}
        self.y_test_ts = {}
        
        self.X_train_lstm = None
        self.X_test_lstm = None
        
        self.scaler_nts = MinMaxScaler()
        self.scaler_ts = MinMaxScaler()
        
        self.var_model_fit = None
        self.rf_models = {}
        self.gb_models = {}
        self.lstm_models = {}
        
        self.forecasts = {}
        self.metrics = {}
    
    def load_and_inspect_data(self):
        self.data = pd.read_csv(self.data_path, skiprows=self.skiprows)
        
        features_to_keep = [
            'Year', 'Month', 'Day', 'Hour', 'Minute', 'Temperature',
            'Clearsky DHI', 'Clearsky DNI', 'Clearsky GHI', 'DHI', 'DNI', 'GHI'
        ]
        
        columns_to_drop = [col for col in self.data.columns 
                          if 'Unnamed' in col or col not in features_to_keep]
        self.data = self.data.drop(columns=columns_to_drop)
        
        self.data['Timestamp'] = pd.to_datetime(
            self.data[['Year', 'Month', 'Day', 'Hour', 'Minute']]
        )
        self.data.set_index('Timestamp', inplace=True)
        self.data.drop(columns=['Year', 'Month', 'Day', 'Hour', 'Minute'], inplace=True)
        
        return self.data
    
    def preprocess_data(self, correlation_threshold=0.7):
        # Select correlated features for GHI
        self.new_data = self._select_correlated_features(
            self.data, target_col='GHI', threshold=correlation_threshold
        ).copy()

        # Seasonal differencing for all three variables
        for col in ['GHI', 'DHI', 'DNI']:
            if col in self.new_data.columns:
                self.new_data[f'{col}_seasonal_diff'] = (
                    self.new_data[col] - self.new_data[col].shift(self.s)
                )

        return self.new_data

    
    def _select_correlated_features(self, data, target_col='GHI', threshold=0.7):
        numerical_data = data.select_dtypes(include=np.number)
        correlation_with_target = numerical_data.corr()[target_col]
        
        correlated_features = correlation_with_target[
            (correlation_with_target > threshold) | (correlation_with_target < -threshold)
        ].index.tolist()
        
        if target_col not in correlated_features:
            correlated_features.append(target_col)
        
        return data[correlated_features]
    
    def split_data(self, train_end='2017-11-30 10:00:00', test_end='2017-12-31 23:00:00'):
        data_for_split = self.new_data.dropna()
        
        train_end_dt = pd.to_datetime(train_end)
        test_end_dt = pd.to_datetime(test_end)
        
        self.train_data = data_for_split.loc[:train_end_dt]
        self.test_data = data_for_split.loc[train_end_dt + timedelta(minutes=15):test_end_dt]
        
        cols_to_drop = []
        if 'Temperature' in self.train_data.columns:
            cols_to_drop.append('Temperature')
        if 'Solar Zenith Angle' in self.train_data.columns:
            cols_to_drop.append('Solar Zenith Angle')
        
        self.train_data = self.train_data.drop(columns=cols_to_drop, errors='ignore')
        self.test_data = self.test_data.drop(columns=cols_to_drop, errors='ignore')
        
        return self.train_data, self.test_data
    
    def prepare_ml_data(self, test_size=0.2, random_state=42):
        data_non_time = self.new_data.copy()
        
        cols_to_drop_nts = ['Temperature', 'Solar Zenith Angle', 'GHI_seasonal_diff', 
                           'GHI', 'DHI', 'DNI']
        cols_to_drop_nts = [col for col in cols_to_drop_nts if col in data_non_time.columns]
        
        X_nts = data_non_time.drop(columns=cols_to_drop_nts, errors='ignore')
        
        for target in ['GHI', 'DHI', 'DNI']:
            if target in data_non_time.columns:
                y_nts = data_non_time[target]
                
                X_train_nts, X_test_nts, y_train, y_test = train_test_split(
                    X_nts, y_nts, test_size=test_size, random_state=random_state, shuffle=True
                )
                
                self.y_train_nts[target] = y_train
                self.y_test_nts[target] = y_test
        
        X_train_nts, X_test_nts, _, _ = train_test_split(
            X_nts, data_non_time['GHI'], test_size=test_size, random_state=random_state, shuffle=True
        )
        self.X_train_nts_scaled = self.scaler_nts.fit_transform(X_train_nts)
        self.X_test_nts_scaled = self.scaler_nts.transform(X_test_nts)
        
        data_time_series = self.new_data.copy()
        
        for col in ['DNI', 'GHI', 'DHI']:
            data_time_series[f'{col}_lag1'] = data_time_series[col].shift(1)
            data_time_series[f'{col}_lag3'] = data_time_series[col].shift(3)
        
        for col in ['DNI', 'GHI', 'DHI']:
            data_time_series[f'{col}_rolling3'] = data_time_series[col].rolling(window=3).mean()
        
        data_time_series['Day of Year'] = data_time_series.index.day_of_year
        data_time_series.dropna(inplace=True)
        
        cols_to_drop_ts = ['Temperature', 'Solar Zenith Angle', 'GHI_seasonal_diff',
                          'GHI', 'DHI', 'DNI']
        cols_to_drop_ts = [col for col in cols_to_drop_ts if col in data_time_series.columns]
        
        X_time_series = data_time_series.drop(columns=cols_to_drop_ts, errors='ignore')
        
        for target in ['GHI', 'DHI', 'DNI']:
            if target in self.new_data.columns:
                y_time_series = data_time_series[target] if target in data_time_series.columns else self.new_data[target].loc[X_time_series.index]
                
                X_train_ts, X_test_ts, y_train, y_test = train_test_split(
                    X_time_series, y_time_series, test_size=test_size, random_state=random_state, shuffle=False
                )
                
                self.y_train_ts[target] = y_train
                self.y_test_ts[target] = y_test
        
        X_train_ts, X_test_ts, _, _ = train_test_split(
            X_time_series, data_time_series['GHI'], test_size=test_size, random_state=random_state, shuffle=False
        )
        
        self.X_train_ts_scaled = self.scaler_ts.fit_transform(X_train_ts)
        self.X_test_ts_scaled = self.scaler_ts.transform(X_test_ts)
        
        self.X_train_lstm = np.reshape(
            self.X_train_ts_scaled,
            (self.X_train_ts_scaled.shape[0], 1, self.X_train_ts_scaled.shape[1])
        )
        self.X_test_lstm = np.reshape(
            self.X_test_ts_scaled,
            (self.X_test_ts_scaled.shape[0], 1, self.X_test_ts_scaled.shape[1])
        )
    
    def train_var_model(self, max_lag=50):
        model = VAR(self.train_data)
        
        max_lag_order = min(max_lag, len(self.train_data) // 4)
        aic_values = []
        
        for lag in range(1, max_lag_order + 1):
            try:
                result = model.fit(lag)
                aic_values.append((lag, result.aic))
            except Exception as e:
                continue
        
        if aic_values:
            best_lag = min(aic_values, key=lambda x: x[1])
            self.var_model_fit = model.fit(maxlags=best_lag[0])
        else:
            self.var_model_fit = model.fit(maxlags=1)
        
        return self.var_model_fit
    
    def train_rf_gb_models(self, use_tuned=True, targets=['GHI', 'DHI', 'DNI']):
        for target in targets:
            if target not in self.y_train_ts:
                continue
            
            if use_tuned:
                rf_model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=10,
                    random_state=42
                )
                
                gb_model = GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=5,
                    subsample=0.8,
                    random_state=42
                )
            else:
                rf_model = RandomForestRegressor(random_state=42)
                gb_model = GradientBoostingRegressor(random_state=42)
            
            rf_model.fit(self.X_train_ts_scaled, self.y_train_ts[target])
            self.rf_models[target] = rf_model
            
            gb_model.fit(self.X_train_ts_scaled, self.y_train_ts[target])
            self.gb_models[target] = gb_model
        
        return self.rf_models, self.gb_models
    
    def train_lstm_model(self, epochs=80, batch_size=48, patience=5, targets=['GHI', 'DHI', 'DNI']):
        for target in targets:
            if target not in self.y_train_ts:
                continue
            
            lstm_model = Sequential([
                LSTM(50, return_sequences=True, 
                     input_shape=(self.X_train_lstm.shape[1], self.X_train_lstm.shape[2])),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(1)
            ])
            
            lstm_model.compile(optimizer='adam', loss='mean_squared_error')
            
            early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
            
            lstm_model.fit(
                self.X_train_lstm, self.y_train_ts[target],
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                verbose=0,
                callbacks=[early_stop]
            )
            
            self.lstm_models[target] = lstm_model
        
        return self.lstm_models
    
    def generate_forecasts(self, alpha=0.05):
        forecast_steps = len(self.test_data)
        last_k_ar_obs = self.train_data.iloc[-self.var_model_fit.k_ar:]

        # VAR interval forecast
        f_mean, f_low, f_up = self.var_model_fit.forecast_interval(
            y=last_k_ar_obs.values,
            steps=forecast_steps,
            alpha=alpha
        )

        forecast_df = pd.DataFrame(f_mean, index=self.test_data.index,
                                columns=self.train_data.columns)
        lower_df = pd.DataFrame(f_low, index=self.test_data.index,
                                columns=self.train_data.columns)
        upper_df = pd.DataFrame(f_up, index=self.test_data.index,
                                columns=self.train_data.columns)

        reverted_mean = {}
        reverted_low = {}
        reverted_up = {}

        for var in ['GHI', 'DHI', 'DNI']:
            diff_col = f"{var}_seasonal_diff"
            if diff_col not in forecast_df.columns:
                continue

            prev = self.new_data[var].shift(self.s).reindex(self.test_data.index)

            reverted_mean[var] = forecast_df[diff_col] + prev
            reverted_low[var] = lower_df[diff_col] + prev
            reverted_up[var] = upper_df[diff_col] + prev

        self.forecasts['VAR'] = pd.DataFrame(reverted_mean)
        self.forecasts['VAR_lower'] = pd.DataFrame(reverted_low)
        self.forecasts['VAR_upper'] = pd.DataFrame(reverted_up)

        # RF
        for tgt, model in self.rf_models.items():
            pred = model.predict(self.X_test_ts_scaled)
            self.forecasts.setdefault('RF', pd.DataFrame(index=self.y_test_ts[tgt].index))
            self.forecasts['RF'][tgt] = pd.Series(pred, index=self.y_test_ts[tgt].index)

        # GB
        for tgt, model in self.gb_models.items():
            pred = model.predict(self.X_test_ts_scaled)
            self.forecasts.setdefault('GB', pd.DataFrame(index=self.y_test_ts[tgt].index))
            self.forecasts['GB'][tgt] = pd.Series(pred, index=self.y_test_ts[tgt].index)

        # LSTM
        for tgt, model in self.lstm_models.items():
            pred = model.predict(self.X_test_lstm, verbose=0).flatten()
            self.forecasts.setdefault('LSTM', pd.DataFrame(index=self.y_test_ts[tgt].index))
            self.forecasts['LSTM'][tgt] = pd.Series(pred, index=self.y_test_ts[tgt].index)

        return self.forecasts

    
    def evaluate_models(self):
        if 'VAR' in self.forecasts:
            for var in ['GHI', 'DHI', 'DNI']:
                if var in self.forecasts['VAR'].columns and var in self.test_data.columns:
                    actual_var = self.test_data[var]
                    pred_var = self.forecasts['VAR'][var]
                    actual_var_aligned, pred_var_aligned = actual_var.align(pred_var, join='inner')
                    
                    metric_key = f'VAR_{var}'
                    self.metrics[metric_key] = {
                        'MAE': mean_absolute_error(actual_var_aligned, pred_var_aligned),
                        'MSE': mean_squared_error(actual_var_aligned, pred_var_aligned),
                        'RMSE': np.sqrt(mean_squared_error(actual_var_aligned, pred_var_aligned))
                    }
        
        for model_name in ['RF', 'GB', 'LSTM']:
            if model_name in self.forecasts:
                for target in ['GHI', 'DHI', 'DNI']:
                    if target in self.forecasts[model_name].columns and target in self.y_test_ts:
                        metric_key = f'{model_name}_{target}'
                        self.metrics[metric_key] = {
                            'MAE': mean_absolute_error(self.y_test_ts[target], self.forecasts[model_name][target]),
                            'MSE': mean_squared_error(self.y_test_ts[target], self.forecasts[model_name][target]),
                            'RMSE': np.sqrt(mean_squared_error(self.y_test_ts[target], self.forecasts[model_name][target]))
                        }
        
        return self.metrics

    def save_models(self, save_dir="models"):
        os.makedirs(save_dir, exist_ok=True)
    
        # Save scalers
        joblib.dump(self.scaler_nts,
                    os.path.join(save_dir, "scaler_nts.pkl"))
    
        joblib.dump(self.scaler_ts,
                    os.path.join(save_dir, "scaler_ts.pkl"))
    
        # Save Random Forest models
        for target, model in self.rf_models.items():
            joblib.dump(
                model,
                os.path.join(save_dir, f"rf_{target}.pkl")
            )
    
        # Save Gradient Boosting models
        for target, model in self.gb_models.items():
            joblib.dump(
                model,
                os.path.join(save_dir, f"gb_{target}.pkl")
            )
    
        # Save LSTM models
        for target, model in self.lstm_models.items():
            model.save(
                os.path.join(save_dir, f"lstm_{target}.keras")
            )

    
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

        # --- FIXED: Correct datetime slicing ---
        start_dt = pd.to_datetime(date)
        end_dt = start_dt + timedelta(hours=hours-1)

        # Actual values
        actual = self.test_data[target]
        actual_window = actual.loc[start_dt:end_dt]

        ax.plot(actual_window.index, actual_window.values,
                label='Actual', color='#FFFFFF', linewidth=1.8)

        # Model colors
        colors = {'VAR': '#EF8354', 'RF': '#66bb6a',
                'GB': '#4fc3f7', 'LSTM': '#ffa726'}

        # --- LOOP MODELS ---
        for model_name in ['VAR', 'RF', 'GB', 'LSTM']:
            if model_name not in self.forecasts:
                continue

            f = self.forecasts[model_name]

            if target not in f.columns:
                continue

            pred_window = f[target].loc[start_dt:end_dt]

            ax.plot(pred_window.index, pred_window.values,
                    label=model_name, color=colors[model_name], linewidth=1.8)

        ax.set_title(f'{target} Forecast – {hours}h Window',
                    color='#EF8354', fontweight='bold')
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

        # ---- 1) Actual Values ----
        if target not in self.test_data.columns:
            raise ValueError(f"Target '{target}' not present in test_data")

        actual = self.test_data[target].loc[date_dt:end_dt]
        if len(actual) == 0:
            raise ValueError("No actual data found for selected date")

        ax.plot(
            actual.index, actual.values,
            label='Actual',
            color='#FFFFFF',
            linewidth=1.2
        )

        # ---- 2) Forecast Colors ----
        colors = {
            'VAR':  '#EF8354',
            'RF':   '#66bb6a',
            'GB':   '#4fc3f7',
            'LSTM': '#ffa726'
        }

        # ---- 3) Loop Through Models ----
        for model_name, color in colors.items():

            if model_name not in self.forecasts:
                continue

            # VAR stores predictions differently
            if model_name == 'VAR':
                if target not in self.forecasts['VAR'].columns:
                    continue
                pred_full = self.forecasts['VAR'][target]

            else:
                # RF, GB, LSTM are DataFrames per target
                if target not in self.forecasts[model_name]:
                    continue
                pred_full = self.forecasts[model_name][target]

            # Slice the prediction window
            pred_day = pred_full.loc[date_dt:end_dt]

            if len(pred_day) == 0:
                continue

            ax.plot(
                pred_day.index,
                pred_day.values,
                linestyle='-',
                color=color,
                linewidth=1.1,
                label=model_name
            )

        # ---- 4) Styling ----
        ax.set_title(
            f'{target} Model Comparison – {date}',
            color='#EF8354',
            fontweight='bold'
        )
        ax.set_xlabel('Time', color='#FFFFFF')
        ax.set_ylabel(target, color='#FFFFFF')

        ax.tick_params(colors='#BFC0C0')
        ax.grid(alpha=0.25, color='#BFC0C0')

        for spine in ax.spines.values():
            spine.set_color('#BFC0C0')

        ax.legend(
            facecolor='#4F5D75',
            edgecolor='#BFC0C0',
            labelcolor='#FFFFFF'
        )

        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def get_preprocessing_correlation(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('#2D3142')
        
        corr = self.new_data.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn', 
                   center=0, square=True, linewidths=1,
                   cbar_kws={"shrink": 0.8}, ax=ax)
        
        ax.set_title('Correlation Heatmap', color='#EF8354', fontweight='bold')
        plt.xticks(color='#FFFFFF', rotation=45, ha='right')
        plt.yticks(color='#FFFFFF', rotation=0)
        
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(colors='#FFFFFF')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
        
    def get_training_metrics(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        fig.patch.set_facecolor('#2D3142')

        models = []
        rmse = []
        mae = []
        colors_list = []

        color_map = {'VAR': '#EF8354', 'RF': '#66bb6a',
                    'GB': '#4fc3f7', 'LSTM': '#ffa726'}

        # --- FIXED: include GHI, DHI, DNI ---
        for key, met in self.metrics.items():
            model_name, tgt = key.split("_")
            if tgt not in ["GHI", "DHI", "DNI"]:
                continue

            models.append(key)
            rmse.append(met['RMSE'])
            mae.append(met['MAE'])
            colors_list.append(color_map[model_name])

        # RMSE
        ax1.bar(models, rmse, color=colors_list, edgecolor='#BFC0C0')
        ax1.set_title("RMSE", color='#EF8354')
        ax1.tick_params(colors='#BFC0C0')
        ax1.set_facecolor('#4F5D75')
        ax1.grid(axis='y', alpha=0.2, color='#BFC0C0')

        for s in ax1.spines.values():
            s.set_color('#BFC0C0')

        # MAE
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
        
        components = ['GHI', 'DHI', 'DNI']
        colors_map = {'VAR': '#EF8354', 'RF': '#66bb6a', 'GB': '#4fc3f7', 'LSTM': '#ffa726'}
        
        for idx, component in enumerate(components):
            ax = axes[idx]
            ax.set_facecolor('#4F5D75')
            
            models = []
            rmse_vals = []
            bar_colors = []
            
            for key, metrics in self.metrics.items():
                if component in key:
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
    
    def plot_forecast_comparison(self, models=['VAR', 'RF', 'GB', 'LSTM'], 
                                 component='GHI', start_time=None, end_time=None):
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor('#2D3142')
        ax.set_facecolor('#4F5D75')
        
        if 'VAR' in models and 'VAR' in self.forecasts:
            actual_data = self.test_data[component]
        else:
            actual_data = self.y_test_ts[component]
        
        if start_time and end_time:
            start_dt = pd.to_datetime(start_time)
            end_dt = pd.to_datetime(end_time)
            actual_data = actual_data.loc[start_dt:end_dt]
        
        ax.plot(actual_data.index, actual_data.values, 
                label=f'Actual {component}', linewidth=2, alpha=0.8, color='#FFFFFF')
        
        colors = {'VAR': '#EF8354', 'RF': '#66bb6a', 'GB': '#4fc3f7', 'LSTM': '#ffa726'}
        linestyles = {'VAR': '--', 'RF': '-.', 'GB': ':', 'LSTM': '-'}
        
        for model_name in models:
            if model_name not in self.forecasts:
                continue
            
            if model_name == 'VAR':
                pred_data = self.forecasts['VAR'][component]
            else:
                pred_data = self.forecasts[model_name][component]
            
            if start_time and end_time:
                pred_data = pred_data.loc[start_dt:end_dt]
            
            ax.plot(pred_data.index, pred_data.values,
                    label=f'{model_name}', 
                    linestyle=linestyles.get(model_name, '-'),
                    linewidth=2, 
                    alpha=0.8,
                    color=colors.get(model_name, '#EF8354'))
        
        ax.set_title(f'Model Predictions vs Actual {component}', color='#EF8354', fontweight='bold')
        ax.set_xlabel('Time', color='#FFFFFF')
        ax.set_ylabel(f'{component} (W/m²)', color='#FFFFFF')
        ax.legend(facecolor='#4F5D75', edgecolor='#BFC0C0', labelcolor='#FFFFFF')
        ax.grid(alpha=0.2, color='#BFC0C0')
        ax.tick_params(colors='#BFC0C0')
        for spine in ax.spines.values():
            spine.set_color('#BFC0C0')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
