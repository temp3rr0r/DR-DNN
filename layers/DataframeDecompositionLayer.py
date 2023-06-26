import numpy as np
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.tsatools import freq_to_period
import pandas as pd
from statsmodels.tsa.stattools import kpss, adfuller, acf, pacf
import statsmodels.api as sm
import warnings

warnings.simplefilter("ignore")

class DataframeDecompositionLayer():
    def __init__(self, period=7, robust=True, forecast_STL=False, multiple_lags=False,
                 zero_imputation=False, interpolation=True,
                 freq="D", return_dataframe=False, endogenous_variables=[],
                 arima_params=None, trained_stl=None, use_moving_filters=True,
                 use_stl=True, use_extra_filters=False):
        super(DataframeDecompositionLayer, self).__init__()
        # Defaults
        self.period = period
        self.robust = robust
        self.num_inputs = 1
        self.num_outputs = self.num_inputs * 4
        self.forecast_STL = forecast_STL
        self.multiple_lags = multiple_lags
        self.zero_imputation = zero_imputation
        self.interpolation = interpolation
        self.freq = freq
        self.return_dataframe = return_dataframe
        self.arima_params = arima_params
        self.trained_stl = trained_stl
        self.use_moving_filters = use_moving_filters
        self.use_stl = use_stl
        self.use_extra_filters = use_extra_filters

    def get_decomposition(self, df_layer_input, period=7, robust=True, forecast_STL=False, multiple_lags=False,
             zero_imputation=True, interpolation=True,
             freq="D", return_dataframe=False, endogenous_variables=[],
             arima_params=None, trained_stl=None, use_moving_filters=True,
             use_stl=True, use_extra_filters=False):

        column_ids = df_layer_input.columns
        self.num_inputs = len(column_ids)
        returning_df = df_layer_input.copy()
        df_endogenous = None

        def get_cyclical_feature(column):
            return np.sin((column - column.min()) * (2. * (np.pi / column.max()))), np.cos(
                (column - column.min()) * (2. * (np.pi / column.max())))

        def get_arima_parameters(y):

            ds = []  # Stationarity tests

            result = adfuller(y, autolag=None)  # ADF: max lag
            ds.append(0 if result[0] < 0.5 else 1)
            result = adfuller(y, autolag="aic")  # ADF: AIC
            ds.append(0 if result[0] < 0.5 else 1)
            result = adfuller(y, autolag="bic")  # ADF: BIC
            ds.append(0 if result[0] < 0.5 else 1)
            result = adfuller(y, autolag="t-stat")  # ADF: t-stat
            ds.append(0 if result[0] < 0.5 else 1)

            kpss_test = kpss(y, regression='c', nlags="auto")  # KPSS: constant stationarity
            ds.append(1 if kpss_test[0] < 0.5 else 0)
            kpss_test = kpss(y, regression='ct', nlags="auto")  # KPSS: trend stationarity
            ds.append(1 if kpss_test[0] < 0.5 else 0)

            p = int(np.argmax(acf(y, fft=True)[1:]) + 1)  # ACF
            d = int(np.round(np.array(ds).mean(), 0))  # Stationarity
            q = int(np.argmax(pacf(y)[1:]) + 1)  # PACF
            return p, d, q

        def get_forecasted_signal(signal):
            p_signal, d_signal, q_signal = get_arima_parameters(signal)  # period (integration), stationarity, MA_range
            p_signal = min(p_signal, self.period)
            q_signal = max(2, min(q_signal, self.period))
           
            model_signal = sm.tsa.statespace.SARIMAX(signal, order=(p_signal, d_signal, q_signal))  # TODO: stable?
            model_signal_fit = model_signal.fit(disp=False)
            signal_prediction = model_signal_fit.predict(
                start=len(signal), end=len(signal) + self.period - 1, dynamic=True)

            signal_with_forecast = pd.concat([signal, signal_prediction]).iloc[self.period:].values  # Ignore first period data
            return signal_with_forecast

        i_s = [column_ids.get_loc(column) for column in endogenous_variables]
        if len(i_s) == 0:  # If no endogenous declared, take first
            i_s = [0]

        for i in i_s:
            df_endogenous = df_layer_input[column_ids[i]]

            if self.interpolation:
                df_endogenous = df_endogenous.dropna().interpolate().bfill()

            if self.zero_imputation:
                df_endogenous = df_endogenous.replace({0: np.nan}).dropna().interpolate().bfill()

            # STL
            if use_stl:
                if trained_stl is None:  # TODO: use trained stl
                    stl = STL(df_endogenous, robust=self.robust, period=self.period).fit()  # Decomposition
                    self.trained_stl = stl
                else:
                    stl = self.trained_stl
                returning_df[column_ids[i] + "_{}".format("trend")] = stl.trend
                returning_df[column_ids[i] + "_{}".format("seasonal")] = stl.seasonal
                returning_df[column_ids[i] + "_{}".format("resid")] = stl.resid

                if self.forecast_STL:  # Do add trend and seasonal forecasts (period ahead)
                    trend_with_forecast = get_forecasted_signal(stl.trend)  # Trend time-domain forecast
                    seasonal_with_forecast = get_forecasted_signal(stl.seasonal)  # Seasonal time-domain forecast
                    returning_df[column_ids[i] + "_{}".format("trend_with_forecast")] = trend_with_forecast
                    returning_df[column_ids[i] + "_{}".format("seasonal_with_forecast")] = seasonal_with_forecast

            # LAGs and MA
            if arima_params is None:  # TODO: use trained p, d, q
                p, d, q = get_arima_parameters(df_endogenous)  # ARIMA params from full signal
                self.arima_params = {"p": p, "d": d, "q": q}
            else:
                p = arima_params["p"]
                d = arima_params["d"]
                q = arima_params["q"]
            p = min(p, self.period)
            q_lag = max(1, min(q, self.period))
            q_ma = max(2, min(q, self.period))

            # Moving filters
            if self.use_moving_filters:
                # Lags, Moving Average/Variance, Diff and Moving Sum
                if self.multiple_lags:  # True to add multiple MA and lags
                    for lag_period in range(1, q_lag + 1):
                        lagged_df = df_endogenous.shift(periods=lag_period).bfill()
                        returning_df[column_ids[i] + "_{}{}".format("lag", lag_period)] = lagged_df

                    for lag_period in range(2, q_ma + 1):
                        if use_stl:
                            moving_average = stl.trend.rolling(window=lag_period, win_type='gaussian').mean(std=3).bfill()
                            returning_df[column_ids[i] + "_{}{}".format("trend_SMA", lag_period)] = moving_average
                        else:
                            moving_average = df_endogenous.rolling(window=lag_period, win_type='gaussian').mean(std=3).bfill()
                            returning_df[column_ids[i] + "_{}{}".format("SMA", lag_period)] = moving_average
                else:
                    lagged_df = df_endogenous.shift(periods=q_lag).bfill()
                    returning_df[column_ids[i] + "_{}{}".format("lag", q_lag)] = lagged_df

                    # Diff: 1-step and MA-period difference
                    diff_df = df_endogenous.diff(periods=1).bfill()
                    returning_df[column_ids[i] + "_{}{}".format("diff", 1)] = diff_df
                    period_diff_df = df_endogenous.diff(periods=q_ma).bfill()
                    returning_df[column_ids[i] + "_{}{}".format("period_diff", q_ma)] = period_diff_df

                if use_extra_filters:
                    if use_stl:
                        moving_average = stl.trend.ewm(span=q_ma, adjust=True).mean(std=3).bfill()
                        returning_df[column_ids[i] + "_{}{}".format("trend_EMA", q_ma)] = moving_average

                        moving_variance = stl.trend.rolling(window=q_ma).var().bfill()
                        returning_df[column_ids[i] + "_{}{}".format("trend_moving_variance", q_ma)] = moving_variance

                        rolling_sum = stl.trend.rolling(window=q_ma, win_type='gaussian').sum(std=3).bfill()
                        returning_df[column_ids[i] + "_{}{}".format("trend_rolling_sum", q_ma)] = rolling_sum
                    else:
                        moving_average = df_endogenous.ewm(span=q_ma, adjust=True).mean(std=3).bfill()
                        returning_df[column_ids[i] + "_{}{}".format("EMA", q_ma)] = moving_average

                        moving_variance = df_endogenous.rolling(window=q_ma).var().bfill()
                        returning_df[column_ids[i] + "_{}{}".format("moving_variance", q_ma)] = moving_variance

                        rolling_sum = df_endogenous.rolling(window=q_ma, win_type='gaussian').sum(std=3).bfill()
                        returning_df[column_ids[i] + "_{}{}".format("rolling_sum", q_ma)] = rolling_sum

        if self.use_moving_filters:
            if self.freq == "D" or self.freq == "H":
                # Add sinusoidal cyclical features
                returning_df["{}_{}".format("sin", "month")] = get_cyclical_feature(df_endogenous.index.month)[0]
                returning_df["{}_{}".format("cos", "month")] = get_cyclical_feature(df_endogenous.index.month)[1]
                returning_df["{}_{}".format("sin", "dayofweek")] = get_cyclical_feature(df_endogenous.index.dayofweek)[0]
                returning_df["{}_{}".format("cos", "dayofweek")] = get_cyclical_feature(df_endogenous.index.dayofweek)[1]
                returning_df["{}_{}".format("sin", "dayofyear")] = get_cyclical_feature(df_endogenous.index.dayofyear)[0]
                returning_df["{}_{}".format("cos", "dayofyear")] = get_cyclical_feature(df_endogenous.index.dayofyear)[1]
                returning_df["{}_{}".format("sin", "week")] = get_cyclical_feature(df_endogenous.index.isocalendar().week)[0]
                returning_df["{}_{}".format("cos", "week")] = get_cyclical_feature(df_endogenous.index.isocalendar().week)[1]
                # One-hot encoding
                returning_df["{}".format("weekend")] = (df_endogenous.index.weekday > 4).astype(float)
                returning_df["{}".format("weekday")] = (df_endogenous.index.weekday < 5).astype(float)

            if self.freq == "H":
                returning_df["{}_{}".format("sin", "hour")] = get_cyclical_feature(df_endogenous.index.hour)[0]  # Sinusoidal cyclical features
                returning_df["{}_{}".format("cos", "hour")] = get_cyclical_feature(df_endogenous.index.hour)[1]
                returning_df["{}".format("daytime")] = ((df_endogenous.index.hour >= 7) & (df_endogenous.index.hour < 22)).astype(float)  # One-hot encoding
                returning_df["{}".format("nighttime")] = ((df_endogenous.index.hour < 7) | (df_endogenous.index.hour >= 22)).astype(float)

        self.num_outputs = returning_df.columns.shape[0]
        if self.return_dataframe:
            return returning_df
        else:
            return returning_df.values
