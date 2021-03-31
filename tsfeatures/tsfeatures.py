#!/usr/bin/env python3

import logging
import numpy as np
import pandas as pd
from numba import jit # @manual
from scipy import stats
import statsmodels.api as sm
from scipy.signal import periodogram  # @manual
from infrastrategy.kats.consts import TimeSeriesData
from infrastrategy.kats.detectors import (
    cusum_detection,
    bocpd,
    robust_stat_detection,
    outlier,
    trend_mk,
    seasonality,
)
from infrastrategy.kats.models.nowcasting import feature_extraction
from statsmodels.tsa.seasonal import STL
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.stattools import acf, pacf, kpss
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from itertools import groupby
from functools import partial

class TsFeatures:
    """
    TODO: add documentation
    """

    def __init__(self,
                 window_size=20,
                 spectral_freq=1,
                 stl_period=7,
                 nbins=10,
                 lag_size=30,
                 acfpacf_lag=6,
                 decomp='additive',
                 iqr_mult=3.0,
                 threshold=.8,
                 window=5,
                 n_fast=26,
                 n_slow=5,
                 selected_features=None,
                 **kwargs):
        # init hyper-parameters
        self.window_size = window_size
        self.spectral_freq = spectral_freq
        self.stl_period = stl_period
        self.nbins = nbins
        self.lag_size = lag_size
        self.acfpacf_lag = acfpacf_lag
        self.decomp = decomp
        self.iqr_mult = iqr_mult
        self.threshold = threshold
        self.window = window
        self.n_fast = n_fast
        self.n_slow = n_slow

        # Mapping group features
        g2f = {
            "stl_features": [
                "trend_strength",
                "seasonality_strength",
                "spikiness",
                "peak",
                "trough",
            ],
            "level_shift_features": [
                "level_shift_idx",
                "level_shift_size",
            ],
            "acfpacf_features": [
                "y_acf1",
                "y_acf5",
                "diff1y_acf1",
                "diff1y_acf5",
                "diff2y_acf1",
                "diff2y_acf5",
                "y_pacf5",
                "diff1y_pacf5",
                "diff2y_pacf5",
                "seas_acf1",
                "seas_pacf1",
            ],
            "special_ac": [
                "firstmin_ac",
                "firstzero_ac",
            ],
            "holt_params": [
                "holt_alpha",
                "holt_beta",
            ],
            "hw_params": [
                "hw_alpha",
                "hw_beta",
                "hw_gamma",
            ],
            "statistics": [
                "length",
                "mean",
                "var",
                "entropy",
                "lumpiness",
                "stability",
                "flat_spots",
                "hurst",
                "std1st_der",
                "crossing_points",
                "binarize_mean",
                "unitroot_kpss",
                "heterogeneity",
                "histogram_mode",
                "linearity",
            ],
            "cusum_detector": [
                "cusum_num",
                "cusum_conf",
                "cusum_cp_index",
                "cusum_delta",
                "cusum_llr",
                "cusum_regression_detected",
                "cusum_stable_changepoint",
                "cusum_p_value",
            ],
            "robust_stat_detector": [
                "robust_num",
                "robust_metric_mean",
            ],
            "bocp_detector": [
                "bocp_num",
                "bocp_conf_max",
                "bocp_conf_mean",
            ],
            "outlier_detector": [
                "outlier_num",
            ],
            "trend_detector": [
                "trend_num",
                "trend_num_increasing",
                "trend_avg_abs_tau",
            ],
            "nowcasting": [
                "ROC_5",
                "MA_5",
                "MOM_5",
                "LAG_5",
                "MACD_26_5",
                "MACDsign_26_5",
                "MACDdiff_26_5",
            ],
            "seasonalities": [
                "seasonal_period",
                "trend_mag",
                "seasonality_mag",
                "residual_std",
            ]
        }
        f2g = {}
        for k, v in g2f.items():
            for f in v:
                f2g[f] = k

        self._total_feature_len_ = len(f2g.keys())
        for f in kwargs.keys():
            assert f in f2g.keys() or f in g2f.keys(), f"""couldn't find your desired feature/group "{f}", please check spelling"""

        # Higher level of features:
        # Once disabled, won't even go inside these groups of features
        # for calculation
        if not selected_features:
            default = True
            self.final_filter = {k:default for k in f2g.keys()}
        elif selected_features:
            default = False
            self.final_filter = {k:default for k in f2g.keys()}
            for f in selected_features:
                assert f in f2g.keys() or f in g2f.keys(), f"""couldn't find your desired feature/group "{f}", please check spelling"""
                if f in g2f.keys(): # the opt-in request is for a feature group
                    kwargs[f] = True
                    for feature in g2f[f]:
                        kwargs[feature] = kwargs.get(feature, True)
                        self.final_filter[feature] = True
                elif f in f2g.keys(): # the opt-in request is for a certain feature
                    assert kwargs.get(f2g[f], True), f"""feature group: {f2g[f]} has to be opt-in based on your opt-in request of feature: {f}"""
                    assert kwargs.get(f, True), f"""you have requested to both opt-in and opt-out feature: {f}"""
                    kwargs[f2g[f]] = True # need to opt-in the feature group first
                    kwargs[f] = True # opt-in the feature
                    self.final_filter[f] = True

        for k, v in kwargs.items():
            self.final_filter[k] = v # final filter for filtering out features user didn't request and keep only the requested ones

        self.stl_features = kwargs.get('stl_features', default)
        self.level_shift_features = kwargs.get('level_shift_features', default)
        self.acfpacf_features = kwargs.get('acfpacf_features', default)
        self.special_ac = kwargs.get('special_ac', default)
        self.holt_params = kwargs.get('holt_params', default)
        self.hw_params = kwargs.get('hw_params', default)
        self.statistics = kwargs.get('statistics', default)
        self.cusum_detector = kwargs.get('cusum_detector', False)
        self.robust_stat_detector = kwargs.get('robust_stat_detector', False)
        self.bocp_detector = kwargs.get('bocp_detector', False)
        self.outlier_detector = kwargs.get('outlier_detector', False)
        self.trend_detector = kwargs.get('trend_detector', False)
        self.nowcasting = kwargs.get('nowcasting', False)
        self.seasonalities = kwargs.get('seasonalities', False)

        # For lower level of the features
        self.__kwargs__ = kwargs
        self.default = default

    def transform(self, x: TimeSeriesData):
        """
        Transform time series into a number of features

        Input: TimeSeriesData object

        Output:
          for univariate input return a map of {feature: value}
          for multivarite input return a list of maps
        """
        if len(x) < 5:
            msg = "Length of time series is too short to calculate features!"
            logging.error(msg)
            raise ValueError(msg)

        if type(x.value.values) != np.ndarray:
            logging.warning(f"expecting values to be a ndarray, instead got {type(x.value.values)}")
            # make sure that values are numpy array for feeding to Numba
            df = pd.DataFrame(
                {
                    'time': x.time.values,
                    'value': np.array(x.value.values, dtype = float)
                }
            )
            x = TimeSeriesData(df)

        if len(x.value.shape) == 1:
            # a single Series: return a map of {feature: value}
            ts_values = x.value.values
            ts_features = self._transform_1d(ts_values, x)
        else:
            # multiple time series: return a list of map {feature: value}
            ts_features = []
            for col in x.value.columns:
                ts_values = x.value[col].values  # extract 1-d numpy array
                ts_features.append(
                    self._transform_1d(ts_values, x.value[col])
                )

        # performing final filter
        to_remove = []
        for feature in ts_features:
            if not self.final_filter[feature]:
                to_remove.append(feature)

        for r in to_remove:
            del ts_features[r]

        return ts_features

    def _transform_1d(self, x, ts):
        """
        Transform single time series
        """
        # calculate STL based features
        dict_stl_features = {}
        if self.stl_features:
            dict_stl_features = self.get_stl_features(x, period=self.stl_period, extra_args=self.__kwargs__,
                                                      default_status=self.default)

        # calculate level shift based features
        dict_level_shift_features = {}
        if self.level_shift_features:
            dict_level_shift_features = self.get_level_shift(
                x, window_size=self.window_size, extra_args=self.__kwargs__,
                default_status=self.default
            )

        # calculate ACF, PACF based features
        dict_acfpacf_features = {}
        if self.acfpacf_features:
            dict_acfpacf_features = self.get_acfpacf_features(
                x,
                acfpacf_lag=self.acfpacf_lag,
                period=self.stl_period,
                extra_args=self.__kwargs__,
                default_status=self.default
            )

        # calculate special AC
        dict_specialac_features = {}
        if self.special_ac:
            dict_specialac_features = self.get_special_ac(x, extra_args=self.__kwargs__, default_status=self.default)

        # calculate holt params
        dict_holt_params_features = {}
        if self.holt_params:
            dict_holt_params_features = self.get_holt_params(x, extra_args=self.__kwargs__, default_status=self.default)

        # calculate hw params
        dict_hw_params_features = {}
        if self.hw_params:
            dict_hw_params_features = self.get_hw_params(x, period=self.stl_period, extra_args=self.__kwargs__,
                                                         default_status=self.default)

        # single features
        _dict_features_ = {}
        if self.statistics:
            dict_features = {
                'length': partial(self.get_length),
                'mean': partial(self.get_mean),
                'var': partial(self.get_var),
                'entropy': partial(self.get_spectral_entropy, freq=self.spectral_freq),
                'lumpiness': partial(self.get_lumpiness, window_size=self.window_size),
                'stability': partial(self.get_stability, window_size=self.window_size),
                'flat_spots': partial(self.get_flat_spots, nbins=self.nbins),
                'hurst': partial(self.get_hurst, lag_size=self.lag_size),
                'std1st_der': partial(self.get_std1st_der),
                'crossing_points': partial(self.get_crossing_points),
                'binarize_mean': partial(self.get_binarize_mean),
                'unitroot_kpss': partial(self.get_unitroot_kpss),
                'heterogeneity': partial(self.get_het_arch),
                'histogram_mode': partial(self.get_histogram_mode, nbins=self.nbins),
                'linearity': partial(self.get_linearity)
            }

            _dict_features_ = {}
            for k, v in dict_features.items():
                if self.__kwargs__.get(k, self.default):
                    _dict_features_[k] = v(x)

        # calculate cusum detector features
        dict_cusum_detector_features = {}
        if self.cusum_detector:
            dict_cusum_detector_features = self.get_cusum_detector(ts, extra_args=self.__kwargs__, default_status=self.default)

        # calculate robust stat detector features
        dict_robust_stat_detector_features = {}
        if self.robust_stat_detector:
            dict_robust_stat_detector_features = self.get_robust_stat_detector(ts, extra_args=self.__kwargs__, default_status=self.default)

        # calculate bocp detector features
        dict_bocp_detector_features = {}
        if self.bocp_detector:
            dict_bocp_detector_features = self.get_bocp_detector(ts, extra_args=self.__kwargs__, default_status=self.default)

        # calculate outlier detector features
        dict_outlier_detector_features = {}
        if self.outlier_detector:
            dict_outlier_detector_features = self.get_outlier_detector(
                ts,
                decomp=self.decomp,
                iqr_mult=self.iqr_mult,
                extra_args=self.__kwargs__,
                default_status=self.default
            )

        # calculate trend detector features
        dict_trend_detector_features = {}
        if self.trend_detector:
            dict_trend_detector_features = self.get_trend_detector(
                ts,
                threshold=self.threshold,
                extra_args=self.__kwargs__,
                default_status=self.default
            )

        # calculate nowcasting features
        dict_nowcasting_features = {}
        if self.nowcasting:
            dict_nowcasting_features = self.get_nowcasting(
                ts,
                window=self.window,
                n_fast=self.n_fast,
                n_slow=self.n_slow,
                extra_args=self.__kwargs__,
                default_status=self.default
            )

        # calculate seasonality features
        dict_seasonality_features = {}
        if self.seasonalities:
            dict_seasonality_features = self.get_seasonalities(
                ts,
                extra_args=self.__kwargs__,
                default_status=self.default
            )

        return {
            **_dict_features_,
            **dict_stl_features,
            **dict_level_shift_features,
            **dict_acfpacf_features,
            **dict_specialac_features,
            **dict_holt_params_features,
            **dict_hw_params_features,
            **dict_cusum_detector_features,
            **dict_robust_stat_detector_features,
            **dict_bocp_detector_features,
            **dict_outlier_detector_features,
            **dict_trend_detector_features,
            **dict_nowcasting_features,
            **dict_seasonality_features,
        }

    # length
    @staticmethod
    @jit(nopython=True)
    def get_length(x):
        return len(x)

    # mean
    @staticmethod
    @jit(nopython=True)
    def get_mean(x):
        return np.mean(x)

    # variance
    @staticmethod
    @jit(nopython=True)
    def get_var(x):
        return np.var(x)

    # spectral entropy
    @staticmethod
    @jit(forceobj=True)
    def get_spectral_entropy(x, freq=1):
        """
        Shannon entropy (normalized) of power spectral density
        PSD is calculated using scipy's periodogram
        """
        # calculate periodogram
        _, psd = periodogram(x, freq)

        # calculate shannon entropy of normalized psd
        psd_norm = psd / np.sum(psd)
        entropy = np.nansum(psd_norm * np.log2(psd_norm))

        return entropy / np.log2(psd_norm.size)

    # lumpiness
    @staticmethod
    @jit(forceobj=True)
    def get_lumpiness(x, window_size=20):
        """
        Lumpiness of time series:
        Variance of chunk-wise variances
        """
        v = [np.var(x_w) for x_w in np.array_split(x, len(x) // window_size + 1)]
        return np.var(v)

    # stability
    @staticmethod
    @jit(forceobj=True)
    def get_stability(x, window_size=20):
        """
        Stability of time series:
        Variance of chunk-wise means
        """
        v = [np.mean(x_w) for x_w in np.array_split(x, len(x) // window_size + 1)]
        return np.var(v)

    # STL decomposition based features
    @staticmethod
    @jit(forceobj=True)
    def get_stl_features(x, period=7, extra_args=None, default_status=True):
        """
        Calculate STL based features: strength of trend,
            seasonality, spikiness, peak/trough
        """
        stl_features = {}

        # STL decomposition
        res = STL(x, period=period).fit()

        # strength of trend
        if extra_args.get('trend_strength', default_status):
            stl_features['trend_strength'] = 1 - \
                np.var(res.resid) / np.var(res.trend + res.resid)

        # strength of seasonality
        if extra_args.get('seasonality_strength', default_status):
            stl_features['seasonality_strength'] = 1 - \
                np.var(res.resid) / np.var(res.seasonal + res.resid)

        # spikiness: variance of the leave-one-out variances of the remainder component
        if extra_args.get('spikiness', default_status):
            resid_array = np.repeat(np.array(res.resid)[:,np.newaxis], len(res.resid), axis = 1)
            resid_array[np.diag_indices(len(resid_array))] = np.NaN
            stl_features['spikiness'] = np.var(np.nanvar(resid_array, axis = 0))

        # location of peak
        if extra_args.get('peak', default_status):
            stl_features['peak'] = np.argmax(res.seasonal[:period])

        # location of trough
        if extra_args.get('trough', default_status):
            stl_features['trough'] = np.argmin(res.seasonal[:period])

        return stl_features

    # Level shift
    @staticmethod
    @jit(forceobj=True)
    def get_level_shift(x, window_size=20, extra_args=None, default_status=True):
        """
        Location and size of maximum mean value difference,
            between two consecutive sliding windows
        """
        level_shift_features = {'level_shift_idx': np.nan, 'level_shift_size': np.nan}
        if len(x) < window_size + 2:
            msg = "Length of time series is shorter than window_size, unable to calculate level shift features!"
            logging.error(msg)
            return level_shift_features

        sliding_idx = (np.arange(len(x))[None,:] + np.arange(window_size)[:,None])[:,:len(x) - window_size + 1]
        means = np.mean(x[sliding_idx], axis = 0)
        mean_diff = np.abs(means[:-1] - means[1:])

        if extra_args.get('level_shift_idx', default_status):
            level_shift_features['level_shift_idx'] = np.argmax(mean_diff)
        if extra_args.get('level_shift_size', default_status):
            level_shift_features['level_shift_size'] = mean_diff[np.argmax(mean_diff)]
        return level_shift_features

    # Flat spots
    @staticmethod
    @jit(forceobj=True)
    def get_flat_spots(x, nbins=10):
        """
        Maximum run-lengths across equally-sized segments of x
        """
        if len(x) <= nbins:
            msg = "Length of time series is shorter than nbins, unable to calculate flat spots feature!"
            logging.error(msg)
            return np.nan

        max_run_length = 0
        window_size = int(len(x) / nbins)
        for i in range(0, len(x), window_size):
            run_length = np.max(
                [len(list(v)) for k, v in groupby(x[i:i + window_size])]
            )
            if run_length > max_run_length:
                max_run_length = run_length
        return max_run_length

    # Hurst Exponent
    @staticmethod
    @jit(forceobj=True)
    def get_hurst(x, lag_size=30):
        """
        Returns the Hurst Exponent of the time series vector x
        Hurst Exponent wiki: https://en.wikipedia.org/wiki/Hurst_exponent
        """
        # Create the range of lag values
        lags = range(2, min(lag_size, len(x) - 1))

        # Calculate the array of the variances of the lagged differences
        tau = [np.std(np.asarray(x)[lag:] - np.asarray(x)[:-lag]) for lag in lags]

        # Use a linear fit to estimate the Hurst Exponent
        poly = np.polyfit(np.log(lags), np.log(tau), 1)

        # Return the Hurst exponent from the polyfit output
        return poly[0] if not np.isnan(poly[0]) else 0

    # ACF and PACF features
    # ACF features
    @staticmethod
    @jit(forceobj=True)
    def get_acf_features(extra_args, default_status, y_acf_list, diff1y_acf_list, diff2y_acf_list):
        (
            y_acf1,
            y_acf5,
            diff1y_acf1,
            diff1y_acf5,
            diff2y_acf1,
            diff2y_acf5,
            seas_acf1,
        ) = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

        # y_acf1: first ACF value of the original series
        if extra_args.get('y_acf1', default_status):
            y_acf1 = y_acf_list[0]

        # y_acf5: sum of squares of first 5 ACF values of original series
        if extra_args.get('y_acf5', default_status):
            y_acf5 = np.sum(np.asarray(y_acf_list)[:5] ** 2)

        # diff1y_acf1: first ACF value of the differenced series
        if extra_args.get('diff1y_acf1', default_status):
            diff1y_acf1 = diff1y_acf_list[0]

        # diff1y_acf5: sum of squares of first 5 ACF values of differenced series
        if extra_args.get('diff1y_acf5', default_status):
            diff1y_acf5 = np.sum(np.asarray(diff1y_acf_list)[:5] ** 2)

        # diff2y_acf1: first ACF value of the twice-differenced series
        if extra_args.get('diff2y_acf1', default_status):
            diff2y_acf1 = diff2y_acf_list[0]

        # diff2y_acf5: sum of squares of first 5 ACF values of twice-differenced series
        if extra_args.get('diff2y_acf5', default_status):
            diff2y_acf5 = np.sum(np.asarray(diff2y_acf_list)[:5] ** 2)

        # Autocorrelation coefficient at the first seasonal lag.
        if extra_args.get('seas_acf1', default_status):
            seas_acf1 = y_acf_list[-1]

        return (
            y_acf1,
            y_acf5,
            diff1y_acf1,
            diff1y_acf5,
            diff2y_acf1,
            diff2y_acf5,
            seas_acf1,
        )

    # PACF features
    @staticmethod
    @jit(forceobj=True)
    def get_pacf_features(extra_args, default_status, y_pacf_list, diff1y_pacf_list, diff2y_pacf_list):
        (
            y_pacf5,
            diff1y_pacf5,
            diff2y_pacf5,
            seas_pacf1,
        ) = (np.nan, np.nan, np.nan, np.nan)

        # y_pacf5: sum of squares of first 5 PACF values of original series
        if extra_args.get('y_pacf5', default_status):
            y_pacf5 = np.nansum(np.asarray(y_pacf_list)[:5] ** 2)

        # diff1y_pacf5: sum of squares of first 5 PACF values of differenced series
        if extra_args.get('diff1y_pacf5', default_status):
            diff1y_pacf5 = np.nansum(np.asarray(diff1y_pacf_list)[:5] ** 2)

        # diff2y_pacf5: sum of squares of first 5 PACF values of twice-differenced series
        if extra_args.get('diff2y_pacf5', default_status):
            diff2y_pacf5 = np.nansum(np.asarray(diff2y_pacf_list)[:5] ** 2)

        # Patial Autocorrelation coefficient at the first seasonal lag.
        if extra_args.get('seas_pacf1', default_status):
            seas_pacf1 = y_pacf_list[-1]

        return (
            y_pacf5,
            diff1y_pacf5,
            diff2y_pacf5,
            seas_pacf1,
        )

    @staticmethod
    @jit(forceobj=True)
    def get_acfpacf_features(x, acfpacf_lag=6, period=7, extra_args=None, default_status=True):
        """
        period: seasonal period.

        Calculate ACF and PACF based features.
        Calculate seasonal acf, pacf base features
        Reference: https://stackoverflow.com/questions/36038927/whats-the-difference-between-pandas-acf-and-statsmodel-acf
        R code: https://cran.r-project.org/web/packages/tsfeatures/vignettes/tsfeatures.html
        Paper: Meta-learning how to forecast time series
        """
        acfpacf_features = {
                'y_acf1': np.nan,
                'y_acf5': np.nan,
                'diff1y_acf1': np.nan,
                'diff1y_acf5': np.nan,
                'diff2y_acf1': np.nan,
                'diff2y_acf5': np.nan,
                'y_pacf5': np.nan,
                'diff1y_pacf5': np.nan,
                'diff2y_pacf5': np.nan,
                'seas_acf1': np.nan,
                'seas_pacf1': np.nan
            }
        if len(x) < 10 or len(x) < period or len(np.unique(x)) == 1:
            msg = "Length is shorter than period, or constant time series! Unable to calculate acf/pacf features!"
            logging.error(msg)
            return acfpacf_features

        nlag = min(acfpacf_lag, len(x) - 2)

        diff1x = [x[i] - x[i - 1] for i in range(1, len(x))]
        diff2x = [diff1x[i] - diff1x[i - 1] for i in range(1, len(diff1x))]

        y_acf_list = acf(x, unbiased=False, fft=True, nlags=period)[1:]
        diff1y_acf_list = acf(diff1x, unbiased=False, fft=True, nlags=nlag)[1:]
        diff2y_acf_list = acf(diff2x, unbiased=False, fft=True, nlags=nlag)[1:]

        y_pacf_list = pacf(x, nlags=period)[1:]
        diff1y_pacf_list = pacf(diff1x, nlags=nlag)[1:]
        diff2y_pacf_list = pacf(diff2x, nlags=nlag)[1:]

        # getting ACF features
        (
            acfpacf_features['y_acf1'],
            acfpacf_features['y_acf5'],
            acfpacf_features['diff1y_acf1'],
            acfpacf_features['diff1y_acf5'],
            acfpacf_features['diff2y_acf1'],
            acfpacf_features['diff2y_acf5'],
            acfpacf_features['seas_acf1'],
        ) = TsFeatures.get_acf_features(
            extra_args,
            default_status,
            y_acf_list,
            diff1y_acf_list,
            diff2y_acf_list,
        )

        # getting PACF features
        (
            acfpacf_features['y_pacf5'],
            acfpacf_features['diff1y_pacf5'],
            acfpacf_features['diff2y_pacf5'],
            acfpacf_features['seas_pacf1'],
        ) = TsFeatures.get_pacf_features(
            extra_args,
            default_status,
            y_pacf_list,
            diff1y_pacf_list,
            diff2y_pacf_list,
        )

        return acfpacf_features

    # standard deviation of the first derivative
    @staticmethod
    @jit(forceobj=True)
    def get_std1st_der(x):
        """
        std1st_der: the standard deviation of the first derivative of the time series.
        Reference: https://cran.r-project.org/web/packages/tsfeatures/vignettes/tsfeatures.html
        """
        std1st_der = np.std(np.gradient(x))
        return std1st_der

    # crossing points
    @staticmethod
    @jit(nopython=True)
    def get_crossing_points(x):
        """
        crossing points: the number of times a time series crosses the median line.
        Reference: https://cran.r-project.org/web/packages/tsfeatures/vignettes/tsfeatures.html
        """
        median = np.median(x)
        cp = 0
        for i in range(len(x) - 1):
            if x[i] <= median < x[i + 1] or x[i] >= median > x[i + 1]:
                cp += 1
        return cp

    # binarize mean
    @staticmethod
    @jit(nopython=True)
    def get_binarize_mean(x):
        """
        converts x into a binarized version.
        Time-series values above its mean are given 1, and those below the mean are 0.
        return the average value of the binarized vector.
        Reference: https://cran.r-project.org/web/packages/tsfeatures/vignettes/tsfeatures.html
        """
        return np.mean(np.asarray(x) > np.mean(x))

    # KPSS unit root test
    @staticmethod
    @jit(forceobj=True)
    def get_unitroot_kpss(x):
        """
        A test statistic based on KPSS test.
        Test a null hypothesis that an observable time series is stationary around a deterministic trend.
        A vector comprising the statistic for the KPSS unit root test with linear trend and lag one
        Wiki: https://en.wikipedia.org/wiki/KPSS_test
        """
        return kpss(x, regression='ct', nlags=1)[0]

    # heterogeneity
    @staticmethod
    @jit(forceobj=True)
    def get_het_arch(x):
        """
        reference: https://www.statsmodels.org/dev/generated/statsmodels.stats.diagnostic.het_arch.html
        Engle’s Test for Autoregressive Conditional Heteroscedasticity (ARCH)
        Returns Lagrange multiplier test statistic
        """
        return het_arch(x, nlags=min(10, len(x) // 5))[0]

    # histogram mode
    @staticmethod
    @jit(nopython=True)
    def get_histogram_mode(x, nbins=10):
        """
        Measures the mode of the data vector using histograms with a given number of bins.
        Reference: https://cran.r-project.org/web/packages/tsfeatures/vignettes/tsfeatures.html
        """
        cnt, val = np.histogram(x, bins=nbins)
        return val[cnt.argmax()]

    # First min/zero AC (2)
    @staticmethod
    @jit(forceobj=True)
    def get_special_ac(x, extra_args=None, default_status=True):
        """
        1. the time of first minimum in the autocorrelation function
        2. the time of first zero crossing the autocorrelation function.
        """
        # First min AC
        special_ac_features = {'firstmin_ac': np.nan, 'firstzero_ac': np.nan}
        AC = acf(x, unbiased=False, fft=True, nlags=len(x))[1:]
        if extra_args.get('firstmin_ac', default_status):
            i = 0
            while i < len(AC) - 1:
                if AC[i] > AC[i + 1]:
                    i += 1
                else:
                    break
            special_ac_features['firstmin_ac'] = i + 1

        # First zero AC
        if extra_args.get('firstzero_ac', default_status):
            j = 0
            while j < len(AC) - 1:
                if AC[j] > 0 > AC[j + 1]:
                    break
                else:
                    j += 1
            special_ac_features['firstzero_ac'] = j + 2
        return special_ac_features

    # Linearity
    @staticmethod
    @jit(forceobj=True)
    def get_linearity(x):
        """
        Linearity: R square from a fitted linear regression.
        """
        _, _, r_value, _, _ = stats.linregress(np.arange(len(x)), x)
        return r_value ** 2

    # Holt Parameters (2)
    @staticmethod
    def get_holt_params(x, extra_args=None, default_status=True):
        """
        Estimates the smoothing parameter for the level-alpha and the smoothing parameter
        for the trend-beta of Holt’s linear trend method.
        'alpha': Level parameter of the Holt model.
        'beta': Trend parameter of the Hold model.
        """
        holt_params_features = {'holt_alpha': np.nan, 'holt_beta': np.nan}
        try:
            m = ExponentialSmoothing(x, trend='add', seasonal=None).fit()
            if extra_args.get('holt_alpha', default_status):
                holt_params_features['holt_alpha'] = m.params['smoothing_level']
            if extra_args.get('holt_beta', default_status):
                holt_params_features['holt_beta'] = m.params['smoothing_slope']
            return holt_params_features
        except Exception:
            return holt_params_features

    # Holt Winter’s Parameters (3)
    @staticmethod
    def get_hw_params(x, period=7, extra_args=None, default_status=True):
        """
        Estimates the smoothing parameter for the level-alpha, trend-beta of HW’s linear trend,
        and additive seasonal trend-gamma.
        """
        hw_params_features = {'hw_alpha': np.nan, 'hw_beta': np.nan, 'hw_gamma': np.nan}
        try:
            m = ExponentialSmoothing(x, seasonal_periods=period, trend='add', seasonal='add').fit(use_boxcox=True)
            if extra_args.get('hw_alpha', default_status):
                hw_params_features['hw_alpha'] = m.params['smoothing_level']
            if extra_args.get('hw_beta', default_status):
                hw_params_features['hw_beta'] = m.params['smoothing_slope']
            if extra_args.get('hw_gamma', default_status):
                hw_params_features['hw_gamma'] = m.params['smoothing_seasonal']
            return hw_params_features
        except Exception:
            return hw_params_features

    # CUSUM Detection Outputs (8)
    @staticmethod
    def get_cusum_detector(ts, extra_args=None, default_status=True):
        """
        Run the Kats CUSUM Detector on the Time Series, extract features from the outputs of the detection
        """
        cusum_detector_features = {'cusum_num': np.nan, 'cusum_conf': np.nan, 'cusum_cp_index': np.nan, 'cusum_delta': np.nan, 'cusum_llr': np.nan, 'cusum_regression_detected': np.nan, 'cusum_stable_changepoint': np.nan, 'cusum_p_value': np.nan}
        try:
            cusum = cusum_detection.CUSUMDetector(ts)
            cusum_cp = cusum.detector()
            if extra_args.get('cusum_num', default_status):
                cusum_detector_features['cusum_num'] = len(cusum_cp)
            if extra_args.get('cusum_conf', default_status):
                cusum_detector_features['cusum_conf'] = 0 if len(cusum_cp)==0 else cusum_cp[0][0].confidence
            if extra_args.get('cusum_cp_index', default_status):
                cusum_detector_features['cusum_cp_index'] = 0 if len(cusum_cp)==0 else cusum_cp[0][1]._cp_index / len(ts)
            if extra_args.get('cusum_delta', default_status):
                cusum_detector_features['cusum_delta'] = 0 if len(cusum_cp)==0 else cusum_cp[0][1]._delta
            if extra_args.get('cusum_llr', default_status):
                cusum_detector_features['cusum_llr'] = 0 if len(cusum_cp)==0 else cusum_cp[0][1]._llr
            if extra_args.get('cusum_regression_detected', default_status):
                cusum_detector_features['cusum_regression_detected'] = False if len(cusum_cp)==0 else cusum_cp[0][1]._regression_detected
            if extra_args.get('cusum_stable_changepoint', default_status):
                cusum_detector_features['cusum_stable_changepoint'] = False if len(cusum_cp)==0 else cusum_cp[0][1]._stable_changepoint
            if extra_args.get('cusum_p_value', default_status):
                cusum_detector_features['cusum_p_value'] = 0 if len(cusum_cp)==0 else cusum_cp[0][1]._p_value
            return cusum_detector_features
        except Exception:
            return cusum_detector_features

    # Robust Stat Detection Outputs (2)
    @staticmethod
    def get_robust_stat_detector(ts, extra_args=None, default_status=True):
        """
        Run the Kats Robust Stat Detector on the Time Series, extract features from the outputs of the detection
        """
        robust_stat_detector_features = {'robust_num': np.nan, 'robust_metric_mean': np.nan}
        try:
            robust = robust_stat_detection.RobustStatDetector(ts)
            robust_cp = robust.detector()
            if extra_args.get('robust_num', default_status):
                robust_stat_detector_features['robust_num'] = len(robust_cp)
            if extra_args.get('robust_metric_mean', default_status):
                robust_stat_detector_features['robust_metric_mean'] = 0 if len(robust_cp)==0 else np.sum([cp[1]._metric for cp in robust_cp]) / len(robust_cp)
            return robust_stat_detector_features
        except Exception:
            return robust_stat_detector_features

    # BOCP Detection Outputs (3)
    @staticmethod
    def get_bocp_detector(ts, extra_args=None, default_status=True):
        """
        Run the Kats BOCP Detector on the Time Series, extract features from the outputs of the detection
        """
        bocp_detector_features = {'bocp_num': np.nan, 'bocp_conf_max': np.nan, 'bocp_conf_mean': np.nan}
        try:
            bocp = bocpd.BOCPDetector(ts)
            bocp_cp = bocp.detector(choose_priors = False)
            if extra_args.get('bocp_num', default_status):
                bocp_detector_features['bocp_num'] = len(bocp_cp)
            if extra_args.get('bocp_conf_max', default_status):
                bocp_detector_features['bocp_conf_max'] = 0 if len(bocp_cp)==0 else np.max([cp[0].confidence for cp in bocp_cp])
            if extra_args.get('bocp_conf_mean', default_status):
                bocp_detector_features['bocp_conf_mean'] = 0 if len(bocp_cp)==0 else np.sum([cp[0].confidence for cp in bocp_cp]) / len(bocp_cp)
            return bocp_detector_features
        except Exception:
            return bocp_detector_features

    # Outlier Detection Outputs (1)
    @staticmethod
    def get_outlier_detector(ts, decomp='additive', iqr_mult=3.0, extra_args=None, default_status=True):
        """
        Run the Kats Outlier Detector on the Time Series, extract features from the outputs of the detection
        """
        outlier_detector_features = {'outlier_num': np.nan}
        try:
            odetector = outlier.OutlierDetector(ts, decomp=decomp, iqr_mult=iqr_mult)
            odetector.detector()
            if extra_args.get('outlier_num', default_status):
                outlier_detector_features['outlier_num'] = len(odetector.outliers[0])
            return outlier_detector_features
        except Exception:
            return outlier_detector_features

    # Trend Detection Outputs (3)
    @staticmethod
    def get_trend_detector(ts, threshold=.8, extra_args=None, default_status=True):
        """
        Run the Kats Trend Detector on the Time Series, extract features from the outputs of the detection
        """
        trend_detector_features = {'trend_num': np.nan, 'trend_num_increasing': np.nan, 'trend_avg_abs_tau': np.nan}
        try:
            tdetector = trend_mk.MKDetector(data=ts, threshold=threshold)
            tdetected_time_points = tdetector.detector(direction='both')
            if extra_args.get('trend_num', default_status):
                trend_detector_features['trend_num'] = len(tdetected_time_points)
            if extra_args.get('trend_num_increasing', default_status):
                trend_detector_features['trend_num_increasing'] = len([p for p in tdetected_time_points if p[1].trend_direction == 'decreasing'])
            if extra_args.get('trend_avg_abs_tau', default_status):
                trend_detector_features['trend_avg_abs_tau'] = 0 if len(tdetected_time_points)==0 else np.sum([abs(p[1].Tau) for p in tdetected_time_points]) / len(tdetected_time_points)
            return trend_detector_features
        except Exception:
            return trend_detector_features

    # Nowcasting features (3)
    @staticmethod
    def get_nowcasting(ts, window=5, n_fast=26, n_slow=5, extra_args=None, default_status=True):
        """
        Run the Kats Nowcasting transformer on the Time Series, extract aggregated features from the outputs of the transformation
        """
        nowcasting_features = {'ROC_5': np.nan, 'MA_5': np.nan, 'MOM_5': np.nan, 'LAG_5': np.nan, 'MACD_26_5': np.nan, 'MACDsign_26_5': np.nan, 'MACDdiff_26_5': np.nan}
        try:
            ts_df = pd.DataFrame({'y':ts.value,'hist_ds':ts.time})
            ts_df = feature_extraction.ROC(ts_df, window)
            ts_df = feature_extraction.MA(ts_df, window)
            ts_df = feature_extraction.MOM(ts_df, window)
            ts_df = feature_extraction.LAG(ts_df, window)
            ts_df = feature_extraction.MACD(ts_df, n_fast=n_fast, n_slow=n_slow)
            nc_feats = dict(ts_df.drop(['y','hist_ds'],axis=1).mean().fillna(0).replace([np.inf,-np.inf], 0))
            if extra_args.get('ROC_5', default_status):
                nowcasting_features['ROC_5'] = nc_feats['ROC_5']
            if extra_args.get('MA_5', default_status):
                nowcasting_features['MA_5'] = nc_feats['MA_5']
            if extra_args.get('MOM_5', default_status):
                nowcasting_features['MOM_5'] = nc_feats['MOM_5']
            if extra_args.get('LAG_5', default_status):
                nowcasting_features['LAG_5'] = nc_feats['LAG_5']
            if extra_args.get('MACD_26_5', default_status):
                nowcasting_features['MACD_26_5'] = nc_feats['MACD_26_5']
            if extra_args.get('MACDsign_26_5', default_status):
                nowcasting_features['MACDsign_26_5'] = nc_feats['MACDsign_26_5']
            if extra_args.get('MACDdiff_26_5', default_status):
                nowcasting_features['MACDdiff_26_5'] = nc_feats['MACDdiff_26_5']
            return nowcasting_features
        except Exception:
            return nowcasting_features

    # seasonality features (4)
    @staticmethod
    def get_seasonalities(ts, extra_args=None, default_status=True):
        """
        Run the Kats seaonality detectors to get the estimated seasonal period, then extract trend, seasonality and residual magnitudes
        """

        seasonality_features = {
            'seasonal_period': np.nan,
            'trend_mag': np.nan,
            'seasonality_mag': np.nan,
            'residual_std': np.nan,
        }

        try:
            # detrending for period estimation
            detrended = TimeSeriesData(pd.DataFrame({
                'time': len(ts.value.values) - 1,
                'value': ts.value.values[1:] - ts.value.values[:-1]
            }))
            detected = seasonality.FFTDetector(detrended).detector()

            if detected['seasonality_presence']:
                _period = int(np.min(detected['seasonalities']))
            elif not detected['seasonality_presence']:
                _period = 7
            res = STL(ts.value.values, period = _period).fit()

            if extra_args.get('seasonal_period', default_status):
                seasonality_features['seasonal_period'] = _period

            # getting seasonality magnitude
            if extra_args.get('seasonality_mag', default_status):
                seasonality_features['seasonality_mag'] = np.round(np.quantile(res.seasonal, 0.95) - np.quantile(res.seasonal, 0.05))

            # fitting linear regression for trend magnitude
            if extra_args.get('trend_mag', default_status):
                exog = res.trend
                _series = exog - exog[0]
                mod = sm.OLS(_series, np.arange(len(_series)))
                _res = mod.fit()
                # trend magnitude
                seasonality_features['trend_mag'] = _res.params[0]

            # residual standard deviation
            if extra_args.get('residual_std', default_status):
                seasonality_features['residual_std'] = np.std(res.resid)

            return seasonality_features
        except Exception:
            return seasonality_features
