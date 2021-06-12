#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
import logging
import statsmodels
from functools import partial
from itertools import groupby
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import statsmodels.api as sm
from kats.consts import TimeSeriesData
from kats.detectors import (
    cusum_detection,
    bocpd,
    robust_stat_detection,
    outlier,
    trend_mk,
    seasonality,
)
from numba import jit  # @manual
from scipy import stats
from scipy.signal import periodogram  # @manual
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, pacf, kpss

"""TsFeatures is a module for performing adhoc feature engineering on time series
data using different statistics.

The module process time series data into features for machine learning models.
We include seasonality, autocorrelation, modeling parameter, changepoints,
moving statistics, and raw statistics of time series array as the adhoc features.

We also offer to compute part of the features or group of features using
selected_features argument, you could also disable feature or group of
features by setting feature_name/feature_group_name = False. You can find
all feature group names in feature_group_mapping attribute.
"""


class TsFeatures:
    """Process time series data into features for machine learning models, with the
    function to opt-in/out feature and feature groups in the calculations.

    Attributes:
        window_size: int; Length of the sliding window for getting level shift features,
            lumpiness, and stability of time series.
        spectral_freq: int; Frequency parameter in getting periodogram through scipy for
            calculating Shannon entropy.
        stl_period: int; Period parameter for performing seasonality trend decomposition
            using LOESS with statsmodels.
        nbins: int; Number of bins to equally segment time series array for getting flat
            spot feature.
        lag_size: int; Maximum number of lag values for calculating Hurst Exponent.
        acfpacf_lag: int; Largest lag number for returning ACF/PACF features via statsmodels.
        decomp: str; Additive or Multiplicative mode for performing outlier detection using
            Kats.Detectors.outlier.OutlierDetector.
        iqr_mult: float; IQR range for determining outliers through
            Kats.Detectors.outlier.OutlierDetector.
        threshold: float; threshold for trend intensity; higher threshold gives
            trend with high intensity (0.8 by default).  If we only want to use the
            p-value to determine changepoints, set threshold = 0.
        window: int; length of window for all nowcasting features.
        n_fast: int; length of "fast" or short period exponential moving average in the MACD
            algorithm in the nowcasting features.
        n_slow: int; length of "slow" or long period exponential moving average in the MACD
            algorithm in the nowcasting features.
        selected_features: None or List[str]; list of feature/feature group name(s)
            selected to be calculated. We will try only calculating selected
            features, since some features are bundled in the calculations. This process
            helps with boosting efficiency, and we will only output selected features.
        feature_group_mapping: The dictionary with the mapping from individual features
            to their bundled feature groups.
        final_filter: A dicitonary with boolean as the values to filter out the features
            not selected, yet calculated due to underlying bundles.
        stl_features: Switch for calculating/outputting stl features.
        level_shift_features: Switch for calculating/outputting level shift features.
        acfpacf_features: Switch for calculating/outputting ACF/PACF features.
        special_ac: Switch for calculating/outputting  features.
        holt_params: Switch for calculating/outputting holt parameter features.
        hw_params: Switch for calculating/outputting holt-winters parameter features.
        statistics: Switch for calculating/outputting raw statistics features.
        cusum_detector: Switch for calculating/outputting features using cusum detector
            in Kats.
        robust_stat_detector: Switch for calculating/outputting features using robust stat detector
            in Kats.
        bocp_detector: Switch for calculating/outputting stl features features using bocp detector
            in Kats.
        outlier_detector: Switch for calculating/outputting stl features using outlier detector
            in Kats.
        trend_detector: Switch for calculating/outputting stl features using trend detector
            in Kats.
        nowcasting: Switch for calculating/outputting stl features using nowcasting detector
            in Kats.
        seasonalities: Switch for calculating/outputting stl features using cusum detector
            in Kats.
        default: The default status of the switch for opt-in/out feature calculations.
    """

    def __init__(
        self,
        window_size: int = 20,
        spectral_freq: int = 1,
        stl_period: int = 7,
        nbins: int = 10,
        lag_size: int = 30,
        acfpacf_lag: int = 6,
        decomp: str = "additive",
        iqr_mult: float = 3.0,
        threshold: float = 0.8,
        window: int = 5,
        n_fast: int = 12,
        n_slow: int = 21,
        selected_features: Optional[List[str]] = None,
        **kwargs,
    ):
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
                "nowcast_roc",
                "nowcast_ma",
                "nowcast_mom",
                "nowcast_lag",
                "nowcast_macd",
                "nowcast_macdsign",
                "nowcast_macddiff",
            ],
            "seasonalities": [
                "seasonal_period",
                "trend_mag",
                "seasonality_mag",
                "residual_std",
            ],
        }
        self.feature_group_mapping = g2f
        f2g = {}
        for k, v in g2f.items():
            for f in v:
                f2g[f] = k

        self._total_feature_len_ = len(f2g.keys())
        for f in kwargs.keys():
            assert (
                f in f2g.keys() or f in g2f.keys()
            ), f"""couldn't find your desired feature/group "{f}", please check spelling"""

        # Higher level of features:
        # Once disabled, won't even go inside these groups of features
        # for calculation
        if not selected_features:
            default = True
            self.final_filter = {k: default for k in f2g.keys()}
        elif selected_features:
            default = False
            self.final_filter = {k: default for k in f2g.keys()}
            for f in selected_features:
                assert (
                    f in f2g.keys() or f in g2f.keys()
                ), f"""couldn't find your desired feature/group "{f}", please check spelling"""
                if f in g2f.keys():  # the opt-in request is for a feature group
                    kwargs[f] = True
                    for feature in g2f[f]:
                        kwargs[feature] = kwargs.get(feature, True)
                        self.final_filter[feature] = True
                elif f in f2g.keys():  # the opt-in request is for a certain feature
                    assert kwargs.get(
                        f2g[f], True
                    ), f"""feature group: {f2g[f]} has to be opt-in based on your opt-in request of feature: {f}"""
                    assert kwargs.get(
                        f, True
                    ), f"""you have requested to both opt-in and opt-out feature: {f}"""
                    kwargs[f2g[f]] = True  # need to opt-in the feature group first
                    kwargs[f] = True  # opt-in the feature
                    self.final_filter[f] = True

        for k, v in kwargs.items():
            self.final_filter[
                k
            ] = v  # final filter for filtering out features user didn't request and keep only the requested ones

        # setting default value for the switches of calculating the group of features or not
        self.stl_features = kwargs.get("stl_features", default)
        self.level_shift_features = kwargs.get("level_shift_features", default)
        self.acfpacf_features = kwargs.get("acfpacf_features", default)
        self.special_ac = kwargs.get("special_ac", default)
        self.holt_params = kwargs.get("holt_params", default)
        self.hw_params = kwargs.get("hw_params", default)
        self.statistics = kwargs.get("statistics", default)
        self.cusum_detector = kwargs.get("cusum_detector", False)
        self.robust_stat_detector = kwargs.get("robust_stat_detector", False)
        self.bocp_detector = kwargs.get("bocp_detector", False)
        self.outlier_detector = kwargs.get("outlier_detector", False)
        self.trend_detector = kwargs.get("trend_detector", False)
        self.nowcasting = kwargs.get("nowcasting", False)
        self.seasonalities = kwargs.get("seasonalities", False)

        # For lower level of the features
        self.__kwargs__ = kwargs
        self.default = default

    def transform(self, x: TimeSeriesData):
        """
        The overall high-level function for transforming
        time series into a number of features

        Args:
            x: Kats TimeSeriesData object.

        Returns:
            Returning maps (dictionary) with feature name and value pair.
            For univariate input return a map of {feature: value}.
            For multivarite input return a list of maps.
        """

        if len(x) < 5:
            msg = "Length of time series is too short to calculate features!"
            logging.error(msg)
            raise ValueError(msg)

        if type(x.value.values) != np.ndarray:
            logging.warning(
                f"expecting values to be a np.ndarray, instead got {type(x.value.values)}"
            )
            # make sure that values are numpy array for feeding to Numba
            df = pd.DataFrame(
                {"time": x.time.values, "value": np.array(x.value.values, dtype=float)}
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
                ts_features.append(self._transform_1d(ts_values, x.value[col]))

        # performing final filter
        to_remove = []
        for feature in ts_features:
            if not self.final_filter[feature]:
                to_remove.append(feature)

        for r in to_remove:
            del ts_features[r]

        return ts_features

    def _transform_1d(self, x: np.ndarray, ts: TimeSeriesData):
        """
        Transform single (univariate) time series

        Args:
            x: The univariate time series array in the form of 1d numpy array.
            ts: The univariate time series array in the form of Kats Timeseries
                Data object.

        Returns:
            The dictionary with all the features aggregated from the outputs of
            each feature group calculator.
        """

        # calculate STL based features
        dict_stl_features = {}
        if self.stl_features:
            dict_stl_features = self.get_stl_features(
                x,
                period=self.stl_period,
                extra_args=self.__kwargs__,
                default_status=self.default,
            )

        # calculate level shift based features
        dict_level_shift_features = {}
        if self.level_shift_features:
            dict_level_shift_features = self.get_level_shift(
                x,
                window_size=self.window_size,
                extra_args=self.__kwargs__,
                default_status=self.default,
            )

        # calculate ACF, PACF based features
        dict_acfpacf_features = {}
        if self.acfpacf_features:
            dict_acfpacf_features = self.get_acfpacf_features(
                x,
                acfpacf_lag=self.acfpacf_lag,
                period=self.stl_period,
                extra_args=self.__kwargs__,
                default_status=self.default,
            )

        # calculate special AC
        dict_specialac_features = {}
        if self.special_ac:
            dict_specialac_features = self.get_special_ac(
                x, extra_args=self.__kwargs__, default_status=self.default
            )

        # calculate holt params
        dict_holt_params_features = {}
        if self.holt_params:
            dict_holt_params_features = self.get_holt_params(
                x, extra_args=self.__kwargs__, default_status=self.default
            )

        # calculate hw params
        dict_hw_params_features = {}
        if self.hw_params:
            dict_hw_params_features = self.get_hw_params(
                x,
                period=self.stl_period,
                extra_args=self.__kwargs__,
                default_status=self.default,
            )

        # single features
        _dict_features_ = {}
        if self.statistics:
            dict_features = {
                "length": partial(self.get_length),
                "mean": partial(self.get_mean),
                "var": partial(self.get_var),
                "entropy": partial(self.get_spectral_entropy, freq=self.spectral_freq),
                "lumpiness": partial(self.get_lumpiness, window_size=self.window_size),
                "stability": partial(self.get_stability, window_size=self.window_size),
                "flat_spots": partial(self.get_flat_spots, nbins=self.nbins),
                "hurst": partial(self.get_hurst, lag_size=self.lag_size),
                "std1st_der": partial(self.get_std1st_der),
                "crossing_points": partial(self.get_crossing_points),
                "binarize_mean": partial(self.get_binarize_mean),
                "unitroot_kpss": partial(self.get_unitroot_kpss),
                "heterogeneity": partial(self.get_het_arch),
                "histogram_mode": partial(self.get_histogram_mode, nbins=self.nbins),
                "linearity": partial(self.get_linearity),
            }

            _dict_features_ = {}
            for k, v in dict_features.items():
                if self.__kwargs__.get(k, self.default):
                    _dict_features_[k] = v(x)

        # calculate cusum detector features
        dict_cusum_detector_features = {}
        if self.cusum_detector:
            dict_cusum_detector_features = self.get_cusum_detector(
                ts, extra_args=self.__kwargs__, default_status=self.default
            )

        # calculate robust stat detector features
        dict_robust_stat_detector_features = {}
        if self.robust_stat_detector:
            dict_robust_stat_detector_features = self.get_robust_stat_detector(
                ts, extra_args=self.__kwargs__, default_status=self.default
            )

        # calculate bocp detector features
        dict_bocp_detector_features = {}
        if self.bocp_detector:
            dict_bocp_detector_features = self.get_bocp_detector(
                ts, extra_args=self.__kwargs__, default_status=self.default
            )

        # calculate outlier detector features
        dict_outlier_detector_features = {}
        if self.outlier_detector:
            dict_outlier_detector_features = self.get_outlier_detector(
                ts,
                decomp=self.decomp,
                iqr_mult=self.iqr_mult,
                extra_args=self.__kwargs__,
                default_status=self.default,
            )

        # calculate trend detector features
        dict_trend_detector_features = {}
        if self.trend_detector:
            dict_trend_detector_features = self.get_trend_detector(
                ts,
                threshold=self.threshold,
                extra_args=self.__kwargs__,
                default_status=self.default,
            )

        # calculate nowcasting features
        dict_nowcasting_features = {}
        if self.nowcasting:
            dict_nowcasting_features = self.get_nowcasting(
                x,
                window=self.window,
                n_fast=self.n_fast,
                n_slow=self.n_slow,
                extra_args=self.__kwargs__,
                default_status=self.default,
            )

        # calculate seasonality features
        dict_seasonality_features = {}
        if self.seasonalities:
            dict_seasonality_features = self.get_seasonalities(
                ts, extra_args=self.__kwargs__, default_status=self.default
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
    def get_length(x: np.ndarray):
        """
        Getting the length of time series array.

        Args:
            x: The univariate time series array in the form of 1d numpy array.

        Returns:
            Length of the time series array.
        """

        return len(x)

    # mean
    @staticmethod
    @jit(nopython=True)
    def get_mean(x: np.ndarray):
        """
        Getting the average value of time series array.

        Args:
            x: The univariate time series array in the form of 1d numpy array.

        Returns:
            Average of the time series array.
        """

        return np.mean(x)

    # variance
    @staticmethod
    @jit(nopython=True)
    def get_var(x: np.ndarray):
        """
        Getting the variance of time series array.

        Args:
            x: The univariate time series array in the form of 1d numpy array.

        Returns:
            Variance of the time series array.
        """

        return np.var(x)

    # spectral entropy
    @staticmethod
    @jit(forceobj=True)
    def get_spectral_entropy(x: np.ndarray, freq: int = 1):
        """
        Getting normalized Shannon entropy of power spectral density.
        PSD is calculated using scipy's periodogram.

        Args:
            x: The univariate time series array in the form of 1d numpy array.
            freq: int; Frequency for calculating the PSD via scipy periodogram.
                Default value is 1.

        Returns:
            Normalized Shannon entropy.
        """

        # calculate periodogram
        _, psd = periodogram(x, freq)

        # calculate shannon entropy of normalized psd
        psd_norm = psd / np.sum(psd)
        entropy = np.nansum(psd_norm * np.log2(psd_norm))

        return -(entropy / np.log2(psd_norm.size))

    # lumpiness
    @staticmethod
    @jit(forceobj=True)
    def get_lumpiness(x: np.ndarray, window_size: int = 20):
        """
        Calculating the lumpiness of time series.
        Lumpiness is defined as the variance of the chunk-wise variances.

        Args:
            x: The univariate time series array in the form of 1d numpy array.
            window_size: int; Window size to split the data into chunks for getting
                variances. Default value is 20.

        Returns:
            Lumpiness of the time series array.
        """

        v = [np.var(x_w) for x_w in np.array_split(x, len(x) // window_size + 1)]
        return np.var(v)

    # stability
    @staticmethod
    @jit(forceobj=True)
    def get_stability(x: np.ndarray, window_size: int = 20):
        """
        Calculating the stability of time series.
        Stability is defined as the variance of chunk-wise means.

        Args:
            x: The univariate time series array in the form of 1d numpy array.
            window_size: int; Window size to split the data into chunks for getting
                variances. Default value is 20.

        Returns:
            Stability of the time series array.
        """

        v = [np.mean(x_w) for x_w in np.array_split(x, len(x) // window_size + 1)]
        return np.var(v)

    # STL decomposition based features
    @staticmethod
    @jit(forceobj=True)
    def get_stl_features(
        x: np.ndarray,
        period: int = 7,
        extra_args: Optional[Dict[str, bool]] = None,
        default_status: bool = True,
    ):
        """
        Calculate STL based features for a time series, including strength of
        trend, seasonality, spikiness, peak/trough.

        Args:
            x: The univariate time series array in the form of 1d numpy array.
            period: int; Period parameter for performing seasonality trend decomposition
                using LOESS with statsmodels. Default value is 7.
            extra_args: A dictionary containing information for disabling calculation
                of a certain feature. Default value is None, i.e. no feature is disabled.
            default_status: Default status of the switch for calculate the features or not.
                Default value is True.

        Returns:
            Seasonality features including strength of trend, seasonality, spikiness,
            peak/trough.
        """

        stl_features = {}

        # STL decomposition
        res = STL(x, period=period).fit()

        # strength of trend
        if extra_args is not None and extra_args.get("trend_strength", default_status):
            stl_features["trend_strength"] = 1 - np.var(res.resid) / np.var(
                res.trend + res.resid
            )

        # strength of seasonality
        if extra_args is not None and extra_args.get("seasonality_strength", default_status):
            stl_features["seasonality_strength"] = 1 - np.var(res.resid) / np.var(
                res.seasonal + res.resid
            )

        # spikiness: variance of the leave-one-out variances of the remainder component
        if extra_args is not None and extra_args.get("spikiness", default_status):
            resid_array = np.repeat(
                np.array(res.resid)[:, np.newaxis], len(res.resid), axis=1
            )
            resid_array[np.diag_indices(len(resid_array))] = np.NaN
            stl_features["spikiness"] = np.var(np.nanvar(resid_array, axis=0))

        # location of peak
        if extra_args is not None and extra_args.get("peak", default_status):
            stl_features["peak"] = np.argmax(res.seasonal[:period])

        # location of trough
        if extra_args is not None and extra_args.get("trough", default_status):
            stl_features["trough"] = np.argmin(res.seasonal[:period])

        return stl_features

    # Level shift
    @staticmethod
    @jit(forceobj=True)
    def get_level_shift(
        x: np.ndarray,
        window_size: int = 20,
        extra_args: Optional[Dict[str, bool]] = None,
        default_status: bool = True,
    ):
        """
        Calculating level shift features.
        level_shift_idx: Location of the maximum mean value difference,
            between two consecutive sliding windows
        level_shift_size: Size of the maximum mean value difference,
            between two consecutive sliding windows

        Args:
            x: The univariate time series array in the form of 1d numpy array.
            window_size: int; Length of the sliding window. Default value is 20.
            extra_args: A dictionary containing information for disabling calculation
                of a certain feature. Default value is None, i.e. no feature is disabled.
            default_status: Default status of the switch for calculate the features or not.
                Default value is True.

        Returns:
            Level shift features including level_shift_idx, and level_shift_size
        """

        level_shift_features = {"level_shift_idx": np.nan, "level_shift_size": np.nan}
        if len(x) < window_size + 2:
            msg = "Length of time series is shorter than window_size, unable to calculate level shift features!"
            logging.error(msg)
            return level_shift_features

        sliding_idx = (np.arange(len(x))[None, :] + np.arange(window_size)[:, None])[
            :, : len(x) - window_size + 1
        ]
        means = np.mean(x[sliding_idx], axis=0)
        mean_diff = np.abs(means[:-1] - means[1:])

        if extra_args is not None and extra_args.get("level_shift_idx", default_status):
            level_shift_features["level_shift_idx"] = np.argmax(mean_diff)
        if extra_args is not None and extra_args.get("level_shift_size", default_status):
            level_shift_features["level_shift_size"] = mean_diff[np.argmax(mean_diff)]
        return level_shift_features

    # Flat spots
    @staticmethod
    @jit(forceobj=True)
    def get_flat_spots(x: np.ndarray, nbins: int = 10):
        """
        Getting flat spots: Maximum run-lengths across equally-sized segments of time series

        Args:
            x: The univariate time series array in the form of 1d numpy array.
            nbins: int; Number of bins to segment time series data into.

        Returns:
            Maximum run-lengths across segmented time series array.
        """

        if len(x) <= nbins:
            msg = "Length of time series is shorter than nbins, unable to calculate flat spots feature!"
            logging.error(msg)
            return np.nan

        max_run_length = 0
        window_size = int(len(x) / nbins)
        for i in range(0, len(x), window_size):
            run_length = np.max(
                [len(list(v)) for k, v in groupby(x[i : i + window_size])]
            )
            if run_length > max_run_length:
                max_run_length = run_length
        return max_run_length

    # Hurst Exponent
    @staticmethod
    @jit(forceobj=True)
    def get_hurst(x: np.ndarray, lag_size: int = 30):
        """
        Getting: Hurst Exponent wiki: https://en.wikipedia.org/wiki/Hurst_exponent

        Args:
            x: The univariate time series array in the form of 1d numpy array.
            lag_size: int; Size for getting lagged time series data. Default value
                is 30.

        Returns:
            The Hurst Exponent of the time series array
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
    def get_acf_features(
        extra_args: Dict[str, bool],
        default_status: bool,
        y_acf_list: List[float],
        diff1y_acf_list: List[float],
        diff2y_acf_list: List[float],
    ):
        """
        Aggregating extracted ACF features from get_acfpacf_features function.

        Args:
            extra_args: A dictionary containing information for disabling calculation
                of a certain feature. Default value is None, i.e. no feature is disabled.
            default_status: Default status of the switch for calculate the features or not.
                Default value is True.
            y_acf_list: List of ACF values acquired from original time series.
            diff1y_acf_list: List of ACF values acquired from differenced time series.
            diff2y_acf_list: List of ACF values acquired from twice differenced time series.

        Returns:
            Auto-correlation function (ACF) features.
        """

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
        if extra_args.get("y_acf1", default_status):
            y_acf1 = y_acf_list[0]

        # y_acf5: sum of squares of first 5 ACF values of original series
        if extra_args.get("y_acf5", default_status):
            y_acf5 = np.sum(np.asarray(y_acf_list)[:5] ** 2)

        # diff1y_acf1: first ACF value of the differenced series
        if extra_args.get("diff1y_acf1", default_status):
            diff1y_acf1 = diff1y_acf_list[0]

        # diff1y_acf5: sum of squares of first 5 ACF values of differenced series
        if extra_args.get("diff1y_acf5", default_status):
            diff1y_acf5 = np.sum(np.asarray(diff1y_acf_list)[:5] ** 2)

        # diff2y_acf1: first ACF value of the twice-differenced series
        if extra_args.get("diff2y_acf1", default_status):
            diff2y_acf1 = diff2y_acf_list[0]

        # diff2y_acf5: sum of squares of first 5 ACF values of twice-differenced series
        if extra_args.get("diff2y_acf5", default_status):
            diff2y_acf5 = np.sum(np.asarray(diff2y_acf_list)[:5] ** 2)

        # Autocorrelation coefficient at the first seasonal lag.
        if extra_args.get("seas_acf1", default_status):
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
    def get_pacf_features(
        extra_args: Dict[str, bool],
        default_status: bool,
        y_pacf_list: List[float],
        diff1y_pacf_list: List[float],
        diff2y_pacf_list: List[float],
    ):
        """
        Aggregating extracted PACF features from get_acfpacf_features function.

        Args:
            extra_args: A dictionary containing information for disabling calculation
                of a certain feature. Default value is None, i.e. no feature is disabled.
            default_status: Default status of the switch for calculate the features or not.
                Default value is True.
            y_pacf_list: List of PACF values acquired from original time series.
            diff1y_pacf_list: List of PACF values acquired from differenced time series.
            diff2y_pacf_list: List of PACF values acquired from twice differenced time series.

        Returns:
            Partial auto-correlation function (PACF) features.
        """

        (
            y_pacf5,
            diff1y_pacf5,
            diff2y_pacf5,
            seas_pacf1,
        ) = (np.nan, np.nan, np.nan, np.nan)

        # y_pacf5: sum of squares of first 5 PACF values of original series
        if extra_args.get("y_pacf5", default_status):
            y_pacf5 = np.nansum(np.asarray(y_pacf_list)[:5] ** 2)

        # diff1y_pacf5: sum of squares of first 5 PACF values of differenced series
        if extra_args.get("diff1y_pacf5", default_status):
            diff1y_pacf5 = np.nansum(np.asarray(diff1y_pacf_list)[:5] ** 2)

        # diff2y_pacf5: sum of squares of first 5 PACF values of twice-differenced series
        if extra_args.get("diff2y_pacf5", default_status):
            diff2y_pacf5 = np.nansum(np.asarray(diff2y_pacf_list)[:5] ** 2)

        # Patial Autocorrelation coefficient at the first seasonal lag.
        if extra_args.get("seas_pacf1", default_status):
            seas_pacf1 = y_pacf_list[-1]

        return (
            y_pacf5,
            diff1y_pacf5,
            diff2y_pacf5,
            seas_pacf1,
        )

    @staticmethod
    @jit(forceobj=True)
    def get_acfpacf_features(
        x: np.ndarray,
        acfpacf_lag: int = 6,
        period: int = 7,
        extra_args: Optional[Dict[str, bool]] = None,
        default_status: bool = True,
    ):
        """
        Calculate ACF and PACF based features. Calculate seasonal ACF, PACF based features
        Reference: https://stackoverflow.com/questions/36038927/whats-the-difference-between-pandas-acf-and-statsmodel-acf
        R code: https://cran.r-project.org/web/packages/tsfeatures/vignettes/tsfeatures.html
        Paper: Meta-learning how to forecast time series

        Args:
            x: The univariate time series array in the form of 1d numpy array.
            acfpacf_lag: int; Largest lag number for returning ACF/PACF features via statsmodels.
                Default value is 6.
            period: int; Seasonal period. Default value is 7.
            extra_args: A dictionary containing information for disabling calculation
                of a certain feature. Default value is None, i.e. no feature is disabled.
            default_status: Default status of the switch for calculate the features or not.
                Default value is True.

        Returns:
            Aggregated ACF, PACF features.
        """

        acfpacf_features = {
            "y_acf1": np.nan,
            "y_acf5": np.nan,
            "diff1y_acf1": np.nan,
            "diff1y_acf5": np.nan,
            "diff2y_acf1": np.nan,
            "diff2y_acf5": np.nan,
            "y_pacf5": np.nan,
            "diff1y_pacf5": np.nan,
            "diff2y_pacf5": np.nan,
            "seas_acf1": np.nan,
            "seas_pacf1": np.nan,
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
            acfpacf_features["y_acf1"],
            acfpacf_features["y_acf5"],
            acfpacf_features["diff1y_acf1"],
            acfpacf_features["diff1y_acf5"],
            acfpacf_features["diff2y_acf1"],
            acfpacf_features["diff2y_acf5"],
            acfpacf_features["seas_acf1"],
        ) = TsFeatures.get_acf_features(
            extra_args,
            default_status,
            y_acf_list,
            diff1y_acf_list,
            diff2y_acf_list,
        )

        # getting PACF features
        (
            acfpacf_features["y_pacf5"],
            acfpacf_features["diff1y_pacf5"],
            acfpacf_features["diff2y_pacf5"],
            acfpacf_features["seas_pacf1"],
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
    def get_std1st_der(x: np.ndarray):
        """
        Calculating std1st_der: the standard deviation of the first derivative of the time series.
        Reference: https://cran.r-project.org/web/packages/tsfeatures/vignettes/tsfeatures.html

        Args:
            x: The univariate time series array in the form of 1d numpy array.

        Returns:
            The standard deviation of the first derivative of the time series.
        """

        std1st_der = np.std(np.gradient(x))
        return std1st_der

    # crossing points
    @staticmethod
    @jit(nopython=True)
    def get_crossing_points(x: np.ndarray):
        """
        Calculating crossing points: the number of times a time series crosses the median line.
        Reference: https://cran.r-project.org/web/packages/tsfeatures/vignettes/tsfeatures.html

        Args:
            x: The univariate time series array in the form of 1d numpy array.

        Returns:
            The number of times a time series crosses the median line.
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
    def get_binarize_mean(x: np.ndarray):
        """
        Converts time series array into a binarized version.
        Time-series values above its mean are given 1, and those below the mean are 0.
        Return the average value of the binarized vector.
        Reference: https://cran.r-project.org/web/packages/tsfeatures/vignettes/tsfeatures.html

        Args:
            x: The univariate time series array in the form of 1d numpy array.

        Returns:
            The binarized version of time series array.
        """

        return np.mean(np.asarray(x) > np.mean(x))

    # KPSS unit root test
    @staticmethod
    @jit(forceobj=True)
    def get_unitroot_kpss(x: np.ndarray):
        """
        Getting a test statistic based on KPSS test.
        Test a null hypothesis that an observable time series is stationary around a deterministic trend.
        A vector comprising the statistic for the KPSS unit root test with linear trend and lag one
        Wiki: https://en.wikipedia.org/wiki/KPSS_test

        Args:
            x: The univariate time series array in the form of 1d numpy array.

        Returns:
            Test statistics acquired using KPSS test.
        """

        return kpss(x, regression="ct", nlags=1)[0]

    # heterogeneity
    @staticmethod
    @jit(forceobj=True)
    def get_het_arch(x: np.ndarray):
        """
        reference: https://www.statsmodels.org/dev/generated/statsmodels.stats.diagnostic.het_arch.html
        Engle’s Test for Autoregressive Conditional Heteroscedasticity (ARCH)

        Args:
            x: The univariate time series array in the form of 1d numpy array.

        Returns:
            Lagrange multiplier test statistic
        """

        return het_arch(x, nlags=min(10, len(x) // 5))[0]

    # histogram mode
    @staticmethod
    @jit(nopython=True)
    def get_histogram_mode(x: np.ndarray, nbins: int = 10):
        """
        Measures the mode of the data vector using histograms with a given number of bins.
        Reference: https://cran.r-project.org/web/packages/tsfeatures/vignettes/tsfeatures.html

        Args:
            x: The univariate time series array in the form of 1d numpy array.
            nbins: int; Number of bins to get the histograms. Default value is 10.

        Returns:
            Mode of the data vector using histograms.
        """

        cnt, val = np.histogram(x, bins=nbins)
        return val[cnt.argmax()]

    # First min/zero AC (2)
    @staticmethod
    @jit(forceobj=True)
    def get_special_ac(
        x: np.ndarray, extra_args: Optional[Dict[str, bool]] = None, default_status: bool = True
    ):
        """
        Gettting special_ac features.
        firstmin_ac: the time of first minimum in the autocorrelation function
        firstzero_ac: the time of first zero crossing the autocorrelation function.

        Args:
            x: The univariate time series array in the form of 1d numpy array.
            extra_args: A dictionary containing information for disabling calculation
                of a certain feature. Default value is None, i.e. no feature is disabled.
            default_status: Default status of the switch for calculate the features or not.
                Default value is True.

        Returns:
            Special autocorrelation features described above.
        """

        # First min AC
        special_ac_features = {"firstmin_ac": np.nan, "firstzero_ac": np.nan}
        AC = acf(x, unbiased=False, fft=True, nlags=len(x))[1:]
        if extra_args is not None and extra_args.get("firstmin_ac", default_status):
            i = 0
            while i < len(AC) - 1:
                if AC[i] > AC[i + 1]:
                    i += 1
                else:
                    break
            special_ac_features["firstmin_ac"] = i + 1

        # First zero AC
        if extra_args is not None and extra_args.get("firstzero_ac", default_status):
            j = 0
            while j < len(AC) - 1:
                if AC[j] > 0 > AC[j + 1]:
                    break
                else:
                    j += 1
            special_ac_features["firstzero_ac"] = j + 2
        return special_ac_features

    # Linearity
    @staticmethod
    @jit(forceobj=True)
    def get_linearity(x: np.ndarray):
        """
        Getting linearity feature: R square from a fitted linear regression.

        Args:
            x: The univariate time series array in the form of 1d numpy array.

        Returns:
            R square from a fitted linear regression.
        """

        _, _, r_value, _, _ = stats.linregress(np.arange(len(x)), x)
        return r_value ** 2

    # Holt Parameters (2)
    @staticmethod
    def get_holt_params(
        x: np.ndarray, extra_args: Optional[Dict[str, bool]] = None, default_status: bool = True
    ):
        """
        Estimates the smoothing parameter for the level-alpha and the smoothing parameter
        for the trend-beta of Holt’s linear trend method.
        'alpha': Level parameter of the Holt model.
        'beta': Trend parameter of the Hold model.

        Args:
            x: The univariate time series array in the form of 1d numpy array.
            extra_args: A dictionary containing information for disabling calculation
                of a certain feature. Default value is None, i.e. no feature is disabled.
            default_status: Default status of the switch for calculate the features or not.
                Default value is True.

        Returns:
            Level and trend parameter of a fitted Holt model.
        """

        holt_params_features = {"holt_alpha": np.nan, "holt_beta": np.nan}
        try:
            m = ExponentialSmoothing(x, trend="add", seasonal=None).fit()
            if extra_args is not None and extra_args.get("holt_alpha", default_status):
                holt_params_features["holt_alpha"] = m.params["smoothing_level"]
            if extra_args is not None and extra_args.get("holt_beta", default_status):
                statsmodels_ver = float(re.findall('([0-9]+\\.[0-9]+)\\..*', statsmodels.__version__)[0])
                if statsmodels_ver < 0.12:
                    holt_params_features["holt_beta"] = m.params["smoothing_slope"]
                elif statsmodels_ver >= 0.12:
                    holt_params_features["holt_beta"] = m.params["smoothing_trend"]
            return holt_params_features
        except Exception:
            return holt_params_features

    # Holt Winter’s Parameters (3)
    @staticmethod
    def get_hw_params(
        x: np.ndarray,
        period: int = 7,
        extra_args: Optional[Dict[str, bool]] = None,
        default_status: bool = True,
    ):
        """
        Estimates the smoothing parameter for the level-alpha, trend-beta of HW’s linear trend,
        and additive seasonal trend-gamma.

        Args:
            x: The univariate time series array in the form of 1d numpy array.
            period: int; Seaonal period for fitting exponential smoothing model. Default
                value is 7.
            extra_args: A dictionary containing information for disabling calculation
                of a certain feature. Default value is None, i.e. no feature is disabled.
            default_status: Default status of the switch for calculate the features or not.
                Default value is True.

        Returns:
            Level, trend and seasonal parameter of a fitted Holt-Winter's model.
        """

        hw_params_features = {"hw_alpha": np.nan, "hw_beta": np.nan, "hw_gamma": np.nan}
        try:
            # addressing issue of use_boxcox arg in different versions of statsmodels
            statsmodels_ver = float(re.findall('([0-9]+\\.[0-9]+)\\..*', statsmodels.__version__)[0])
            _args_ = {
                "seasonal_periods": period,
                "trend": "add",
                "seasonal": "add",
            }
            # performing version check on statsmodels
            if statsmodels_ver >= 0.12:
                _args_["use_boxcox"] = True
                _args_["initialization_method"] = 'estimated'
                m = ExponentialSmoothing(x, **_args_).fit()
            elif statsmodels_ver < 0.12:
                m = ExponentialSmoothing(x, **_args_).fit(use_boxcox = True)
            if extra_args is not None and extra_args.get("hw_alpha", default_status):
                hw_params_features["hw_alpha"] = m.params["smoothing_level"]
            if extra_args is not None and extra_args.get("hw_beta", default_status):
                if statsmodels_ver < 0.12:
                    hw_params_features["hw_beta"] = m.params["smoothing_slope"]
                elif statsmodels_ver >= 0.12:
                    hw_params_features["hw_beta"] = m.params["smoothing_trend"]
            if extra_args is not None and extra_args.get("hw_gamma", default_status):
                hw_params_features["hw_gamma"] = m.params["smoothing_seasonal"]
            return hw_params_features
        except Exception:
            return hw_params_features

    # CUSUM Detection Outputs (8)
    @staticmethod
    def get_cusum_detector(
        ts: TimeSeriesData, extra_args: Optional[Dict[str, bool]] = None, default_status: bool = True
    ):
        """
        Run the Kats CUSUM Detector on the Time Series, extract features from the outputs of the detection.

        Args:
            ts: The univariate time series array in the form of Kats TimeSeriesData object.
            extra_args: A dictionary containing information for disabling calculation
                of a certain feature. Default value is None, i.e. no feature is disabled.
            default_status: Default status of the switch for calculate the features or not.
                Default value is True.

        Returns:
            Outputs of the CUSUM Detector, which include (1) Number of changepoints, either 1
            or 0, (2) Confidence of the changepoint detected, 0 if not changepoint, (3) index,
            or position of the changepoint detected within the time series, (4) delta of the
            mean levels before and after the changepoint, (5) log-likelihood ratio of changepoint,
            (6) Boolean - whether regression is detected by CUSUM, (7) Boolean - whether
            changepoint is stable, (8) p-value of changepoint.
        """

        cusum_detector_features = {
            "cusum_num": np.nan,
            "cusum_conf": np.nan,
            "cusum_cp_index": np.nan,
            "cusum_delta": np.nan,
            "cusum_llr": np.nan,
            "cusum_regression_detected": np.nan,
            "cusum_stable_changepoint": np.nan,
            "cusum_p_value": np.nan,
        }
        try:
            cusum = cusum_detection.CUSUMDetector(ts)
            cusum_cp = cusum.detector()
            if extra_args is not None and extra_args.get("cusum_num", default_status):
                cusum_detector_features["cusum_num"] = len(cusum_cp)
            if extra_args is not None and extra_args.get("cusum_conf", default_status):
                cusum_detector_features["cusum_conf"] = (
                    0 if len(cusum_cp) == 0 else cusum_cp[0][0].confidence
                )
            if extra_args is not None and extra_args.get("cusum_cp_index", default_status):
                cusum_detector_features["cusum_cp_index"] = (
                    0 if len(cusum_cp) == 0 else cusum_cp[0][1]._cp_index / len(ts)
                )
            if extra_args is not None and extra_args.get("cusum_delta", default_status):
                cusum_detector_features["cusum_delta"] = (
                    0 if len(cusum_cp) == 0 else cusum_cp[0][1]._delta
                )
            if extra_args is not None and extra_args.get("cusum_llr", default_status):
                cusum_detector_features["cusum_llr"] = (
                    0 if len(cusum_cp) == 0 else cusum_cp[0][1]._llr
                )
            if extra_args is not None and extra_args.get("cusum_regression_detected", default_status):
                cusum_detector_features["cusum_regression_detected"] = (
                    False if len(cusum_cp) == 0 else cusum_cp[0][1]._regression_detected
                )
            if extra_args is not None and extra_args.get("cusum_stable_changepoint", default_status):
                cusum_detector_features["cusum_stable_changepoint"] = (
                    False if len(cusum_cp) == 0 else cusum_cp[0][1]._stable_changepoint
                )
            if extra_args is not None and extra_args.get("cusum_p_value", default_status):
                cusum_detector_features["cusum_p_value"] = (
                    0 if len(cusum_cp) == 0 else cusum_cp[0][1]._p_value
                )
            return cusum_detector_features
        except Exception:
            return cusum_detector_features

    # Robust Stat Detection Outputs (2)
    @staticmethod
    def get_robust_stat_detector(
        ts: TimeSeriesData, extra_args: Optional[Dict[str, bool]] = None, default_status: bool = True
    ):
        """
        Run the Kats Robust Stat Detector on the Time Series, extract features from the outputs of the detection.

        Args:
            ts: The univariate time series array in the form of Kats TimeSeriesData object.
            extra_args: A dictionary containing information for disabling calculation
                of a certain feature. Default value is None, i.e. no feature is disabled.
            default_status: Default status of the switch for calculate the features or not.
                Default value is True.

        Returns:
            (1) Number changepoints detected by the Robust Stat Detector, and (2) Mean of
            the Metric values from the Robust Stat Detector.
        """

        robust_stat_detector_features = {
            "robust_num": np.nan,
            "robust_metric_mean": np.nan,
        }
        try:
            robust = robust_stat_detection.RobustStatDetector(ts)
            robust_cp = robust.detector()
            if extra_args is not None and extra_args.get("robust_num", default_status):
                robust_stat_detector_features["robust_num"] = len(robust_cp)
            if extra_args is not None and extra_args.get("robust_metric_mean", default_status):
                robust_stat_detector_features["robust_metric_mean"] = (
                    0
                    if len(robust_cp) == 0
                    else np.sum([cp[1]._metric for cp in robust_cp]) / len(robust_cp)
                )
            return robust_stat_detector_features
        except Exception:
            return robust_stat_detector_features

    # BOCP Detection Outputs (3)
    @staticmethod
    def get_bocp_detector(
        ts: TimeSeriesData, extra_args: Optional[Dict[str, bool]] = None, default_status: bool = True
    ):
        """
        Run the Kats BOCP Detector on the Time Series, extract features from the outputs of the detection.

        Args:
            ts: The univariate time series array in the form of Kats TimeSeriesData object.
            extra_args: A dictionary containing information for disabling calculation
                of a certain feature. Default value is None, i.e. no feature is disabled.
            default_status: Default status of the switch for calculate the features or not.
                Default value is True.

        Returns:
            (1) Number of changepoints detected by BOCP Detector, (2) Max value of the
            confidience of the changepoints detected, (3) Mean value of the confidience
             of the changepoints detected.
        """

        bocp_detector_features = {
            "bocp_num": np.nan,
            "bocp_conf_max": np.nan,
            "bocp_conf_mean": np.nan,
        }
        try:
            bocp = bocpd.BOCPDetector(ts)
            bocp_cp = bocp.detector(choose_priors=False)
            if extra_args is not None and extra_args.get("bocp_num", default_status):
                bocp_detector_features["bocp_num"] = len(bocp_cp)
            if extra_args is not None and extra_args.get("bocp_conf_max", default_status):
                bocp_detector_features["bocp_conf_max"] = (
                    0
                    if len(bocp_cp) == 0
                    else np.max([cp[0].confidence for cp in bocp_cp])
                )
            if extra_args is not None and extra_args.get("bocp_conf_mean", default_status):
                bocp_detector_features["bocp_conf_mean"] = (
                    0
                    if len(bocp_cp) == 0
                    else np.sum([cp[0].confidence for cp in bocp_cp]) / len(bocp_cp)
                )
            return bocp_detector_features
        except Exception:
            return bocp_detector_features

    # Outlier Detection Outputs (1)
    @staticmethod
    def get_outlier_detector(
        ts: TimeSeriesData,
        decomp: str = "additive",
        iqr_mult: float = 3.0,
        extra_args: Optional[Dict[str, bool]] = None,
        default_status: bool = True,
    ):
        """
        Run the Kats Outlier Detector on the Time Series, extract features from the outputs of the detection.

        Args:
            ts: The univariate time series array in the form of Kats TimeSeriesData object.
            decomp: str; Additive or Multiplicative mode for performing outlier detection using
                OutlierDetector. Default value is 'additive'.
            iqr_mult: float; IQR range for determining outliers through
                OutlierDetector. Default value is 3.0.
            extra_args: A dictionary containing information for disabling calculation
                of a certain feature. Default value is None, i.e. no feature is disabled.
            default_status: Default status of the switch for calculate the features or not.
                Default value is True.

        Returns:
            Number of outliers by the Outlier Detector.
        """

        outlier_detector_features = {"outlier_num": np.nan}
        try:
            odetector = outlier.OutlierDetector(ts, decomp=decomp, iqr_mult=iqr_mult)
            odetector.detector()
            if extra_args is not None and extra_args.get("outlier_num", default_status):
                # pyre-fixme[16]: `OutlierDetector` has no attribute `outliers`.
                outlier_detector_features["outlier_num"] = len(odetector.outliers[0])
            return outlier_detector_features
        except Exception:
            return outlier_detector_features

    # Trend Detection Outputs (3)
    @staticmethod
    def get_trend_detector(
        ts: TimeSeriesData,
        threshold: float = 0.8,
        extra_args: Optional[Dict[str, bool]] = None,
        default_status: bool = True,
    ):
        """
        Run the Kats Trend Detector on the Time Series, extract features from the outputs of the detection.

        Args:
            ts: The univariate time series array in the form of Kats TimeSeriesData object.
            threshold: float; threshold for trend intensity; higher threshold gives trend
                with high intensity (0.8 by default).  If we only want to use the p-value
                to determine changepoints, set threshold = 0.
            extra_args: A dictionary containing information for disabling calculation
                of a certain feature. Default value is None, i.e. no feature is disabled.
            default_status: Default status of the switch for calculate the features or not.
                Default value is True.

        Returns:
            (1) Number of trends detected by the Kats Trend Detector, (2) Number of increasing
            trends, (3) Mean of the abolute values of Taus of the trends detected.
        """

        trend_detector_features = {
            "trend_num": np.nan,
            "trend_num_increasing": np.nan,
            "trend_avg_abs_tau": np.nan,
        }
        try:
            tdetector = trend_mk.MKDetector(data=ts, threshold=threshold)
            tdetected_time_points = tdetector.detector(direction="both")
            if extra_args is not None and extra_args.get("trend_num", default_status):
                trend_detector_features["trend_num"] = len(tdetected_time_points)
            if extra_args is not None and extra_args.get("trend_num_increasing", default_status):
                trend_detector_features["trend_num_increasing"] = len(
                    [
                        p
                        for p in tdetected_time_points
                        if p[1].trend_direction == "decreasing"
                    ]
                )
            if extra_args is not None and extra_args.get("trend_avg_abs_tau", default_status):
                trend_detector_features["trend_avg_abs_tau"] = (
                    0
                    if len(tdetected_time_points) == 0
                    else np.sum([abs(p[1].Tau) for p in tdetected_time_points])
                    / len(tdetected_time_points)
                )
            return trend_detector_features
        except Exception:
            return trend_detector_features

    @staticmethod
    @jit(nopython=True)
    def _ewma(
        arr: np.ndarray,
        span: int,
        min_periods: int
    ):
        """
        Exponentialy weighted moving average specified by a decay ``window``
        to provide better adjustments for small windows via:
            y[t] = (x[t] + (1-a)*x[t-1] + (1-a)^2*x[t-2] + ... + (1-a)^n*x[t-n]) /
                   (1 + (1-a) + (1-a)^2 + ... + (1-a)^n).

        Args:
        arr : np.ndarray; A single dimenisional numpy array.
        span : int; The decay window, or 'span'.
        min_periods: int; Minimum amount of data points we'd like to include in the output.

        Returns:
            A np.ndarray. The exponentially weighted moving average of the array.
        """
        output_array = np.empty(arr.shape[0], dtype=np.float64)
        output_array[:] = np.NaN

        arr = arr[~np.isnan(arr)]
        n = arr.shape[0]
        ewma = np.empty(n, dtype=np.float64)
        alpha = 2 / float(span + 1)
        w = 1
        ewma_old = arr[0]
        ewma[0] = ewma_old
        for i in range(1, n):
            w += (1-alpha)**i
            ewma_old = ewma_old*(1-alpha) + arr[i]
            ewma[i] = ewma_old / w

        output_subset = ewma[(min_periods-1):]
        output_array[-len(output_subset):] = output_subset
        return output_array

    @staticmethod
    @jit(forceobj=True)
    def _get_nowcasting_np(
        x: np.ndarray,
        window: int = 5,
        n_fast: int = 12,
        n_slow: int = 21,
        extra_args: Optional[Dict[str, bool]] = None,
        default_status: bool = True,
    ):
        """
        Internal function for actually performing feature engineering using the same procedure as
        nowcasting feature_extraction under kats/models.

        Args:
            x: The univariate time series array in the form of 1d numpy array.
            window: int; Length of window size for all Nowcasting features. Default value is 5.
            n_fast: int; length of "fast" or short period exponential moving average in the MACD
                algorithm in the nowcasting features. Default value is 12.
            n_slow: int; length of "slow" or long period exponential moving average in the MACD
                algorithm in the nowcasting features. Default value is 21.
            extra_args: A dictionary containing information for disabling calculation
                of a certain feature. Default value is None, i.e. no feature is disabled.
            default_status: Default status of the switch for calculate the features or not.
                Default value is True.

        Returns:
            A list containing extracted nowcast features.
        """

        # initializing the outputs
        nowcasting_features = [np.nan for _ in range(7)]

        # ROC: indicating return comparing to step n back.
        if extra_args is not None and extra_args.get("nowcast_roc", default_status):
            M = x[(window-1):] - x[:-(window-1)]
            N = x[:-(window-1)]
            arr = M / N
            nowcasting_features[0] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).mean()

        # MOM: indicating momentum: difference of current value and n steps back.
        if extra_args is not None and extra_args.get("nowcast_mom", default_status):
            M = x[window:] - x[:-window]
            nowcasting_features[1] = np.nan_to_num(M, nan=0.0, posinf = 0.0, neginf=0.0).mean()

        # MA: indicating moving average in the past n steps.
        if extra_args is not None and extra_args.get("nowcast_ma", default_status):
            ret = np.cumsum(x, dtype=float)
            ret[window:] = ret[window:] - ret[:-window]
            ma = ret[window - 1:] / window
            nowcasting_features[2] = np.nan_to_num(ma, nan=0.0, posinf=0.0, neginf=0.0).mean()

        # LAG: indicating lagged value at the past n steps.
        if extra_args is not None and extra_args.get("nowcast_lag", default_status):
            N = x[:-window]
            nowcasting_features[3] = np.nan_to_num(N, nan=0.0, posinf=0.0, neginf=0.0).mean()

        # MACD: https://www.investopedia.com/terms/m/macd.asp.
        ema_fast = TsFeatures._ewma(x, n_fast, n_slow-1)
        ema_slow = TsFeatures._ewma(x, n_slow, n_slow-1)
        MACD = ema_fast - ema_slow
        if extra_args is not None and extra_args.get("nowcast_macd", default_status):
            nowcasting_features[4] = np.nan_to_num(np.nanmean(MACD), nan=0.0, posinf=0.0, neginf=0.0)

        MACDsign = TsFeatures._ewma(MACD, 9, 8)
        if extra_args is not None and extra_args.get("nowcast_macdsign", default_status):
            nowcasting_features[5] = np.nan_to_num(np.nanmean(MACDsign), nan=0.0, posinf=0.0, neginf=0.0)

        MACDdiff = MACD - MACDsign
        if extra_args is not None and extra_args.get("nowcast_macddiff", default_status):
            nowcasting_features[6] = np.nan_to_num(np.nanmean(MACDdiff), nan=0.0, posinf=0.0, neginf=0.0)

        return nowcasting_features

    # Nowcasting features (7)
    @staticmethod
    def get_nowcasting(
        x: np.ndarray,
        window: int = 5,
        n_fast: int = 12,
        n_slow: int = 21,
        extra_args: Optional[Dict[str, bool]] = None,
        default_status: bool = True,
    ):
        """
        Run the Kats Nowcasting transformer on the Time Series, extract aggregated features from the outputs of the transformation.

        Args:
            x: The univariate time series array in the form of 1d numpy array.
            window: int; Length of window size for all Nowcasting features. Default value is 5.
            n_fast: int; length of "fast" or short period exponential moving average in the MACD
                algorithm in the nowcasting features. Default value is 12.
            n_slow: int; length of "slow" or long period exponential moving average in the MACD
                algorithm in the nowcasting features. Default value is 21.
            extra_args: A dictionary containing information for disabling calculation
                of a certain feature. Default value is None, i.e. no feature is disabled.
            default_status: Default status of the switch for calculate the features or not.
                Default value is True.

        Returns:
            Mean values of the Kats Nowcasting algorithm time series outputs using the parameters
            (window, n_fast, n_slow) indicated above. These outputs inclue : (1) Mean of Rate of
            Change (ROC) time series, (2) Mean of Moving Average (MA) time series,(3) Mean of
            Momentum (MOM) time series, (4) Mean of LAG time series, (5) Means of MACD, MACDsign,
            and MACDdiff from Kats Nowcasting.
        """
        nowcasting_features = {}
        features = [
            "nowcast_roc",
            "nowcast_mom",
            "nowcast_ma",
            "nowcast_lag",
            "nowcast_macd",
            "nowcast_macdsign",
            "nowcast_macddiff",
        ]
        for feature in features:
            if extra_args is not None and extra_args.get(feature, default_status):
                nowcasting_features[feature] = np.nan

        try:
            _features = TsFeatures._get_nowcasting_np(x, window, n_fast, n_slow, extra_args, default_status)
            for idx, feature in enumerate(features):
                if extra_args is not None and extra_args.get(feature, default_status):
                    nowcasting_features[feature] = _features[idx]
            return nowcasting_features
        except Exception:
            return nowcasting_features

    # seasonality features (4)
    @staticmethod
    def get_seasonalities(
        ts: TimeSeriesData, extra_args: Optional[Dict[str, bool]] = None, default_status: bool = True
    ):
        """
        Run the Kats seaonality detectors to get the estimated seasonal period, then extract trend,
        seasonality and residual magnitudes.

        Args:
            ts: The univariate time series array in the form of Kats TimeSeriesData object.
            extra_args: A dictionary containing information for disabling calculation
                of a certain feature. Default value is None, i.e. no feature is disabled.
            default_status: Default status of the switch for calculate the features or not.
                Default value is True.

        Returns:
            Returns the detected seasonality period.
            Slope acquired via fitting simple linear regression model on the trend component
            as trend magnitude.
            Difference between the 95 percentile and 5 percentile of the seasonal component
            as the seasonality magnitude.
            Standard deviation of the residual component.
        """

        seasonality_features = {
            "seasonal_period": np.nan,
            "trend_mag": np.nan,
            "seasonality_mag": np.nan,
            "residual_std": np.nan,
        }

        try:
            # detrending for period estimation
            detrended = TimeSeriesData(
                pd.DataFrame(
                    {
                        "time": len(ts.value.values) - 1,
                        "value": ts.value.values[1:] - ts.value.values[:-1],
                    }
                )
            )
            detected = seasonality.FFTDetector(detrended).detector()

            if detected["seasonality_presence"]:
                _period = int(np.min(detected["seasonalities"]))
            elif not detected["seasonality_presence"]:
                _period = 7
            res = STL(ts.value.values, period=_period).fit()

            if extra_args is not None and extra_args.get("seasonal_period", default_status):
                seasonality_features["seasonal_period"] = _period

            # getting seasonality magnitude
            if extra_args is not None and extra_args.get("seasonality_mag", default_status):
                seasonality_features["seasonality_mag"] = np.round(
                    np.quantile(res.seasonal, 0.95) - np.quantile(res.seasonal, 0.05)
                )

            # fitting linear regression for trend magnitude
            if extra_args is not None and extra_args.get("trend_mag", default_status):
                exog = res.trend
                _series = exog - exog[0]
                mod = sm.OLS(_series, np.arange(len(_series)))
                _res = mod.fit()
                # trend magnitude
                seasonality_features["trend_mag"] = _res.params[0]

            # residual standard deviation
            if extra_args is not None and extra_args.get("residual_std", default_status):
                seasonality_features["residual_std"] = np.std(res.resid)

            return seasonality_features
        except Exception:
            return seasonality_features
