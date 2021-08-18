# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Module with generic outlier detection models. Supports a univariate algorithm that
treates each metric separately to identify outliers and a multivariate detection
algorithm that determines outliers based on joint distribution of metrics
"""

import datetime as dt
import logging
import sys
import traceback
from datetime import datetime
from enum import Enum
from importlib import import_module
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kats.consts import Params, TimeSeriesData, TimeSeriesIterator
from kats.detectors.detector import Detector
from scipy import stats
from scipy.spatial import distance
from statsmodels.tsa.seasonal import seasonal_decompose


class OutlierDetector(Detector):
    """
    This univariate outlier detection algorithm mimics the outlier
    detection algorithm in R

    Attributes:
        data: TimeSeriesData object with the time series
        decomp : 'additive' or 'multiplicative' decomposition
        iqr_mult : iqr_mult * inter quartile range is used to classify outliers
    """

    outliers_index: Optional[List] = None
    outliers: Optional[List[List]] = None

    def __init__(
        self, data: TimeSeriesData, decomp: str = "additive", iqr_mult: float = 3.0
    ) -> None:
        super().__init__(data)
        if decomp in ["additive", "multiplicative"]:
            self.decomp = decomp
        else:
            logging.info("Invalid decomposition setting specified")
            logging.info("Defaulting to Additive Decomposition")
            self.decomp = "additive"
        self.iqr_mult = iqr_mult

    def __clean_ts__(self, original: pd.DataFrame) -> List:
        """
        Performs detection for a single metric. First decomposes the time series
        and detects outliers when the values in residual time series are beyond the
        specified multiplier times the inter quartile range
        Args:
            original: original time series as DataFrame
        Returns: List of detected outlier timepoints in each metric
        """

        original.index = pd.to_datetime(original.index)

        if pd.infer_freq(original.index) is None:
            # pyre-fixme[9]: original has type `DataFrame`; used as
            #  `Union[pd.core.frame.DataFrame, pd.core.series.Series]`.
            original = original.asfreq("D")
            logging.info("Setting frequency to Daily since it cannot be inferred")

        # pyre-fixme[9]: original has type `DataFrame`; used as `Union[None,
        #  pd.core.frame.DataFrame, pd.core.series.Series]`.
        original = original.interpolate(
            method="polynomial", limit_direction="both", order=3
        )

        # This is a hack since polynomial interpolation is not working here
        if sum((np.isnan(x) for x in original["y"])):
            # pyre-fixme[9]: original has type `DataFrame`; used as `Union[None,
            #  pd.core.frame.DataFrame, pd.core.series.Series]`.
            original = original.interpolate(method="linear", limit_direction="both")

        # Once our own decomposition is ready, we can directly use it here
        result = seasonal_decompose(original, model=self.decomp)
        rem = result.resid
        detrend = original["y"] - result.trend
        strength = float(1 - np.nanvar(rem) / np.nanvar(detrend))
        if strength >= 0.6:
            original["y"] = original["y"] - result.seasonal
        # using IQR as threshold
        resid = original["y"] - result.trend
        resid_q = np.nanpercentile(resid, [25, 75])
        iqr = resid_q[1] - resid_q[0]
        limits = resid_q + (self.iqr_mult * iqr * np.array([-1, 1]))

        outliers = resid[(resid >= limits[1]) | (resid <= limits[0])]
        self.outliers_index = outliers_index = list(outliers.index)
        return outliers_index

    def detector(self):
        """
        Detects outliers and stores in self.outliers
        """

        self.iter = TimeSeriesIterator(self.data)
        self.outliers = []
        logging.Logger("Detecting Outliers")
        for ts in self.iter:
            try:
                outlier = self.__clean_ts__(ts)
                self.outliers.append(outlier)
            except BaseException:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
                logging.error("".join("!! " + line for line in lines))
                logging.error("Outlier Detection Failed")
                self.outliers.append([])


class MultivariateAnomalyDetectorType(Enum):
    VAR = "var.VARModel"
    BAYESIAN_VAR = "bayesian_var.BayesianVAR"


class MultivariateAnomalyDetector(Detector):
    """
    Detector class for Multivariate Outlier Detection.
    Provides utilities to calculate anomaly scores, get anomalous
    datapoints and anomalous metrics at those points.

    Attributes:
        data: Input metrics TimeSeriesData
        params: Parameter class for multivariate VAR/ BVAR model
        training_days: num of days of data to use for initial training.
                    If less than a day, specify as fraction of a day
        model_type: The type of multivariate anomaly detector (currently 'BAYESIAN_VAR' and 'VAR' options available)
    """

    resid: Optional[pd.DataFrame] = None
    sigma_u: Optional[pd.DataFrame] = None
    anomaly_score_df: Optional[pd.DataFrame] = None

    def __init__(
        self,
        data: TimeSeriesData,
        params: Params,
        training_days: float,
        model_type: MultivariateAnomalyDetectorType = MultivariateAnomalyDetectorType.VAR,
    ) -> None:
        super().__init__(data)
        df = data.to_dataframe().set_index("time")
        self.df = df

        params.validate_params()
        self.params = params

        # pyre-fixme[16]: `Optional` has no attribute `diff`.
        time_diff = data.time.sort_values().diff().dropna()
        if len(time_diff.unique()) == 1:  # check constant frequenccy
            freq = time_diff.unique()[0].astype("int")
            self.granularity_days = freq / (24 * 3600 * (10 ** 9))
        else:
            raise RuntimeError(
                "Frequency of metrics is not constant."
                "Please check for missing or duplicate values"
            )

        self.training_days = training_days
        self.detector_model = model_type

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove outliers in training data based on zscore

        Args:
            df: input dataframe

        Returns: Clean dataframe
        """

        z_score_threshold = 3
        # pyre-fixme[16]: Module `stats` has no attribute `zscore`.
        zscore_df = stats.zscore(df)
        non_outlier_flag = zscore_df < z_score_threshold
        df_clean = df.where(non_outlier_flag, np.nan)
        df_clean = df_clean.interpolate(
            method="linear", order=2, limit_direction="both"
        )
        return df_clean

    def _is_pos_def(self, mat: np.ndarray) -> bool:
        """
        Check if matrix is positive definite.

        Args:
            mat: numpy matrix

        Returns:
            True if mat is positive definite
        """
        return np.all(np.linalg.eigvals(mat) > 0)

    def _create_model(self, data: TimeSeriesData, params: Params) -> Any:
        model_name = f"kats.models.{self.detector_model.value}"
        module_name, model_name = model_name.rsplit(".", 1)
        return getattr(import_module(module_name), model_name)(data, params)

    def _generate_forecast(self, t: datetime) -> pd.DataFrame:
        """
        Fit VAR model and generate 1 step ahead forecasts (t+1)

        Args:
            t: time until which to use for training the VAR model

        Returns:
            DataFrame with predicted expected value for each metric value at (t+1)
        """
        logging.info(f"Generating forecasts at {t}")
        look_back = dt.timedelta(days=self.training_days)
        train = self.df.loc[t - look_back : t, :]
        train_clean = self._clean_data(train)

        # fit VAR
        model = self._create_model(
            TimeSeriesData(train_clean.reset_index()), self.params
        )
        model.fit()
        lag_order = model.k_ar
        logging.info(f"Fitted VAR model of order {lag_order}")
        self.resid = model.resid
        self.sigma_u = sigma_u = model.sigma_u
        if ~(self._is_pos_def(sigma_u)):
            msg = f"Fitted Covariance matrix at time {t} is not positive definite"
            logging.error(msg)
            raise RuntimeError(msg)

        # predict
        pred = model.predict(steps=1)
        forecast = [[k, float(pred[k].value["fcst"])] for k, v in pred.items()]
        pred_df = pd.DataFrame(columns=["index", "est"], data=forecast).set_index(
            "index"
        )
        test = self.df.loc[t + dt.timedelta(days=self.granularity_days), :]
        pred_df["actual"] = test

        return pred_df

    def _calc_anomaly_scores(self, pred_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate overall anomay score at time t based on multivariate Mahalonabis distance
        and individual anomaly scores as zscores

        Args:
            pred_df: Dataframe with forecasted values

        Returns:
            Dictionary with overall and individual anomaly scores along with p-value
        """

        # individual anomaly scores
        cov = self.sigma_u
        resid = self.resid
        assert cov is not None and resid is not None
        residual_score = {}
        rt = pred_df["est"] - pred_df["actual"]
        for col in cov.columns:
            residual_mean = resid[col].mean()
            residual_var = resid[col].var()
            residual_score[col] = np.abs((rt[col] - residual_mean)) / np.sqrt(
                residual_var
            )

        # overall anomaly score
        cov_inv = np.linalg.inv(cov.values)
        overall_anomaly_score = distance.mahalanobis(
            rt.values, resid.mean().values, cov_inv
        )
        residual_score["overall_anomaly_score"] = overall_anomaly_score
        # calculate p-values
        dof = len(self.df.columns)
        # pyre-ignore[16]: Module `stats` has no attribute `chi2`.
        residual_score["p_value"] = stats.chi2.sf(overall_anomaly_score, df=dof)

        return residual_score

    # pyre-fixme[14]: `detector` overrides method defined in `Detector` inconsistently.
    def detector(self) -> pd.DataFrame:
        """
        Fit the detection model and return the results

        Returns:
            DataFrame with colums corresponding to individual anomaly scores
            of each metric and the overall anomaly score for the whole timeseries
        """
        anomaly_score_df = pd.DataFrame()
        look_back = dt.timedelta(days=self.training_days)
        fcstTime = self.df.index.min() + look_back
        while fcstTime < self.df.index.max():
            # forecast for fcstTime+ 1
            pred_df = self._generate_forecast(fcstTime)
            # calculate anomaly scores
            anomaly_scores_t = self._calc_anomaly_scores(pred_df)
            # process next observation
            fcstTime += dt.timedelta(days=self.granularity_days)
            anomaly_scores_t = pd.DataFrame(anomaly_scores_t, index=[fcstTime])
            anomaly_score_df = anomaly_score_df.append(anomaly_scores_t)

        self.anomaly_score_df = anomaly_score_df
        return anomaly_score_df

    def get_anomaly_timepoints(self, alpha: float) -> List:
        """
        Helper function to get anomaly timepoints based on the significance level

        Args:
            alpha: significance level to consider the timeperiod anomalous
            Use .plot() to help choose a good threshold

        Returns:
            List of time instants when the system of metrics show anomalous behavior
        """
        anomaly_score_df = self.anomaly_score_df
        if anomaly_score_df is None:
            raise ValueError(
                "detector() must be called before get_anomaly_timepoints()"
            )

        flag = anomaly_score_df.p_value < alpha
        anomaly_ts = anomaly_score_df[flag].index

        return list(anomaly_ts)

    def get_anomalous_metrics(
        self, t: datetime, top_k: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Find top k metrics with most anomalous behavior at time t

        Args:
            t: time instant of interest (same type as TimeSeriesData.time)
            top_k: Top few metrics to return. If None, returns all

        Returns:
            DataFrame with metrics and their corresponding anomaly score
        """
        anomaly_score_df = self.anomaly_score_df
        if anomaly_score_df is None:
            raise ValueError("detector() must be called before get_anomalous_metrics()")

        residual_scores = anomaly_score_df.drop(columns="overall_anomaly_score")
        residual_scores_t = (
            residual_scores.loc[t, :].sort_values(ascending=False).reset_index()
        )
        residual_scores_t.columns = ["metric", "anomaly_score"]

        return residual_scores_t[:top_k]

    def plot(self):
        """
        Plot overall anomaly score of system of metrics at each instant.
        Useful for threshold selection
        """
        f, ax = plt.subplots(2, 1, sharex=True, figsize=(15, 8 * 2))
        a = pd.merge(
            self.df,
            self.anomaly_score_df["overall_anomaly_score"],
            left_index=True,
            right_index=True,
            how="right",
        )
        a.drop(columns=["overall_anomaly_score"]).plot(ax=ax[0])
        ax[0].set_title("Input time series metrics")

        a["overall_anomaly_score"].plot(legend=False, ax=ax[1])
        ax[1].set_title("Overall Anomaly Score")
