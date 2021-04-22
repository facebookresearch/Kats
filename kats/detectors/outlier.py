#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import datetime as dt
import logging
import sys
import traceback
from enum import Enum
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kats.consts import Params, TimeSeriesData, TimeSeriesIterator
from kats.detectors.detector import Detector
from kats.models.bayesian_var import BayesianVAR
from kats.models.var import VARModel
from scipy import stats
from scipy.spatial import distance
from statsmodels.tsa.seasonal import seasonal_decompose


class OutlierDetector(Detector):
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
        original.index = pd.to_datetime(original.index)

        if pd.infer_freq(original.index) is None:
            original = original.asfreq("D")
            logging.info("Setting frequency to Daily since it cannot be inferred")

        original = original.interpolate(
            method="polynomial", limit_direction="both", order=3
        )

        # This is a hack since polynomial interpolation is not working here
        if sum((np.isnan(x) for x in original["y"])):
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
        self.outliers_index = list(outliers.index)
        return list(outliers.index)

    def detector(self):
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
    VAR = 0
    BAYESIAN_VAR = 1


class MultivariateAnomalyDetector(Detector):
    DETECTOR_CONVERSION = {
        MultivariateAnomalyDetectorType.VAR: VARModel,
        MultivariateAnomalyDetectorType.BAYESIAN_VAR: BayesianVAR,
    }

    def __init__(
        self,
        data: TimeSeriesData,
        params: Params,
        training_days: float,
        model_type: MultivariateAnomalyDetectorType = MultivariateAnomalyDetectorType.VAR,
    ) -> None:
        """
        Arg:
            data: Input metrics TimeSeriesData
            params: Params class
            training_days: num of days of data to use for initial training.
                        If less than a day, specify as fraction of a day
            model_type: The type of multivariate anomaly detector (currently 'BAYESIAN_VAR' and 'VAR' options available)
        """
        super().__init__(data)
        df = data.to_dataframe().set_index("time")
        self.df = df

        params.validate_params()
        self.params = params

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
        self.detector_model = MultivariateAnomalyDetector.DETECTOR_CONVERSION[
            model_type
        ]

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        z_score_threshold = 3
        zscore_df = stats.zscore(df)
        non_outlier_flag = zscore_df < z_score_threshold
        df_clean = df.where(non_outlier_flag, np.nan)
        df_clean = df_clean.interpolate(
            method="linear", order=2, limit_direction="both"
        )
        return df_clean

    def _is_pos_def(self, mat: np.ndarray) -> bool:
        return np.all(np.linalg.eigvals(mat) > 0)

    def _generate_forecast(self, t: Any) -> pd.DataFrame:
        """
        Fit VAR model and generate 1 step ahead forecasts (t+1)
        """
        logging.info(f"Generating forecasts at {t}")
        look_back = dt.timedelta(days=self.training_days)
        train = self.df.loc[t - look_back : t, :]
        train_clean = self._clean_data(train)

        # fit VAR
        model = self.detector_model(
            TimeSeriesData(train_clean.reset_index()), self.params
        )
        model.fit()
        lag_order = model.k_ar
        logging.info(f"Fitted VAR model of order {lag_order}")
        self.resid = model.resid
        self.sigma_u = model.sigma_u
        if ~(self._is_pos_def(self.sigma_u)):
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

        # individual anomaly scores
        cov = self.sigma_u
        residual_score = {}
        rt = pred_df["est"] - pred_df["actual"]
        for col in cov.columns:
            residual_mean = self.resid[col].mean()
            residual_var = self.resid[col].var()
            residual_score[col] = np.abs((rt[col] - residual_mean)) / np.sqrt(residual_var)

        # overall anomaly score
        cov_inv = np.linalg.inv(self.sigma_u.values)
        overall_anomaly_score = distance.mahalanobis(
            rt.values, self.resid.mean().values, cov_inv
        )
        residual_score["overall_anomaly_score"] = overall_anomaly_score
        # calculate p-values
        dof = len(self.df.columns)
        residual_score['p_value']= stats.chi2.sf(overall_anomaly_score, df=dof)

        return residual_score

    def detector(self) -> pd.DataFrame:
        """
        Returns:
            DataFrame with colums corresponding to individual anomaly scores
            of each metric and the overall anomaly score
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
        Arg:
            alpha: significance level to consider the timeperiod anomalous
            Use .plot() to help choose a good threshold

        Returns:
            List of time instants when the system of metrics show anomalous behavior
        """

        flag = self.anomaly_score_df.p_value < alpha
        anomaly_ts = self.anomaly_score_df[flag].index

        return list(anomaly_ts)

    def get_anomalous_metrics(self, t: Any, top_k: int = None) -> pd.DataFrame:
        """
        Find top k metrics with most anomalous behavior at time t

        Arg:
            t: time instant of interest (same type as TimeSeriesData.time)
            top_k: Top few metrics to return. If None, returns all

        Returns:
            DataFrame with metrics and their corresponding anomaly score
        """

        residual_scores = self.anomaly_score_df.drop(columns="overall_anomaly_score")
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
