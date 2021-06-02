#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""This module contains the class TemporalHierarchicalModel class.
"""

import logging
from math import gcd
from typing import List, Dict, Optional, Callable

import numpy as np
import pandas as pd
from kats.consts import TimeSeriesData
from kats.models import (
    arima,
    holtwinters,
    linear_model,
    prophet,
    quadratic_model,
    sarima,
    theta,
)
from kats.models.reconciliation.base_models import (
    BaseTHModel,
    calc_mape,
    calc_mae,
    GetAggregateTS,
)
from sklearn.covariance import MinCovDet


BASE_MODELS = {
    "arima": arima.ARIMAModel,
    "holtwinters": holtwinters.HoltWintersModel,
    "sarima": sarima.SARIMAModel,
    "prophet": prophet.ProphetModel,
    "linear": linear_model.LinearModel,
    "quadratic": quadratic_model.QuadraticModel,
    "theta": theta.ThetaModel,
}


class TemporalHierarchicalModel:
    """Temporal hierarchical model class.

    This framework combines the base models of different temporal aggregation levels to generate reconciled forecasts.
    This class provides fit, get_S, get_W, predict and median_validation.

    Attributes:
        data: A TimeSeriesData object storing the time series data for level 1 (i.e., the most disaggregate level).
        baseModels: A list BaseTHModel objects representing the base models for different levels.
    """

    def __init__(self, data: TimeSeriesData, baseModels: List[BaseTHModel]) -> None:

        if not data.is_univariate():
            msg = f"Only support univariate time series, but get {type(data.value)}."
            logging.error(msg)
            raise ValueError(msg)

        self.data = data
        for basemodel in baseModels:
            if not isinstance(basemodel, BaseTHModel):
                msg = f"Base model should be a BaseTHModel object but receive {type(basemodel)}."
                logging.info(msg)
                raise ValueError(msg)

        levels = [bm.level for bm in baseModels]

        if 1 not in levels:
            msg = "Model of level 1 is missing."
            logging.error(msg)
            raise ValueError(msg)

        if len(levels) != len(set(levels)):
            msg = "One level cannot receive multiple models."
            logging.error(msg)
            raise ValueError(msg)

        self.levels = sorted(levels, reverse=True)

        m = self._get_m(levels)
        self.m = m
        self.freq = {k: int(m / k) for k in self.levels}
        self.baseModels = baseModels
        self.info_fcsts = {}
        self.info_residuals = {}

    def _get_m(self, ks: List[int]) -> int:
        """Calculate m.
            m is the minimum common multiple of all levels.
        Args:
            ks: the list of integers representing all the levels.

        Returns:
            An integer representing the minimum common multiple.
        """

        base = 1
        for c in ks:
            base = base * c // gcd(base, c)
        return base

    def fit(self) -> None:
        """Fit all base models (if base model only has residuals and forecasts, store the information.)

        Args:
            None.

        Returns:
            None.
        """

        levels = self.levels
        TSs = GetAggregateTS(self.data).aggregate(levels)
        models = {}
        residuals = {}
        fcsts = {}
        for bm in self.baseModels:
            if bm.model_name is None:  # only residuals and fcsts are provided
                models[bm.level] = None
                residuals[bm.level] = bm.residuals
                fcsts[bm.level] = bm.fcsts
            else:
                # pyre-fixme[6]: Expected `str` for 1st param but got `Optional[str]`.
                m = BASE_MODELS[bm.model_name](
                    # pyre-fixme[6]: Expected `ARIMAParams` for 2nd param but got
                    #  `Optional[object]`.
                    # pyre-fixme[6]: Expected `HoltWintersParams` for 2nd param but
                    #  got `Optional[object]`.
                    # pyre-fixme[6]: Expected `LinearModelParams` for 2nd param but
                    #  got `Optional[object]`.
                    # pyre-fixme[6]: Expected `ProphetParams` for 2nd param but got
                    #  `Optional[object]`.
                    # pyre-fixme[6]: Expected `QuadraticModelParams` for 2nd param
                    #  but got `Optional[object]`.
                    # pyre-fixme[6]: Expected `SARIMAParams` for 2nd param but got
                    #  `Optional[object]`.
                    # pyre-fixme[6]: Expected `ThetaParams` for 2nd param but got
                    #  `Optional[object]`.
                    data=TSs[bm.level], params=bm.model_params
                )
                m.fit()
                models[bm.level] = m
        # pyre-fixme[16]: `TemporalHierarchicalModel` has no attribute `models`.
        self.models = models
        self.info_fcsts = fcsts
        self.info_residuals = residuals

    def get_S(self) -> np.ndarray:
        """Calculate S matrix.

        Args:
            None.

        Returns:
            A np.array representing the S matrix
        """

        ans = []
        levels = self.levels
        m = self.m
        for k in levels:
            for i in range(self.freq[k]):
                tem = np.zeros(m)
                tem[(i * k) : (i * k + k)] = 1.0
                ans.append(tem)
        return np.row_stack(ans)

    def _aggregate_data(self, data: np.ndarray, k: int) -> np.ndarray:
        """Aggregate data according to level k."""

        if k == 1:
            return data
        n = len(data)
        h = n // k
        return (data[: int(h * k)]).reshape(-1, k).sum(axis=1)

    def _get_residuals(self, model: Callable) -> np.ndarray:
        """Calculate residuals of each base model.

        Args:
            model: a callable model object representing the trained base model.

        Return:
            A np.ndarray of residuals.
        """
        try:
            # pyre-fixme[16]: Anonymous callable has no attribute `model`.
            resid = model.model.resid.values
            return resid
        except Exception:
            # pyre-fixme[16]: Anonymous callable has no attribute `predict`.
            fcst = model.predict(steps=1, freq="D", include_history=True)
            # pyre-fixme[16]: Anonymous callable has no attribute `data`.
            merge = fcst.merge(model.data.to_dataframe(), on="time")
            for col in merge.columns:
                if col != "time" and ("fcst" not in col):
                    lab = col
                    break
            return merge[lab].values - merge["fcst"].values

    def _get_all_residuals(self) -> Dict[int, np.ndarray]:
        """
        Calculate residuals for all base models.

        :Returns: Dict[int, np.ndarray]
            Dictionary for residuals, whose key is level and value is residual array.
        """
        # if residuals have not been calculated yet
        if not hasattr(self, "residuals"):
            levels = self.levels
            # pyre-fixme[16]: `TemporalHierarchicalModel` has no attribute `models`.
            models = self.models
            residuals = {}
            for k in levels:
                if models[k] is not None:
                    try:
                        vals = self._get_residuals(models[k])
                    except Exception as e:
                        msg = f"Fail to get residuals for level {k} with error message {e}."
                        logging.error(msg)
                        raise ValueError(msg)

                    residuals[k] = vals
                else:
                    residuals[k] = self.info_residuals[k]
            # pyre-fixme[16]: `TemporalHierarchicalModel` has no attribute `residuals`.
            self.residuals = residuals
        return self.residuals

    def _get_residual_matrix(self) -> np.ndarray:
        """
        Reshape residuals into matrix format.

        :Returns: np.ndarray
            Residual matrix
        """
        if not hasattr(self, "res_matrix"):
            residuals = self._get_all_residuals()
            ks = self.levels
            freq = self.freq
            h = np.min([len(residuals[k]) // freq[k] for k in ks])
            res_matrix = []
            for k in ks:
                n = h * freq[k]
                res_matrix.append(residuals[k][-n:].reshape(h, -1).T)
            res_matrix = np.row_stack(res_matrix)
            # pyre-fixme[16]: `TemporalHierarchicalModel` has no attribute `res_matrix`.
            self.res_matrix = res_matrix
        return self.res_matrix

    def get_W(self, method: str = "struc", eps: float = 1e-5) -> np.ndarray:
        """
        Calculate W matrix.

        :Parameters:
        method: str = "struc"
            Reconciliation method for temporal hierarchical model. Valid methods include 'struc', 'svar', 'hvar',
           'mint_sample', and 'mint_shrink'.
        eps: float = 1e-5
            Epsilons added to W for numerical stability.

        :Returns: np.ndarray
            W matrix. (If W is a diagnoal matrix, only returns its diagnoal elements).
        """
        levels = self.levels
        freq = self.freq
        if method == "struc":
            ans = []
            for k in levels:
                ans.extend([k] * freq[k])
            return np.array(ans)

        elif method == "svar":
            residuals = self._get_all_residuals()
            ans = []
            for k in levels:
                ans.extend([np.nanmean(np.square(residuals[k]))] * freq[k])
            return np.array(ans) + eps

        elif method == "hvar":
            res_matrix = self._get_residual_matrix()
            return np.nanvar(res_matrix, axis=1) + eps

        elif method == "mint_shrink":
            cov = np.cov(self._get_residual_matrix())
            # get correlation matrix
            sqrt = np.sqrt(np.diag(cov))
            cor = (
                (cov / sqrt).T
            ) / sqrt  # due to symmetry, no need to transpose the matrix again.
            mask = ~np.eye(cor.shape[0], dtype=bool)
            cor = cor[mask]
            lam = np.var(cor) / np.sum(cor ** 2)
            lam = np.max([0, lam])
            cov = np.diag(np.diag(cov)) * lam + (1.0 - lam) * cov
            cov += np.eye(len(cov)) * eps
            return cov

        elif method == "mint_sample":
            cov = np.cov(self._get_residual_matrix())
            cov += np.eye(len(cov)) * eps
            return cov

        else:
            msg = f"{method} is invalid for get_W() method."
            logging.error(msg)
            raise ValueError(msg)

    def _predict_origin(self, steps: int, method="struc") -> Dict[int, np.ndarray]:
        """
        Generate original forecasts from each base model (without time index).

        :Parameters:
        steps: int
            Number of forecasts for level 1.
        methd: str = 'struc'
            Reconciliation method.

        :Returns: Dict[int, np.ndarray]
            Dictionary of forecasts of each level, whose key is level and value is forecast array.
        """
        m = self.m
        levels = self.levels
        freq = self.freq
        h = int(np.ceil(steps / m))
        hf = steps // m
        orig_fcst = {}
        # generate forecasts for each level
        for k in levels:
            num = int(freq[k] * h)
            # pyre-fixme[16]: `TemporalHierarchicalModel` has no attribute `models`.
            if self.models[k] is not None:
                orig_fcst[k] = (
                    self.models[k].predict(steps=num, freq="D")["fcst"].values
                )
            else:
                fcst_num = len(self.info_fcsts[k])
                if fcst_num < num:
                    if fcst_num >= hf * freq[k]:
                        # since the final output only needs hf*freq[k] forecasts for level k, we pad the forecast array to desirable length.
                        # (note that the padding values would be ignored in the final output.)
                        orig_fcst[k] = np.concatenate(
                            [
                                self.info_fcsts[k],
                                [self.info_fcsts[k][-1]] * (num - fcst_num),
                            ]
                        )

                    elif method == "bu" and k != 1:
                        # for 'bu' only level 1 is needed.
                        orig_fcst[k] = self.info_fcsts[k]
                    else:
                        msg = f"{hf*freq[k]} steps of forecasts for level {k} are needed, but only receive {fcst_num} steps (and forecast model is None)."
                        logging.error(msg)
                        raise ValueError(msg)
                else:
                    orig_fcst[k] = self.info_fcsts[k][:num]
        return orig_fcst

    def _predict(
        self,
        steps: int,
        method="struc",
        origin_fcst: bool = False,
        fcst_levels: Optional[List[int]] = None,
    ) -> Dict[str, Dict[int, np.ndarray]]:
        """
        Generate forecasts for each level (without time index).

        :Parameters:
        steps: int
            Number of forecasts for level 1.
        methd: str = 'struc'
            Reconciliation method.
        origin_fcst: bool = False
            Whether to return the forecasts of base models.
        fcst_levels: Optional[List[int]] = None
            Levels that one wants to generate forecasts for.
            If None, then all forecasts for all levels of the base models are generated.

        :Returns: Dict[str, Dict[int, np.ndarray]]
        Dictionary of forecasts, whose key is level and value is forecast array.
        """
        if not hasattr(self, "models"):
            msg = "Please fit base models via .fit() first."
            logging.info(msg)
            raise ValueError(msg)

        m = self.m
        levels = self.levels
        freq = self.freq
        h = int(np.ceil(steps / m))
        if fcst_levels is None:
            fcst_levels = list(levels)
        fcst = {}
        orig_fcst = self._predict_origin(steps, method)

        if method in ["bu", "median"]:
            if method == "bu":
                # bottom_up method
                yhat = orig_fcst[1]
            else:
                # median method
                tem = []
                for k in levels:
                    tem.append(np.repeat(orig_fcst[k] / k, k))
                tem = np.row_stack(tem)
                yhat = np.median(tem, axis=0)

        elif method in {"struc", "svar", "hvar", "mint_shrink", "mint_sample"}:
            # transform fcsts into matrix
            yh = []
            for k in levels:
                yh.append(orig_fcst[k].reshape(h, -1).T)
            yh = np.row_stack(yh)
            S = self.get_S()
            W = self.get_W(method)
            # when W is a vector, i.e., a simpler represent for a diagnoal matrix
            if len(W.shape) == 1:
                T = (S.T) / W
            else:
                T = np.linalg.solve(W, S).T
            yhat = np.dot(S, np.linalg.solve(T.dot(S), T)).dot(yh)
            # extract forecasts for level 1
            yhat = (yhat[(-freq[1]) :, :].T).flatten()[:steps]
        else:
            msg = f"Reconciliation method {method} is invalid."
            logging.info(msg)
            raise ValueError(msg)
        # aggregate fcsts
        for k in fcst_levels:
            fcst[k] = self._aggregate_data(yhat, k)[: (steps // k)]
        ans = {"fcst": fcst}
        if origin_fcst:
            for elm in orig_fcst:
                orig_fcst[elm] = orig_fcst[elm][: (steps // elm)]
            ans["origin_fcst"] = orig_fcst
        return ans

    def predict(
        self,
        steps: int,
        method="struc",
        freq: Optional[str] = None,
        origin_fcst: bool = False,
        fcst_levels: Optional[List[int]] = None,
        last_timestamp: Optional[pd.Timestamp] = None,
    ) -> Dict[str, Dict[int, pd.DataFrame]]:
        """Generate reconciled forecasts (with time index).

        Args:
            steps: An integer representing the number of forecasts needed for level 1.
            methd: Optional; A string representing the name of the reconciliation method. Can be 'bu' (bottom-up), 'median', 'struc' (structure-variance), 'svar', 'hvar', 'mint_shrink' or 'mint_sample'.
                   Default is 'struc'.
            freq: Optional; A string representing the frequency of the time series at level 1. If None, then we infer the frequency via ts.infer_freq_robust(). Default is None.
            origin_fcst: Optional; A boolean to specify whether or not to return the forecasts of base models. Default is False.
            fcst_levels: Optional; A list of integers representing the levels to generate forecasts for. Default is None, which generates forecasts for all the levels of the base models.

        Returns:
            A dictionary of forecasts, whose key is the level and the corresponding value is a np.array storing the forecasts.
        """

        if freq is None:
            freq = self.data.infer_freq_robust()
        last_timestamp = self.data.time.max()
        fcsts = self._predict(
            steps, method=method, origin_fcst=origin_fcst, fcst_levels=fcst_levels
        )
        ans = {}
        for elm in fcsts:
            tmp = {}
            for k in fcsts[elm]:
                fcst_num = len(fcsts[elm][k])
                time = pd.date_range(
                    last_timestamp + freq * k,
                    last_timestamp + freq * k * fcst_num,
                    periods=fcst_num,
                )
                tmp[k] = pd.DataFrame({"time": time, "fcst": fcsts[elm][k]})
            ans[elm] = tmp
        return ans

    def median_validation(
        self, steps, dist_metric: str = "mae", threshold: float = 5.0
    ) -> List[int]:
        """Filtering out bad fcsts based on median forecasts.

        This function detects the levels whose forecasts are greatly deviate from median forecasts, which is a strong indication of bad forecasts.

        Args:
            steps: An integer representing the number of forecasts needed for level 1 for validation.
            dist_metric: Optional; A string representing the distance metric used to measure the distance between the base forecasts and the median forecasts. Default is 'mae'.
            threshold: A float representing the threshold for deviance. The forecast whose distance from the median forecast is greater than threshold*std is taken as bad forecasts. Default is 3.

        Returns:
            A list of integers representing the levels whose forecasts are bad.
        """

        diffs = {}
        ks = self.levels
        if dist_metric == "mae":
            func = calc_mae
        elif dist_metric == "mape":
            func = calc_mape
        else:
            raise ValueError(f"Invalid dist_metric {dist_metric}")
        median_fcst = self._predict(steps, method="median", origin_fcst=True)
        for k in ks:
            diffs[k] = func(median_fcst["fcst"][k], median_fcst["origin_fcst"][k])
            if dist_metric == "mae":
                diffs[k] /= k
        vals = np.array(list(diffs.values()))
        try:
            cov = MinCovDet().fit(vals[:, None])
            lqr = np.sqrt(cov.covariance_.flatten()[0])
        except Exception:
            low, up = np.percentile(vals, [25, 75])
            lqr = up - low
        up = np.median(vals) + lqr * threshold
        return [k for k in diffs if diffs[k] >= up]
