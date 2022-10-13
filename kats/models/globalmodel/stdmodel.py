# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
import operator
import time
from copy import copy
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool

from typing import cast, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from kats.consts import Params, TimeSeriesData
from kats.models.globalmodel.ensemble import GMEnsemble
from kats.models.globalmodel.model import GMModel
from kats.models.globalmodel.utils import GMParam

from kats.models.prophet import ProphetModel, ProphetParams
from kats.utils.decomposition import TimeSeriesDecomposition
from statsmodels.tsa.tsatools import freq_to_period


class STDGlobalModel:
    """A class for seasonal-trend-decomposed Global model, which utilizes time series seasonality decomposition method to
    enhance seasonality in forecasts.

    Attributes:
        gmparam: A :class:`kats.models.globalmodel.utils.GMParam` object for build global models.
        model_type: A string for the type of global model. Can be 'single' (for `GMModel`) or 'ensemble (for `GMEnsemble`). Default is 'ensemble'.
        decomposition: A string for the method of decompositing the seasonality. Can be 'additive' or 'multiplicative'. Default is 'additive'.
        decomposition_model: A string for the decomposition model. Can be 'prophet', 'stl' or 'seasonal_decompose'. Default is 'prophet'.
        decomposition_params: A `Param` object for the parameters for decomposition model. Default is None.
        fit_trend: A boolean for whether to directly fit global model on extracted trend or not (i.e., only removing seasonality). Default is False.
        fit_original: A boolean for whether to directly fit global model on the original time series. Default is False.
        ensemble_type: Optional; A string representing the ensemble type if `model_type` is 'ensemble'. Can be 'median' or 'mean'. Default is 'median'.
        splits: Optional; An positive integer representing the number of sub-datasets to be built if `model_type` is 'ensemble'. Default is 3.
        overlap: Optional; A boolean representing whether or not sub-datasets overlap with each other or not if `model_type` is 'ensemble'. Default is True.
        replicate: Optional; A positive integer representing the number of global models to be trained on each sub-datasets if `model_type` is 'ensemble'. Default is 1.
        multi: Optional; A boolean representing whether or not to use multi-processing for training and prediction if `model_type` is 'ensemble'. Default is False.
        max_core: Optional; A positive integer representing the number of available cpu cores if `model_type` is 'ensemble'. Default is None, which sets the number of cores to (total_cores - 1) // 2.


    """

    def __init__(
        self,
        gmparam: Optional[GMParam] = None,
        model_type: str = "ensemble",
        decomposition: str = "additive",
        decomposition_model: str = "prophet",
        decomposition_params: Optional[Params] = None,
        fit_trend: bool = False,
        fit_original: bool = False,
        period: Optional[int] = None,
        ensemble_type: str = "median",
        splits: int = 3,
        overlap: bool = True,
        replicate: int = 1,
        multi: bool = False,
        max_core: Optional[int] = None,
    ) -> None:
        self.params = gmparam
        self.decomposition: str = decomposition.lower()
        self.decomposition_model: str = decomposition_model.lower()
        self.decomposition_params = decomposition_params
        self.ensemble_type = ensemble_type
        self.splits = splits
        self.overlap = overlap
        self.replicate = replicate
        self.multi = multi
        self.max_core = max_core

        total_cores = cpu_count()
        if isinstance(max_core, int) and max_core > 0 and max_core <= total_cores:
            self.max_core = max_core
        else:
            self.max_core = max((total_cores - 1) // 2, 1)
            msg = f"Receive invalid max_core = {max_core}, and re-assign max_core = {self.max_core}."
            logging.warning(msg)

        self.fit_trend = fit_trend
        self.fit_original = fit_original
        self.period: Optional[int] = self._get_period(period)

        if self.decomposition == "multiplicative":
            # pyre-fixme Missing attribute annotation [4]: Attribute `deseasonal_operator` of class `STDGlobalModel` has no type specified.
            self.deseasonal_operator = operator.truediv
            # pyre-fixme Missing attribute annotation [4]: Attribute `reseasonal_operator` of class `STDGlobalModel` has no type specified.
            self.reseasonal_operator = operator.mul
        else:
            assert self.decomposition == "additive"
            self.deseasonal_operator = operator.sub
            self.reseasonal_operator = operator.add

        self.gm: Union[None, GMModel, GMEnsemble] = None

    def load_global_model(self, gm: Union[GMModel, GMEnsemble]) -> None:
        """Load pre-trained global model.

        Args:
            gm: a `GMModel` or `GMEnsemble` object for the pre-trained model.

        Returns:
            None.
        """
        self.gm = gm
        self.params = gm.params
        logging.info("Successfully load global model!")

    def _get_period(self, period: Optional[int]) -> Optional[int]:
        """Infer parameter `period` from data frequency if `decomposition_model` is `stl` or `seasonal_decompose`."""
        if period is None and self.decomposition_model in {
            "stl",
            "seasonal_decompose",
        }:
            try:
                assert isinstance(self.params, GMParam)
                period = freq_to_period(self.params.freq)
            except Exception as e:
                logging.warning(
                    f"Fail to infer period via freq with error message {e}."
                    "Please consider setting period yourself based on the input data."
                    "Defaulting to a period of 7"
                )
                period = 7
        return period

    def _decompose(
        self, ts: TimeSeriesData
    ) -> Tuple[Union[ProphetModel, np.ndarray], Dict[str, TimeSeriesData]]:
        """Decompose time series into trend, seasonality."""
        if self.decomposition_model == "prophet":
            if self.decomposition_params is None:
                self.decomposition_params = ProphetParams(
                    seasonality_mode=self.decomposition
                )

            time_col_name = ts.time_col_name

            assert isinstance(self.decomposition_params, ProphetParams)
            decomp_prophet = ProphetModel(data=ts, params=self.decomposition_params)
            decomp_prophet.fit()
            decomp_res = decomp_prophet.predict(steps=0, include_history=True, raw=True)

            decomp_res.rename(columns={"ds": time_col_name}, inplace=True)
            logging.info("Successfully decompose time series data with prophet model.")

            if self.decomposition == "additive":
                tag = "additive_terms"

            else:  # when seasonality is multiplicative
                decomp_res["multiplicative_terms"] += 1.0
                tag = "multiplicative_terms"
            return (
                decomp_prophet,
                {
                    "trend": TimeSeriesData(
                        decomp_res[[time_col_name, "trend"]],
                        time_col_name=time_col_name,
                    ),
                    "seasonal": TimeSeriesData(
                        decomp_res[[time_col_name, tag]], time_col_name=time_col_name
                    ),
                    # "res": TimeSeriesData(decomp_res[[time_col_name, "res"]], time_col_name = time_col_name),
                },
            )
        else:
            tsd = TimeSeriesDecomposition(
                ts, decomposition=self.decomposition, method=self.decomposition_model
            )
            decomp = tsd.decomposer()
            return (decomp["seasonal"].value.values, decomp)

    def _deseasonal(
        self, ts: TimeSeriesData
    ) -> Tuple[Union[ProphetModel, np.ndarray], TimeSeriesData]:
        tsd_model, tsd_res = self._decompose(ts)
        if self.fit_trend:
            return tsd_model, tsd_res["trend"]
        elif self.fit_original:
            return tsd_model, ts
        else:
            new_ts = copy(ts)
            new_ts.value = self.deseasonal_operator(ts.value, tsd_res["seasonal"].value)
            return tsd_model, new_ts

    def _predict_seasonality(
        self, steps: int, tsd_model: Union[ProphetModel, np.ndarray]
    ) -> np.ndarray:

        """Predict the future seasonality.

        Args:
            steps: an integer for the steps to be predicted.

        Returns:
            A `pd.DataFrame` with column named `seasonal` for predicted seasonality.
        """
        if self.decomposition_model == "prophet":
            assert isinstance(tsd_model, ProphetModel)
            fcst = tsd_model.predict(steps=steps, raw=True, include_historty=False)
            assert isinstance(fcst, pd.DataFrame)
            if self.decomposition == "additive":
                ans = fcst["additive_terms"].values
            else:
                fcst["multiplicative_terms"] += 1
                ans = fcst["multiplicative_terms"].values

        else:
            assert isinstance(tsd_model, np.ndarray)
            period = cast(int, self.period)
            seasonality = tsd_model[-period:]
            rep = int(1 + steps // period)

            ans = np.tile(seasonality, rep)[:steps]

        return ans

    def _prepare_ts(
        self, tag: Union[str, int], ts: TimeSeriesData, steps: int
    ) -> Tuple[Union[str, int], TimeSeriesData, np.ndarray]:
        """Prepare time series into seasonality and non-seasonal part."""
        tsd_model, new_ts = self._deseasonal(ts)
        new_seasonal = self._predict_seasonality(steps, tsd_model)
        return (tag, new_ts, new_seasonal)

    def _reseasonal(
        self, tag: Union[int, str], fcst: pd.DataFrame, seasonal_fcst: np.ndarray
    ) -> Tuple[Union[str, int], pd.DataFrame]:
        cols = [c for c in fcst.columns if c != "time"]
        m = len(cols)
        seasonal = np.tile(seasonal_fcst, m).reshape(m, -1).T
        fcst[cols] = self.reseasonal_operator(fcst[cols], seasonal)
        return tag, fcst

    def predict(
        self,
        test_TSs: Union[
            TimeSeriesData, List[TimeSeriesData], Dict[Union[str, int], TimeSeriesData]
        ],
        steps: int,
    ) -> Dict[Union[str, int], pd.DataFrame]:
        """Generate forecasts using the seasonality-trend decomposite global model.

        Args:
            test_TSs: A TimeSeriesDdata object, list or a dictionary of time series to generate forecasts for.
            steps: An integer representing the forecast steps.

        Returns:
            A dictionary of forecasts, whose keys are the ids for time series, and values are the corresponding forecasts.
        """
        if self.gm is None:
            raise ValueError("Should initiate/load a global model first.")
        if isinstance(test_TSs, TimeSeriesData):
            test_TSs = [test_TSs]
        n = len(test_TSs)
        keys = list(test_TSs.keys()) if isinstance(test_TSs, dict) else np.arange(n)
        t0 = time.time()
        if self.multi:
            pool = Pool(self.max_core)
            prepared_TSs = pool.starmap(
                self._prepare_ts, [(key, test_TSs[key], steps) for key in keys]
            )
            pool.close()
            pool.join()
        else:
            prepared_TSs = [self._prepare_ts(key, test_TSs[key], steps) for key in keys]
        logging.info(
            f"Successfully preparing all timeseries with time = {time.time()-t0}"
        )

        # generate fcsts using GM
        # pyre-fixme Undefined attribute [16]: Item `None` of `typing.Union[None, GMEnsemble, GMModel]` has no attribute `predict`.
        gm_fcsts = self.gm.predict({t[0]: t[1] for t in prepared_TSs}, steps=steps)

        if self.multi:
            pool = Pool(self.max_core)
            fcsts = pool.starmap(
                self._reseasonal, [(t[0], gm_fcsts[t[0]], t[2]) for t in prepared_TSs]
            )
            pool.close()
            pool.join()
        else:
            fcsts = [
                self._reseasonal(t[0], cast(pd.DataFrame, gm_fcsts[t[0]]), t[2])
                for t in prepared_TSs
            ]
        return {t[0]: t[1] for t in fcsts}
