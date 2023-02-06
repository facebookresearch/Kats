# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""The NeuralProphet model

Neural Prophet model is a neural network based time-series model, inspired by
Facebook Prophet and AR-Net, built on PyTorch.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

try:
    from neuralprophet import NeuralProphet  # noqa

    _no_neuralprophet = False
except ImportError:
    _no_neuralprophet = True
    NeuralProphet = Dict[str, Any]  # for Pyre

TorchLoss = torch.nn.modules.loss._Loss

from kats.consts import Params, TimeSeriesData
from kats.models.model import Model
from kats.utils.parameter_tuning_utils import (
    get_default_neuralprophet_parameter_search_space,
)


def _error_msg(msg: str) -> None:
    logging.error(msg)
    raise ValueError(msg)


class NeuralProphetParams(Params):
    """Parameter class for NeuralProphet model

    This is the parameter class for the neural prophet model. It contains all necessary
    parameters as definied in Prophet implementation:
    https://github.com/ourownstory/neural_prophet/blob/master/neuralprophet/forecaster.py

    Attributes:
          growth: A string to specify no trend or a linear trend. Can be "off" (no trend) or "linear" (linear trend).
              no trend or a linear trend.
              Note: 'discontinuous' setting is actually not a trend per se.
              only use if you know what you do.
          changepoints: A list of dates at which to include potential changepoints.
              If not specified, potential changepoints are selected automatically.
              data format: list of str, list of np.datetimes, np.array of np.datetimes
              (not np.array of np.str)
          n_changepoints: Number of potential changepoints to include.
              Changepoints are selected uniformly from the first `changepoints_range`
              proportion of the history.
              Not used if input `changepoints` is supplied. If `changepoints` is not
              supplied.
          changepoints_range: Proportion of history in which trend changepoints
              wil be estimated. Defaults to 0.9 for the first 90%.
              Not used if `changepoints` is specified.
          trend_reg: Parameter modulating the flexibility of the automatic
              changepoint selection.
              Large values (~1-100) will limit the variability of changepoints.
              Small values (~0.001-1.0) will allow changepoints to change faster.
              default: 0 will fully fit a trend to each segment.
          trend_reg_threshold: Allowance for trend to change
              without regularization.
              True: Automatically set to a value that leads to a smooth trend.
              False: All changes in changepoints are regularized
          yearly_seasonality: Fit yearly seasonality.
              Can be 'auto', True, False, or a number of Fourier/linear terms to generate.
          weekly_seasonality: Fit monthly seasonality.
              Can be 'auto', True, False, or a number of Fourier/linear terms to generate.
          daily_seasonality: Fit daily seasonality.
              Can be 'auto', True, False, or a number of Fourier/linear terms to generate.
          seasonality_mode: 'additive' (default) or 'multiplicative'.
          seasonality_reg: Parameter modulating the strength of the seasonality model.
              Smaller values~0.1-1) allow the model to fit larger seasonal fluctuations,
              larger values~1-100) dampen the seasonality.
              default: 0, no regularization
          n_lags: Previous time series steps to include in auto-regression. Aka AR-order
          ar_reg: [0-100], how much sparsity to enduce in the AR-coefficients.
              Large values (~1-100) will limit the number of nonzero coefficients dramatically.
              Small values (~0.001-1.0) will allow more non-zero coefficients.
              default: 0 no regularization of coefficients.
          n_forecasts: Number of steps ahead of prediction time step to forecast.
          num_hidden_layers: Number of hidden layer to include in AR-Net. defaults to 0.
          d_hidden: dimension of hidden layers of the AR-Net. Ignored if num_hidden_layers == 0.
          learning_rate: Maximum learning rate setting for 1cycle policy scheduler.
              default: None: Automatically sets the learning_rate based on a learning rate range test.
              For manual values, try values ~0.001-10.
          epochs: Number of epochs (complete iterations over dataset) to train model.
              default: None: Automatically sets the number of epochs based on dataset size.
              For best results also leave batch_size to None.
              For manual values, try ~5-500.
          batch_size: Number of samples per mini-batch.
              default: None: Automatically sets the batch_size based on dataset size.
              For best results also leave epochs to None.
              For manual values, try ~1-512.
          newer_samples_weight: Sets factor by which the model fit is skewed towards more recent observations.
              Controls the factor by which final samples are weighted more compared to initial samples.
              Applies a positional weighting to each sample's loss value.
          newer_samples_start: Sets beginning of 'newer' samples as fraction of training data.
              Throughout the range of 'newer' samples, the weight is increased
              from ``1.0/newer_samples_weight`` initially to 1.0 at the end,
              in a monotonously increasing function (cosine from pi to 2*pi).
          loss_func: Type of loss to use: str ['Huber', 'MSE'],
              or torch loss or callable for custom loss, eg. asymmetric Huber loss
          normalize: Type of normalization to apply to the time series.
              options: ['auto', 'soft', 'off', 'minmax, 'standardize']
              default: 'auto' uses 'minmax' if variable is binary, else 'soft'
              'soft' scales minimum to 0.1 and the 90th quantile to 0.9
          impute_missing: Whether to automatically impute missing dates/values
              imputation follows a linear method up to 10 missing values, more are filled with trend.
          custom_seasonalities: Customized seasonalities, dict with keys
              "name", "period", "fourier_order"
          extra_future_regressors: A list of dictionaries representing the additional regressors.
              Each regressor is a dictionary with required key "name"and optional keys "regularization" and "normalize".
          extra_lagged_regressors: A list of dictionaries representing the additional regressors.
              Each regressor is a dictionary with required key "names"and optional keys "regularization" and "normalize".
    """

    changepoints: Optional[Union[List[str], List[np.datetime64], np.ndarray]]
    n_changepoints: int
    changepoints_range: float
    trend_reg: float
    trend_reg_threshold: Union[float, bool]
    yearly_seasonality: Union[str, bool, int]
    weekly_seasonality: Union[str, bool, int]
    daily_seasonality: Union[str, bool, int]
    seasonality_mode: str
    seasonality_reg: float
    n_forecasts: int
    n_lags: int
    num_hidden_layers: int
    d_hidden: Optional[int]
    ar_reg: Optional[float]
    learning_rate: Optional[float]
    epochs: Optional[int]
    batch_size: Optional[int]
    newer_samples_weight: Optional[float]
    newer_samples_start: Optional[float]
    loss_func: Union[str, TorchLoss, Callable[..., float]]
    optimizer: str
    normalize: str
    impute_missing: bool
    custom_seasonalities: List[Dict[str, Any]]
    extra_future_regressors: List[Dict[str, Any]]
    extra_lagged_regressors: List[Dict[str, Any]]

    def __init__(
        self,
        growth: str = "linear",
        # TODO:
        # when Numpy 1.21 is supported (for np.typing), do
        # import np.typing as npt
        # replace 'np.ndarray' by npt.NDArray['np.datetime64']
        changepoints: Optional[
            Union[List[str], List[np.datetime64], np.ndarray]
        ] = None,
        n_changepoints: int = 10,
        changepoints_range: float = 0.9,
        trend_reg: float = 0,
        trend_reg_threshold: Union[float, bool] = False,
        yearly_seasonality: Union[str, bool, int] = "auto",
        weekly_seasonality: Union[str, bool, int] = "auto",
        daily_seasonality: Union[str, bool, int] = "auto",
        seasonality_mode: str = "additive",
        seasonality_reg: float = 0,
        n_forecasts: int = 1,
        n_lags: int = 0,
        num_hidden_layers: int = 0,
        d_hidden: Optional[int] = None,
        ar_reg: Optional[float] = None,
        learning_rate: Optional[float] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        newer_samples_weight: Optional[float] = 2.0,
        newer_samples_start: Optional[float] = 0.0,
        loss_func: Union[str, TorchLoss, Callable[..., float]] = "Huber",
        optimizer: str = "AdamW",
        normalize: str = "auto",
        impute_missing: bool = True,
        custom_seasonalities: Optional[List[Dict[str, Any]]] = None,
        extra_future_regressors: Optional[List[Dict[str, Any]]] = None,
        extra_lagged_regressors: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if _no_neuralprophet:
            raise RuntimeError("requires neuralprophet to be installed")
        super().__init__()
        self.growth = growth
        self.changepoints = changepoints
        self.n_changepoints = n_changepoints
        self.changepoints_range = changepoints_range
        self.trend_reg = trend_reg
        self.trend_reg_threshold = trend_reg_threshold
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_mode = seasonality_mode
        self.seasonality_reg = seasonality_reg
        self.n_forecasts = n_forecasts
        self.n_lags = n_lags
        self.num_hidden_layers = num_hidden_layers
        self.d_hidden = d_hidden
        self.ar_reg = ar_reg
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.newer_samples_weight = newer_samples_weight
        self.newer_samples_start = newer_samples_start
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.normalize = normalize
        self.impute_missing = impute_missing
        self.custom_seasonalities = (
            [] if custom_seasonalities is None else custom_seasonalities
        )
        self.extra_future_regressors = (
            [] if extra_future_regressors is None else extra_future_regressors
        )
        self.extra_lagged_regressors = (
            [] if extra_lagged_regressors is None else extra_lagged_regressors
        )
        self._reqd_regressor_names: List[str] = []
        logging.debug(
            "Initialized Neural Prophet with parameters. "
            f"growth:{growth},"
            f"changepoints:{changepoints},"
            f"n_changepoints:{n_changepoints},"
            f"changepoints_range:{changepoints_range},"
            f"trend_reg:{trend_reg},"
            f"trend_reg_threshold:{trend_reg_threshold},"
            f"yearly_seasonality:{yearly_seasonality},"
            f"weekly_seasonality:{weekly_seasonality},"
            f"daily_seasonality:{daily_seasonality},"
            f"seasonality_mode:{seasonality_mode},"
            f"seasonality_reg:{seasonality_reg},"
            f"n_forecasts:{n_forecasts},"
            f"n_lags:{n_lags},"
            f"num_hidden_layers:{num_hidden_layers},"
            f"d_hidden:{d_hidden},"
            f"ar_reg:{ar_reg},"
            f"learning_rate:{learning_rate},"
            f"epochs:{epochs},"
            f"batch_size:{batch_size},"
            f"newer_samples_weight:{newer_samples_weight},"
            f"newer_samples_start:{newer_samples_start},"
            f"loss_func:{loss_func},"
            f"optimizer:{optimizer},"
            f"normalize:{normalize},"
            f"impute_missing:{impute_missing}"
        )
        self.validate_params()

    def validate_params(self) -> None:
        """Validate Neural Prophet Parameters"""
        # If custom_seasonalities passed, ensure they contain the required keys.
        reqd_seasonality_keys = ["name", "period", "fourier_order"]
        if not all(
            req_key in seasonality
            for req_key in reqd_seasonality_keys
            for seasonality in self.custom_seasonalities
        ):
            msg = f"Custom seasonality dicts must contain the following keys:\n{reqd_seasonality_keys}"
            logging.error(msg)
            raise ValueError(msg)

        self._reqd_regressor_names = []

        # If extra_future_regressors or extra_lagged_regressors passed, ensure
        # they contain the required keys.
        all_future_regressor_keys = {"name", "regularization", "normalize"}
        for regressor in self.extra_future_regressors:
            if not isinstance(regressor, dict):
                msg = f"Elements in `extra_future_regressors` should be a dictionary but receives {type(regressor)}."
                _error_msg(msg)
            if "name" not in regressor:
                msg = "Extra regressor dicts must contain the following keys: 'name'."
                _error_msg(msg)
            else:
                self._reqd_regressor_names.append(regressor["name"])
            if not set(regressor.keys()).issubset(all_future_regressor_keys):
                msg = f"Elements in `extra_future_regressor` should only contain keys in {all_future_regressor_keys} but receives {regressor.keys()}."
                _error_msg(msg)

        all_lagged_regressor_keys = {"names", "regularization", "normalize"}
        for regressor in self.extra_lagged_regressors:
            if not isinstance(regressor, dict):
                msg = f"Elements in `extra_lagged_regressors` should be a dictionary but receives {type(regressor)}."
                _error_msg(msg)
            if "names" not in regressor:
                msg = "Extra regressor dicts must contain the following keys: 'names'."
                _error_msg(msg)
            else:
                self._reqd_regressor_names.append(regressor["names"])
            if not set(regressor.keys()).issubset(all_lagged_regressor_keys):
                msg = f"Elements in `extra_lagged_regressor` should only contain keys in {all_lagged_regressor_keys} but receives {regressor.keys()}."
                _error_msg(msg)


class NeuralProphetModel(Model[NeuralProphetParams]):
    def __init__(self, data: TimeSeriesData, params: NeuralProphetParams) -> None:
        super().__init__(data, params)
        if _no_neuralprophet:
            raise RuntimeError("requires neuralprophet to be installed")
        self.data: TimeSeriesData = data
        self.df: pd.DataFrame
        self.model: Optional[NeuralProphet]
        self._data_params_validation()

        self.df = pd.DataFrame()
        self.model = None

    def _data_params_validation(self) -> None:
        """Validate whether `data` contains specified regressors or not."""
        extra_regressor_names = set(self.params._reqd_regressor_names)
        # univariate case
        if self.data.is_univariate():
            if len(extra_regressor_names) != 0:
                msg = (
                    f"Missing data for extra regressors: {self.params._reqd_regressor_names}! "
                    "Please include the missing regressors in `data`."
                )
                raise ValueError(msg)
        # multivariate case
        else:
            value_cols = set(self.data.value.columns)
            if "y" not in value_cols:
                msg = "`data` should contain a column called `y` representing the responsive value."
                raise ValueError(msg)
            if not extra_regressor_names.issubset(value_cols):
                msg = f"`data` should contain all columns listed in {extra_regressor_names}."
                raise ValueError(msg)

    def _ts_to_df(self) -> pd.DataFrame:
        if self.data.is_univariate():
            # handel corner case: `value` column is not named as `y`.
            df = pd.DataFrame({"ds": self.data.time, "y": self.data.value}, copy=False)
        else:
            df = self.data.to_dataframe()
            df.rename(columns={self.data.time_col_name: "ds"}, inplace=True)

        col_names = self.params._reqd_regressor_names + ["y", "ds"]

        return df[col_names]

    def fit(self, freq: Optional[str] = None, **kwargs: Any) -> None:
        """Fit NeuralProphet model

        Args:
            freq: Optional; A string representing the frequency of timestamps.

        Returns:
            The fitted neuralprophet model object
        """

        logging.debug(
            "Call fit() with parameters: "
            f"growth:{self.params.growth},"
            f"changepoints:{self.params.changepoints},"
            f"n_changepoints:{self.params.n_changepoints},"
            f"changepoints_range:{self.params.changepoints_range},"
            f"trend_reg:{self.params.trend_reg},"
            f"trend_reg_threshold:{self.params.trend_reg_threshold},"
            f"yearly_seasonality:{self.params.yearly_seasonality},"
            f"weekly_seasonality:{self.params.weekly_seasonality},"
            f"daily_seasonality:{self.params.daily_seasonality},"
            f"seasonality_mode:{self.params.seasonality_mode},"
            f"seasonality_reg:{self.params.seasonality_reg},"
            f"n_forecasts:{self.params.n_forecasts},"
            f"n_lags:{self.params.n_lags},"
            f"num_hidden_layers:{self.params.num_hidden_layers},"
            f"d_hidden:{self.params.d_hidden},"
            f"ar_reg:{self.params.ar_reg},"
            f"learning_rate:{self.params.learning_rate},"
            f"epochs:{self.params.epochs},"
            f"batch_size:{self.params.batch_size},"
            f"newer_samples_weight:{self.params.newer_samples_weight},"
            f"newer_samples_start:{self.params.newer_samples_start},"
            f"loss_func:{self.params.loss_func},"
            f"optimizer:{self.params.optimizer},"
            f"normalize:{self.params.normalize},"
            f"impute_missing:{self.params.impute_missing}"
        )

        neuralprophet = NeuralProphet(
            growth=self.params.growth,
            changepoints=self.params.changepoints,
            n_changepoints=self.params.n_changepoints,
            changepoints_range=self.params.changepoints_range,
            trend_reg=self.params.trend_reg,
            trend_reg_threshold=self.params.trend_reg_threshold,
            yearly_seasonality=self.params.yearly_seasonality,
            weekly_seasonality=self.params.weekly_seasonality,
            daily_seasonality=self.params.daily_seasonality,
            seasonality_mode=self.params.seasonality_mode,
            seasonality_reg=self.params.seasonality_reg,
            n_forecasts=self.params.n_forecasts,
            n_lags=self.params.n_lags,
            num_hidden_layers=self.params.num_hidden_layers,
            d_hidden=self.params.d_hidden,
            ar_reg=self.params.ar_reg,
            learning_rate=self.params.learning_rate,
            epochs=self.params.epochs,
            batch_size=self.params.batch_size,
            newer_samples_weight=self.params.newer_samples_weight,
            newer_samples_start=self.params.newer_samples_start,
            loss_func=self.params.loss_func,
            optimizer=self.params.optimizer,
            normalize=self.params.normalize,
            impute_missing=self.params.impute_missing,
        )
        # Prepare dataframe for NeuralProphet.fit()
        self.df = self._ts_to_df()

        # Add any specified custom seasonalities
        for custom_seasonality in self.params.custom_seasonalities:
            neuralprophet.add_seasonality(**custom_seasonality)

        # Add any extra regressors
        for future_regressor in self.params.extra_future_regressors:
            neuralprophet.add_future_regressor(**future_regressor)
        for lagged_regressor in self.params.extra_lagged_regressors:
            neuralprophet.add_lagged_regressor(**lagged_regressor)

        neuralprophet.fit(df=self.df, freq=freq)
        self.model = neuralprophet
        logging.info("Fitted NeuralProphet model.")

    # pyre-fixme[15]: `predict` overrides method defined in `Model` inconsistently.
    def predict(
        self,
        steps: int,
        raw: bool = False,
        future: Optional[pd.DataFrame] = None,
        *args: Any,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Predict with fitted NeuralProphet model.

        Args:
            steps: The steps or length of prediction horizon
            raw: Optional; Whether to return the raw forecasts of prophet model, default is False.
            future: Optional; A `pd.DataFrame` object containing necessary information (e.g., extra regressors) to generate forecasts.
                The length of `future` should be no less than `steps` and it should contain a column named `ds` representing the timestamps.
                Default is None.
        Returns:
            The predicted dataframe with following columns:
                `time`, `fcst`, `fcst_lower`, and `fcst_upper`
        """
        model = self.model
        if model is None:
            raise ValueError("Call fit() before predict().")

        logging.debug(
            "Call predict() with parameters: "
            f"steps:{steps}, raw:{raw}, future:{future}, kwargs:{kwargs}."
        )

        # when extra_regressors are needed
        if (
            len(self.params.extra_future_regressors)
            + len(self.params.extra_lagged_regressors)
            > 0
        ):
            if future is None:
                msg = "`future` should not be None when extra regressors are needed."
                _error_msg(msg)
            elif not set(self.params._reqd_regressor_names).issubset(future.columns):
                msg = "`future` is missing some regressors!"
                _error_msg(msg)
            elif "ds" not in future.columns:
                msg = "`future` should contain a column named 'ds' representing the timestamps."
                _error_msg(msg)
        elif future is None:
            future = model.make_future_dataframe(
                df=self.df,
                periods=steps,
            )

        if len(future) < steps:
            msg = f"Input `future` is not long enough to generate forecasts of {steps} steps."
            _error_msg(msg)
        future.sort_values("ds", inplace=True)

        future["y"] = 0.0
        fcst = model.predict(future)
        if raw:
            return fcst

        logging.info("Generated forecast data from Prophet model.")
        logging.debug("Forecast data: {fcst}".format(fcst=fcst))

        self.fcst_df = fcst_df = pd.DataFrame(
            {k: fcst[k] for k in fcst.columns if k == "ds" or k.startswith("yhat")},
            copy=False,
        )

        logging.debug("Return forecast data: {fcst_df}".format(fcst_df=self.fcst_df))
        return fcst_df

    # pyre-fixme[14]: `kats.models.neuralprophet.NeuralProphetModel.plot` overrides method defined in `Model` inconsistently.
    def plot(
        self, fcst: pd.DataFrame, figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Axes:
        fcst["y"] = None
        # pyre-fixme[16]: `Optional` has no attribute `plot`.
        return self.model.plot(fcst, figsize=figsize)

    def __str__(self) -> str:
        return "NeuralProphet"

    @staticmethod
    # pyre-fixme[15]: `kats.models.neuralprophet.NeuralProphetModel.get_parameter_search_space` overrides method defined in `Model` inconsistently.
    def get_parameter_search_space() -> List[Dict[str, object]]:
        """Get default parameter search space for Prophet model"""
        return get_default_neuralprophet_parameter_search_space()
