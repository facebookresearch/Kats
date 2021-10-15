# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The NeuralProphet model

Neural Prophet model is a neural network based time-series model, inspired by
Facebook Prophet and AR-Net, built on PyTorch.
"""

import logging

try:
    from neuralprophet import NeuralProphet

    _no_neural_prophet = False
except ImportError:
    _no_neural_prophet = True
    NeuralProphet = None  # for Pyre

from kats.consts import Params


class NeuralProphetParams(Params):
    """Parameter class for NeuralProphet model

    This is the parameter class for the neural prophet model. It contains all necessary
    parameters as definied in Prophet implementation:
    https://github.com/ourownstory/neural_prophet/blob/master/neuralprophet/forecaster.py

    Attributes:

       ## Trend Parameters:
          growth (str): ['off', 'linear'] to specify
              no trend or a linear trend.
              Note: 'discontinuous' setting is actually not a trend per se.
              only use if you know what you do.
          changepoints list: Dates at which to include potential changepoints.
              If not specified, potential changepoints are selected automatically.
              data format: list of str, list of np.datetimes, np.array of np.datetimes
              (not np.array of np.str)
          n_changepoints (int): Number of potential changepoints to include.
              Changepoints are selected uniformly from the first `changepoints_range`
              proportion of the history.
              Not used if input `changepoints` is supplied. If `changepoints` is not
              supplied.
          changepoints_range (float): Proportion of history in which trend changepoints
              wil be estimated. Defaults to 0.8 for the first 80%.
              Not used if `changepoints` is specified.
          trend_reg (float): Parameter modulating the flexibility of the automatic
              changepoint selection.
              Large values (~1-100) will limit the variability of changepoints.
              Small values (~0.001-1.0) will allow changepoints to change faster.
              default: 0 will fully fit a trend to each segment.
          trend_reg_threshold (bool, float): Allowance for trend to change
              without regularization.
              True: Automatically set to a value that leads to a smooth trend.
              False: All changes in changepoints are regularized

       ## Seasonality Parameters
          yearly_seasonality (bool, int): Fit yearly seasonality.
              Can be 'auto', True, False, or a number of Fourier/linear terms to generate.
          weekly_seasonality (bool, int): Fit monthly seasonality.
              Can be 'auto', True, False, or a number of Fourier/linear terms to generate.
          daily_seasonality (bool, int): Fit daily seasonality.
              Can be 'auto', True, False, or a number of Fourier/linear terms to generate.
          seasonality_mode (str): 'additive' (default) or 'multiplicative'.
          seasonality_reg (float): Parameter modulating the strength of the seasonality model.
              Smaller values (~0.1-1) allow the model to fit larger seasonal fluctuations,
              larger values (~1-100) dampen the seasonality.
              default: None, no regularization

       ## AR Parameters
          n_lags (int): Previous time series steps to include in auto-regression. Aka AR-order
          ar_sparsity (float): [0-1], how much sparsity to enduce in the AR-coefficients.
              Should be around (# nonzero components) / (AR order), eg. 3/100 = 0.03

       ## Neural Network Model Parameters
          n_forecasts (int): Number of steps ahead of prediction time step to forecast.
          num_hidden_layers (int): number of hidden layer to include in AR-Net. defaults to 0.
          d_hidden (int): dimension of hidden layers of the AR-Net. Ignored if num_hidden_layers == 0.

      ## Train Parameters
          learning_rate (float): Maximum learning rate setting for 1cycle policy scheduler.
              default: None: Automatically sets the learning_rate based on a learning rate range test.
              For manual values, try values ~0.001-10.
          epochs (int): Number of epochs (complete iterations over dataset) to train model.
              default: None: Automatically sets the number of epochs based on dataset size.
              For best results also leave batch_size to None.
              For manual values, try ~5-500.
          batch_size (int): Number of samples per mini-batch.
              default: None: Automatically sets the batch_size based on dataset size.
              For best results also leave epochs to None.
              For manual values, try ~1-512.
          loss_func (str, torch.nn.modules.loss._Loss, 'typing.Callable'):
              Type of loss to use: str ['Huber', 'MSE'],
              or torch loss or callable for custom loss, eg. asymmetric Huber loss
          train_speed (int, float) a quick setting to speed up or slow down
              model fitting [-3, -2, -1, 0, 1, 2, 3]
              potentially useful when under-, over-fitting, or simply in a hurry.
              applies
              epochs *= 2**-train_speed, batch_size *= 2**train_speed, learning_rate *= 2**train_speed,
              default None: equivalent to 0.

       ## Data Parameters
          normalize (str): Type of normalization to apply to the time series.
              options: ['auto', 'soft', 'off', 'minmax, 'standardize']
              default: 'auto' uses 'minmax' if variable is binary, else 'soft'
              'soft' scales minimum to 0.1 and the 90th quantile to 0.9
          impute_missing (bool): whether to automatically impute missing dates/values
              imputation follows a linear method up to 10 missing values, more are filled with trend.
    """

    def __init__(
        self,
        growth="linear",
        changepoints=None,
        n_changepoints=10,
        changepoints_range=0.9,
        trend_reg=0,
        trend_reg_threshold=False,
        yearly_seasonality="auto",
        weekly_seasonality="auto",
        daily_seasonality="auto",
        seasonality_mode="additive",
        seasonality_reg=0,
        n_forecasts=1,
        n_lags=0,
        num_hidden_layers=0,
        d_hidden=None,
        ar_sparsity=None,
        learning_rate=None,
        epochs=None,
        batch_size=None,
        loss_func="Huber",
        optimizer="AdamW",
        train_speed=None,
        normalize="auto",
        impute_missing=True,
    ) -> None:
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
        self.ar_sparsity = ar_sparsity
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.train_speed = train_speed
        self.normalize = normalize
        self.impute_missing = impute_missing
        if _no_neural_prophet:
            raise RuntimeError("requires neuralprophet to be installed")

        logging.debug(
            "Initialized Neural Prophet with parameters. "
            "growth:{growth},"
            "changepoints:{changepoints},"
            "n_changepoints:{n_changepoints},"
            "changepoints_range:{changepoints_range},"
            "trend_reg:{trend_reg},"
            "trend_reg_threshold:{trend_reg_threshold},"
            "yearly_seasonality:{yearly_seasonality},"
            "weekly_seasonality:{weekly_seasonality},"
            "daily_seasonality:{daily_seasonality},"
            "seasonality_mode:{seasonality_mode},"
            "seasonality_reg:{seasonality_reg},"
            "n_forecasts:{n_forecasts},"
            "n_lags:{n_lags},"
            "num_hidden_layers:{num_hidden_layers},"
            "d_hidden:{d_hidden},"
            "ar_sparsity:{ar_sparsity},"
            "learning_rate:{learning_rate},"
            "epochs:{epochs},"
            "batch_size:{batch_size},"
            "loss_func:{loss_func},"
            "optimizer:{optimizer},"
            "train_speed:{train_speed},"
            "normalize:{normalize},"
            "impute_missing:{impute_missing}".format(
                growth=growth,
                changepoints=changepoints,
                n_changepoints=n_changepoints,
                changepoints_range=changepoints_range,
                trend_reg=trend_reg,
                trend_reg_threshold=trend_reg_threshold,
                yearly_seasonality=yearly_seasonality,
                weekly_seasonality=weekly_seasonality,
                daily_seasonality=daily_seasonality,
                seasonality_mode=seasonality_mode,
                seasonality_reg=seasonality_reg,
                n_forecasts=n_forecasts,
                n_lags=n_lags,
                num_hidden_layers=num_hidden_layers,
                d_hidden=d_hidden,
                ar_sparsity=ar_sparsity,
                learning_rate=learning_rate,
                epochs=epochs,
                batch_size=batch_size,
                loss_func=loss_func,
                optimizer=optimizer,
                train_speed=train_speed,
                normalize=normalize,
                impute_missing=impute_missing,
            )
        )

    def validate_params(self):
        """Validate Neural Prophet Parameters"""
        logging.debug("Not yet implemented")
        pass
