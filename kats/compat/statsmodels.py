# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from kats.compat import compat
from statsmodels.tsa import holtwinters

ArrayLike = Union[np.ndarray, Sequence[float]]
Frequency = Union[int, str, pd.Timedelta]


version: compat.Version = compat.Version("statsmodels")


class HoltWintersResults(holtwinters.HoltWintersResults):
    smoothing_trend: float = 0.0

    def __init__(self, results: holtwinters.HoltWintersResults) -> None:
        if version < "0.12":
            super().__init__(results.model, results.params)
            self.params["smoothing_trend"] = self.params["smoothing_slope"]
        else:
            #pyre-fixme
            super().__init__(
                results.model,
                results.params,
                #pyre-fixme
                results.sse,
                #pyre-fixme
                results.aic,
                #pyre-fixme
                results.aicc,
                #pyre-fixme
                results.bic,
                #pyre-fixme
                results.optimized,
                #pyre-fixme
                results.level,
                #pyre-fixme
                results.trend,
                #pyre-fixme
                results.season,

                #pyre-fixme
                results.params_formatted,
                #pyre-fixme
                results.resid,
                #pyre-fixme
                results.k,
                #pyre-fixme
                results.fittedvalues,
                #pyre-fixme
                results.fittedfcast,
                #pyre-fixme
                results.fcastvalues,
                #pyre-fixme
                results.mle_retvals,
            )


class ExponentialSmoothing(holtwinters.ExponentialSmoothing):
    _use_boxcox: bool
    _initialization_method: str

    def __init__(
        self,
        endog: ArrayLike,
        trend: Optional[str] = None,
        damped_trend: bool = False,
        seasonal: Optional[str] = None,
        *,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        dates: Optional[Union[ArrayLike, pd.DatetimeIndex, pd.PeriodIndex]] = None,
        freq: Optional[Frequency] = None,
        initialization_method: str = "estimated",
        initial_level: Optional[float] = None,
        initial_seasonal: Optional[ArrayLike] = None,
        initial_trend: Optional[float] = None,
        missing: str = "none",
        seasonal_periods: Optional[int] = None,
        use_boxcox: bool = False
    ) -> None:
        if version < "0.12":
            self._use_boxcox = use_boxcox
            self._initialization_method = initialization_method
            if bounds is not None:
                logging.warning(
                    "ExponentialSmoothing parameter 'bounds' not supported by statsmodels"
                )
            if initialization_method != "missing":
                logging.warning(
                    "ExponentialSmoothing parameter 'initialization_method' not supported by statsmodels"
                )
            if initial_level is not None:
                logging.warning(
                    "ExponentialSmoothing parameter 'initial_level' not supported by statsmodels"
                )
            if initial_seasonal is not None:
                logging.warning(
                    "ExponentialSmoothing parameter 'initial_seasonal' not supported by statsmodels"
                )
            if initial_trend is not None:
                logging.warning(
                    "ExponentialSmoothing parameter 'initial_trend' not supported by statsmodels"
                )
            if seasonal_periods is not None:
                logging.warning(
                    "ExponentialSmoothing parameter 'seasonal_periods' not supported by statsmodels"
                )
            super().__init__(
                endog,
                trend,
                damped_trend,
                seasonal,
                seasonal_periods,
                dates,
                freq,
                missing,
            )
        else:
            #pyre-fixme
            super().__init__(
                endog,
                trend,
                damped_trend,
                seasonal,
                seasonal_periods=seasonal_periods,
                initialization_method=initialization_method,
                initial_level=initial_level,
                initial_trend=initial_trend,
                initial_seasonal=initial_seasonal,
                use_boxcox=use_boxcox,
                bounds=bounds,
                dates=dates,
                freq=freq,
                missing=missing,
            )

    def fit(self, *args: Any, **kwargs: Any) -> holtwinters.HoltWintersResults:
        if version < "0.12":
            result = super().fit(*args, use_boxcox=self._use_boxcox, **kwargs)
        else:
            result = super().fit(*args, **kwargs)
        return HoltWintersResults(result)
