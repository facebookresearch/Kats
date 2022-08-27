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
            # pyre-fixme[20]: Argument `aic` expected.
            super().__init__(results.model, results.params)
            self.params["smoothing_trend"] = self.params["smoothing_slope"]
        else:
            super().__init__(
                results.model,
                results.params,
                results.sse,
                results.aic,
                results.aicc,
                results.bic,
                results.optimized,
                results.level,
                results.trend,
                results.season,
                results.params_formatted,
                results.resid,
                results.k,
                results.fittedvalues,
                results.fittedfcast,
                results.fcastvalues,
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
        use_boxcox: bool = False,
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
