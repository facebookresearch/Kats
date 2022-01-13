from datetime import datetime, timedelta
from unittest import TestCase

import numpy as np
import pandas as pd
from kats.consts import TimeSeriesData
from kats.internal.gp_model.autoforecast_daily import (
    AutomaticForecastingGPModel,
    GaussianProcessModelParams,
)
from parameterized import parameterized


class GaussianProcessModelTest(TestCase):
    # pyre-fixme[16]: Module `parameterized.parameterized` has no attribute `expand`.
    @parameterized.expand(
        [
            (
                GaussianProcessModelParams(
                    "y", adam_learning_rate=1e-2, training_iters=100
                ),
                False,
            ),
            (
                GaussianProcessModelParams(
                    "y",
                    linear_covariates=("a",),
                    adam_learning_rate=1e-2,
                    training_iters=100,
                ),
                True,
            ),
            (
                GaussianProcessModelParams(
                    "y",
                    linear_covariates=("a",),
                    nonlinear_covariates=("b"),
                    adam_learning_rate=1e-2,
                    training_iters=100,
                ),
                False,
            ),
        ]
    )
    def test_af_model_smoke(
        self, params: GaussianProcessModelParams, include_history: bool
    ) -> None:
        data = TimeSeriesData(
            pd.DataFrame(
                {
                    "t": np.arange(
                        datetime(2005, 1, 1), datetime(2005, 4, 11), timedelta(days=1)
                    ),
                    "a": np.random.randn(100),
                    "b": np.random.randn(100),
                    "y": np.random.randn(100),
                }
            ),
            time_col_name="t",
        )
        N = 90
        train_data, test_data = data[:N], data[N:]
        model = AutomaticForecastingGPModel(train_data, params)
        model.fit()
        y = model.predict(test_data, include_history=include_history)
        assert len(y["fcst"]) == 100 if include_history else 10
        model.plot()
