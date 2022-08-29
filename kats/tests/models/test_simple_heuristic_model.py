# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest import TestCase

import numpy as np
import pandas as pd
from kats.consts import TimeSeriesData
from kats.models.simple_heuristic_model import (
    SimpleHeuristicModel,
    SimpleHeuristicModelParams,
)
from parameterized.parameterized import parameterized

test_univariate_df = pd.DataFrame(
    {"ds": pd.date_range("2022-01-01", periods=30), "y": np.arange(1, 31)}
)
test_univariate_ts = TimeSeriesData(test_univariate_df, time_col_name="ds")


METHODS = ["last", "mean", "median", "percentile"]


class testSimpleHeuristicModel(TestCase):
    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator `parameter...
    @parameterized.expand(
        [
            ("last", False),
            ("mean", False),
            ("median", True),
            ("percentile", True),
        ]
    )
    def test_univariate_data(
        self,
        test_name: str,
        include_history: bool,
    ) -> None:
        horizon = 10
        params = SimpleHeuristicModelParams(method=test_name)
        m = SimpleHeuristicModel(data=test_univariate_ts, params=params)
        m.fit()
        fcst_df = m.predict(steps=horizon, include_history=include_history)
        fcst_df = fcst_df.iloc[-horizon:, :]

        if test_name == "last":
            self.assertTrue(test_univariate_df.y[-horizon:].sum() == fcst_df.fcst.sum())
        elif test_name == "mean":
            self.assertTrue(
                np.mean(
                    np.reshape(np.asarray(test_univariate_df.y), [-1, horizon]), 0
                ).sum()
                == fcst_df.fcst.sum()
            )
        elif test_name == "median":
            self.assertTrue(
                np.median(
                    np.reshape(np.asarray(test_univariate_df.y), [-1, horizon]), 0
                ).sum()
                == fcst_df.fcst.sum()
            )
        else:
            self.assertTrue(test_name == "percentile")
            # check percentile 95 (default value)
            self.assertTrue(
                np.percentile(
                    np.reshape(np.asarray(test_univariate_df.y), [-1, horizon]), 95, 0
                ).sum()
                == fcst_df.fcst.sum()
            )

            # check percentile 80 (not default)
            params2 = SimpleHeuristicModelParams(method=test_name, quantile=80)
            m2 = SimpleHeuristicModel(data=test_univariate_ts, params=params2)
            m2.fit()
            fcst_df2 = m2.predict(steps=horizon)
            self.assertTrue(
                np.percentile(
                    np.reshape(np.asarray(test_univariate_df.y), [-1, horizon]), 80, 0
                ).sum()
                == fcst_df2.fcst.sum()
            )


if __name__ == "__main__":
    unittest.main()
