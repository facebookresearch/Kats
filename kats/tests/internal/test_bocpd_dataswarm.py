# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest
from unittest import TestCase

import numpy as np
import pandas as pd
from kats.internal.client.kats_cli import bocp  # @manual
from kats.internal.client.kats_client import bocpd_detection
from libfb.py.testutil import is_devserver
from scipy.stats import norm  # @manual


DEVSERVER_TEST = unittest.skipUnless(is_devserver(), "Tests only run on devservers.")


# make some time series data with changepoints
# trying to imitate the well-log data in the paper
def make_level_ts():
    np.random.seed(seed=100)
    # constants
    t_start = 0
    t_end = 450

    # calculation
    num_points = t_end - t_start
    y_val = norm.rvs(loc=1.35, scale=0.05, size=num_points)

    # make changepoints
    y_val[100:200] = y_val[100:200] - 0.2
    y_val[350:450] = y_val[350:450] - 0.15

    df = pd.DataFrame({"time": list(range(t_start, t_end)), "value": y_val})

    return df


class testBOCPD(TestCase):
    def test_kats_client(self) -> None:
        level_ts_df = make_level_ts()
        result_df = bocpd_detection(level_ts_df, "%Y-%m-%d", {})
        self.assertTrue(result_df.shape[0] > 0)

    @DEVSERVER_TEST
    def test_kats_cli(self) -> None:
        config = {
            "job_type": "bocp",
            "history_query": """
            SELECT
                time, value
            FROM task_bootcamp_rgreenbaum
            ORDER BY
                time;
            """,
            "history_namespace": "bi",
            "output_table": "task_bootcamp_rgreenbaum_output1",
            "output_namespace": "bi",
            "ds_partition": "2020-09-09",
            "column_types": ["INT", "DOUBLE"],
            "retention": 5,
        }
        bocp(config)
