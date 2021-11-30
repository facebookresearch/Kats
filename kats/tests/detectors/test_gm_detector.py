# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import TestCase

import numpy as np
import pandas as pd
from kats.consts import TimeSeriesData
from kats.detectors.gm_detector import GMDetectorModel
from kats.models.globalmodel.model import GMModel
from kats.models.globalmodel.serialize import global_model_to_json
from kats.models.globalmodel.utils import GMParam


class TestGMDetector(TestCase):
    def test_detector(self) -> None:

        gm = GMModel(
            GMParam(
                freq="D",
                input_window=10,
                fcst_window=10,
                seasonality=7,
                nn_structure=[[1]],
                state_size=10,
                h_size=5,
                quantile=[0.5, 0.01, 0.05, 0.95, 0.99],
            )
        )
        gm._initiate_nn()

        gm_str = global_model_to_json(gm).encode()

        gmd = GMDetectorModel(
            serialized_model=gm_str,
            scoring_confidence_interval=0.90,
            remove_outliers=True,
            outlier_confidence_interval=0.98,
            outlier_removal_window=7,
            max_abnormal_continuation=3,
        )
        self.assertEqual(gmd.serialize(), gm_str)
        # constant historical data
        historical_data_const = TimeSeriesData(
            time=pd.date_range("2021-05-06", periods=30), value=pd.Series([1.0] * 30)
        )
        data = TimeSeriesData(
            time=pd.date_range("2021-06-06", periods=5), value=pd.Series([1.0] * 5)
        )

        res_const = gmd.fit_predict(data, historical_data_const)
        # For a constant time series, we expect all anomaly values to be zero.
        self.assertTrue(
            res_const.scores.value.abs().sum() == 0,
            "For constant time series all anomaly scores should be zero.",
        )

        # Positive linear historical data
        historical_data_linear = TimeSeriesData(
            time=pd.date_range("2021-05-06", periods=30),
            value=pd.Series(np.arange(1, 30).astype(float)),
        )

        res_linear = gmd.predict(data, historical_data_linear)
        self.assertTrue(
            (res_linear.scores.value.values != 0).sum() > 0,
            "We expect to detect at least one anomaly.",
        )

        # Raise ValueError when the length of historical data is less than the seasonality of GM.
        self.assertRaises(ValueError, gmd.predict, data, historical_data_linear[:5])
        # Raise ValueError when the length of historical data is less than
        # the seasonality of GM plus the outlier_remove_length when remove_outlier is True.
        self.assertRaises(ValueError, gmd.predict, data, historical_data_linear[:12])
