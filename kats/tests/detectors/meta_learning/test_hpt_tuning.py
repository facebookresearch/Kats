# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List
from unittest import TestCase

import numpy as np

import pandas as pd
from kats.detectors.cusum_model import CUSUMDetectorModel
from kats.detectors.meta_learning.hpt_tuning import (
    metadata_detect_reader,
    MetaDetectHptSelect,
    NNParams,
)
from kats.detectors.threshold_detector import StaticThresholdModel

BASE_MODELS = ["cusum", "static"]
MODELS_PARAMS: Dict[str, List[str]] = {
    "cusum": [
        "scan_window",
        "historical_window",
        "threshold_low",
        "threshold_high",
    ],
    "static": (
        [
            "threshold_low",
            "threshold_high",
        ]
    ),
}


def gen_synthmetadata(n: int) -> pd.DataFrame:
    data = []
    features = np.random.randn(n * 40).reshape(n, -1)
    for i in range(n):
        hpt_res = {}
        for key in BASE_MODELS:
            if key == "cusum":
                params = {}
                params["scan_window"] = np.random.rand()  # num
                params["historical_window"] = 10  # constant example
                params["threshold_low"] = np.random.choice([1, 2, 3])  # cat
                params["threshold_high"] = np.random.choice([4, 5, 6])  # cat
            else:
                params = {}
                params["threshold_low"] = np.random.rand()  # num
                params["threshold_high"] = np.random.choice([4, 5, 6])  # cat

            hpt_res[key] = (params, np.random.rand())

        feature_dict = {str(k): features[i, k] for k in range(features.shape[1])}
        best_model = BASE_MODELS[np.random.randint(0, len(BASE_MODELS))]
        data.append([hpt_res, feature_dict, best_model])
    return pd.DataFrame(data, columns=["hpt_res", "features", "best_model"])


class TestMetaDetectHptSelect(TestCase):
    def setUp(self) -> None:
        self.synth_meatdata = gen_synthmetadata(500)
        self.meta_data_cusum = metadata_detect_reader(
            rawdata=self.synth_meatdata,
            algorithm_name="cusum",
            params_to_scale_down=set(),
        )
        self.meta_data_static = metadata_detect_reader(
            rawdata=self.synth_meatdata,
            algorithm_name="static",
            params_to_scale_down=set(),
        )
        self.nnparams = NNParams(
            scale=False,
            loss_scale=1.0,
            lr=0.001,
            n_epochs=10,
            batch_size=100,
            method="SGD",
            val_size=0.1,
            momentum=0.9,
        )

    def test_metalearn_flow_cusum(self) -> None:
        datax = self.meta_data_cusum["data_x"]
        datay = self.meta_data_cusum["data_y"]

        mdhs = MetaDetectHptSelect(
            data_x=datax, data_y=datay, detector_model=CUSUMDetectorModel
        )

        const_params_dict = mdhs.const_params_dict
        self.assertEqual(len(const_params_dict), 1)
        self.assertEqual(list(const_params_dict.keys())[0], "historical_window")
        self.assertEqual(list(const_params_dict.values())[0], 10)

        self.assertEqual(mdhs._data_y.shape, (500, 3))

        mdhs.train(
            num_idx=["scan_window"],
            cat_idx=["threshold_low", "threshold_high"],
            n_hidden_shared=[20],
            n_hidden_cat_combo=[[5], [5]],
            n_hidden_num=[5],
            nnparams=self.nnparams,
        )

        mdhs.plot()

        res = mdhs.get_hpt_from_features(np.random.normal(0, 1, [10, 40]))
        self.assertEqual(res.shape, (10, 4))

    def test_metalearn_flow_static(self) -> None:
        datax = self.meta_data_static["data_x"]
        datay = self.meta_data_static["data_y"]

        mdhs = MetaDetectHptSelect(
            data_x=datax, data_y=datay, detector_model=StaticThresholdModel
        )
        const_params_dict = mdhs.const_params_dict
        self.assertEqual(len(const_params_dict), 0)

        self.assertEqual(mdhs._data_y.shape, (500, 2))

        mdhs.train(
            num_idx=["threshold_low"],
            cat_idx=["threshold_high"],
            n_hidden_shared=[20],
            n_hidden_cat_combo=[[5]],
            n_hidden_num=[5],
            nnparams=self.nnparams,
        )

        mdhs.plot()

        res = mdhs.get_hpt_from_features(np.random.normal(0, 1, [10, 40]))
        self.assertEqual(res.shape, (10, 2))

    def test_module_errors(self) -> None:
        datax = self.meta_data_cusum["data_x"]
        datay = self.meta_data_cusum["data_y"]

        mdhs = MetaDetectHptSelect(
            data_x=datax, data_y=datay, detector_model=CUSUMDetectorModel
        )

        # provide constant params "historical_window"
        with self.assertRaises(ValueError):
            mdhs.train(
                num_idx=["scan_window", "historical_window"],
                cat_idx=["threshold_low", "threshold_high"],
                n_hidden_shared=[20],
                n_hidden_cat_combo=[[5], [5]],
                n_hidden_num=[5],
                nnparams=self.nnparams,
            )

        # haven't been trained
        with self.assertRaises(AssertionError):
            _ = mdhs.get_hpt_from_features(np.random.normal(0, 1, [10, 40]))

        mdhs.train(
            num_idx=["scan_window"],
            cat_idx=["threshold_low", "threshold_high"],
            n_hidden_shared=[20],
            n_hidden_cat_combo=[[5], [5]],
            n_hidden_num=[5],
            nnparams=self.nnparams,
        )

        # unmatched dimension
        with self.assertRaises(RuntimeError):
            _ = mdhs.get_hpt_from_features(np.random.normal(0, 1, [10, 30]))

        # unmatched format
        with self.assertRaises(IndexError):
            _ = mdhs.get_hpt_from_features(np.random.normal(0, 1, 40))
