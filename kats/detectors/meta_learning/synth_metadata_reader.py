# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This module reads synthetic data from daiquery and preprocess it as data_x=features, data_y=hpt_res for given algorithm
"""

import io
import os
import pkgutil
from typing import Any, Dict, Optional, Set

import pandas as pd


class SynthMetadataReader:
    NUM_SECS_IN_DAY: int = 3600 * 24
    PARAMS_TO_SCALE_DOWN: Set[str] = {
        "n_control",
        "n_test",
        "historical_window",
        "scan_window",
    }
    _rawdata: Optional[pd.DataFrame] = None
    _metadata: Optional[Dict[str, Any]] = None

    def _get_raw_data(self) -> pd.DataFrame:
        rawdata = self._rawdata
        if rawdata is None:
            root = "kats"
            path = "data/" if "kats" in os.getcwd().lower() else "kats/data/"
            filename = "meta_learning_detection_training_data_pmo_sample_1000.csv"
            data_object = pkgutil.get_data(root, path + filename)
            assert data_object is not None
            self._rawdata = rawdata = pd.read_csv(
                io.BytesIO(data_object),
                index_col=0,
                dtype={"idx": object},
                encoding="utf8",
            )
        return rawdata.copy()

    def get_metadata(self, algorithm_name: str) -> Dict[str, pd.DataFrame]:
        metadata = self._metadata
        if metadata is None:
            rawdata = self._get_raw_data()

            metadata = {}
            metadata["data_x"] = (
                rawdata.features.map(eval)
                .map(lambda d: {k: float(v) for k, v in d.items()})
                .apply(pd.Series)  # expend dict to columns
            )
            algorithm_names = (
                rawdata.hpt_res.map(eval)
                .map(lambda kv: list(kv.keys()))
                .explode()
                .unique()
                .tolist()
            )

            metadata["data_y"] = {}
            for a in algorithm_names:
                metadata["data_y"][a] = (
                    rawdata.hpt_res.map(eval)
                    .map(lambda kv: kv[a][0])
                    .map(
                        lambda kv: {
                            k: v
                            if k not in self.PARAMS_TO_SCALE_DOWN
                            else v / SynthMetadataReader.NUM_SECS_IN_DAY
                            for k, v in kv.items()
                        }
                    )
                    .apply(pd.Series)  # expend dict to columns
                    .convert_dtypes(convert_integer=False)
                )

            self._metadata = metadata
        return {
            "data_x": metadata["data_x"].copy(),
            "data_y": metadata["data_y"][algorithm_name].copy(),
        }
