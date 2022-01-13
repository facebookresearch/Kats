# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from unittest import TestCase

import numpy as np
import pandas as pd
from kats.consts import TimeSeriesData
from kats.tsfeatures.transformer import transform
from kats.tsfeatures.tsfeatures import TsFeatures


class TSfeaturesTransformerTest(TestCase):
    def test_transformer(self) -> None:
        # single timeseries data for testing
        metadata = {}
        line = {
            "id": "182328635174555",
            "values": "[13, 0, 1, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 1,\
             0, 0, 0, 0, 0, 1, 0, 5, 2, 0, 0, 1, 6, 0, 0, 0, 0, 0, 0, 7,\
             0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]",
        }
        # test if the default method works in transformer
        hive_transformed = next(transform(metadata, line))
        hive_transformed.pop("id")

        values = json.loads(line["values"])
        ts_obj = TimeSeriesData(
            pd.DataFrame({"time": list(range(len(values))), "values": values})
        )
        TsFeature_transformed = TsFeatures().transform(ts_obj)
        self.assertEqual(hive_transformed, TsFeature_transformed)

        # single timeseries data for testing string IDs
        metadata = {}
        line = {
            "id": "ids_ID-182328635174555",
            "values": "[13, 0, 1, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 1,\
             0, 0, 0, 0, 0, 1, 0, 5, 2, 0, 0, 1, 6, 0, 0, 0, 0, 0, 0, 7,\
             0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]",
        }
        # test if the default method works in transformer
        hive_transformed = next(transform(metadata, line))
        hive_transformed.pop("id")

        values = json.loads(line["values"])
        ts_obj = TimeSeriesData(
            pd.DataFrame({"time": list(range(len(values))), "values": values})
        )
        TsFeature_transformed = TsFeatures().transform(ts_obj)
        self.assertEqual(hive_transformed, TsFeature_transformed)

        # test if opt-in features works
        selected_features = [
            "trend_strength",
            "seasonality_strength",
            "diff1y_acf1",
            "holt_alpha",
        ]
        metadata = {"selected_features": ",".join(selected_features)}
        hive_transformed = next(transform(metadata, line))
        hive_transformed.pop("id")

        TsFeature_transformed = TsFeatures(
            selected_features=selected_features
        ).transform(ts_obj)
        self.assertEqual(hive_transformed, TsFeature_transformed)

        # test if opt-in groups works
        selected_groups = [
            "special_ac",
            "stl_features",
            "statistics",
        ]
        metadata = {"selected_features": ",".join(selected_groups)}
        hive_transformed = next(transform(metadata, line))
        hive_transformed.pop("id")

        TsFeature_transformed = TsFeatures(selected_features=selected_groups).transform(
            ts_obj
        )
        self.assertEqual(hive_transformed, TsFeature_transformed)

        # test if the difference on cumulatively-summed time series works
        metadata = {"difference": True}
        line = {
            "id": "182328635174555",
            "values": "[13, 0, 1, 8, 0, 0, 0, 1, 0, 5, 2, 0, 0, 1, 6, 0, 7]",
        }
        line_cusum = {
            "id": "182328635174556",
            "values": "[13, 13, 14, 22, 22, 22, 22, 23, 23, 28, 30, 30, 30, 31, 37, 37, 44]",
        }
        hive_transformed = next(transform(metadata, line_cusum))
        hive_transformed.pop("id")

        values = json.loads(line["values"])
        ts_obj = TimeSeriesData(
            pd.DataFrame({"time": list(range(len(values))), "values": values})
        )
        TsFeature_transformed = TsFeatures().transform(ts_obj)
        self.assertEqual(hive_transformed, TsFeature_transformed)

        # test if the treatment on flipped time series works
        metadata = {"reverse": True}
        line = {
            "id": "182328635174555",
            "values": "[13, 0, 1, 8, 0, 0, 0, 1, 0, 5, 2, 0, 0, 1, 6, 0, 7]",
        }
        line_flip = {
            "id": "182328635174556",
            "values": "[7, 0, 6, 1, 0, 0, 2, 5, 0, 1, 0, 0, 0, 8, 1, 0, 13]",
        }
        hive_transformed = next(transform(metadata, line_flip))
        hive_transformed.pop("id")

        values = json.loads(line["values"])
        ts_obj = TimeSeriesData(
            pd.DataFrame({"time": list(range(len(values))), "values": values})
        )
        TsFeature_transformed = TsFeatures().transform(ts_obj)
        self.assertEqual(hive_transformed, TsFeature_transformed)

        # test if the treatment on flipped, and then cumulatively-summed time series works
        metadata = {"difference": True, "reverse": True}
        line = {
            "id": "182328635174555",
            "values": "[13, 0, 1, 8, 0, 0, 0, 1, 0, 5, 2, 0, 0, 1, 6, 0, 7]",
        }
        line_diff_flip = {
            "id": "182328635174556",
            "values": "[7, 7, 13, 14, 14, 14, 16, 21, 21, 22, 22, 22, 22, 30, 31, 31, 44]",
        }
        hive_transformed = next(transform(metadata, line_diff_flip))
        hive_transformed.pop("id")

        values = json.loads(line["values"])
        ts_obj = TimeSeriesData(
            pd.DataFrame({"time": list(range(len(values))), "values": values})
        )
        TsFeature_transformed = TsFeatures().transform(ts_obj)
        self.assertEqual(hive_transformed, TsFeature_transformed)

        # test if method works for single value array
        __selected_features = ["diff1y_acf1", "holt_alpha"]
        __metadata = {
            "difference": True,
            "selected_features": ",".join(__selected_features),
        }
        __line = {"id": "182328635174555", "values": "[16]"}
        __hive_transformed = next(transform(__metadata, __line))
        __hive_transformed.pop("id")

        self.assertEqual(
            __hive_transformed, {"diff1y_acf1": np.nan, "holt_alpha": np.nan}
        )

        # test if method works for null value array
        __selected_features = ["diff1y_acf1", "holt_alpha"]
        __metadata = {"selected_features": ",".join(__selected_features)}
        __line = {"id": "182328635174555", "values": None}
        __hive_transformed = next(transform(__metadata, __line))
        __hive_transformed.pop("id")

        self.assertEqual(
            __hive_transformed, {"diff1y_acf1": np.nan, "holt_alpha": np.nan}
        )
