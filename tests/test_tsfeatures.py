#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import TestCase

import json
import numpy as np
import pandas as pd
from infrastrategy.kats.consts import TimeSeriesData
from infrastrategy.kats.tsfeatures.tsfeatures import TsFeatures
from infrastrategy.kats.tsfeatures.transformer import transform

DATA = pd.read_csv("infrastrategy/kats/data/air_passengers.csv")
DATA.columns = ["time", "y"]
TSData = TimeSeriesData(DATA)
TSData_short = TimeSeriesData(DATA.iloc[:8, :])
TSData_mini = TimeSeriesData(DATA.iloc[:2, :])


class TSfeaturesTest(TestCase):
    def test_tsfeatures(self):
        feature_vector = TsFeatures().transform(TSData)

        feature_vector_round = {key: round(feature_vector[key], 6) for key in feature_vector}

        # test there is no nan in feature vector
        self.assertEqual(
            np.isnan(np.asarray(list(feature_vector.values()))).any(),
            False,
        )

        # test there are 40 features in the feature vector now
        self.assertEqual(
            len(np.asarray(list(feature_vector.values()))) == 40,
            True,
        )

        # test feature vector value
        self.assertEqual(
            feature_vector_round,
            {
                'length': 144,
                'mean': 280.298611,
                'var': 14291.973331,
                'entropy': -0.428737,
                'lumpiness': 3041164.562906,
                'stability': 12303.627267,
                'flat_spots': 2,
                'hurst': -0.080233,
                'std1st_der': 27.206288,
                'crossing_points': 7,
                'binarize_mean': 0.444444,
                'unitroot_kpss': 0.128475,
                'heterogeneity': 126.064506,
                'histogram_mode': 155.8,
                'linearity': 0.853638,
                'trend_strength': 0.9681,
                'seasonality_strength': 0.440863,
                'spikiness': 33.502886,
                'peak': 6,
                'trough': 3,
                'level_shift_idx': 118,
                'level_shift_size': 15.6,
                'y_acf1': 0.948047,
                'y_acf5': 3.392072,
                'diff1y_acf1': 0.302855,
                'diff1y_acf5': 0.259459,
                'diff2y_acf1': -0.191006,
                'diff2y_acf5': 0.134207,
                'y_pacf5': 1.003288,
                'diff1y_pacf5': 0.219412,
                'diff2y_pacf5': 0.26101,
                'seas_acf1': 0.662904,
                'seas_pacf1': 0.15617,
                'firstmin_ac': 8,
                'firstzero_ac': 52,
                'holt_alpha': 1.0,
                'holt_beta': 0.0,
                'hw_alpha': 0.842106,
                'hw_beta': 0.052631,
                'hw_gamma': 0.157901
            }
        )

    def test_feature_selections(self):
        # test disabling functions
        disable_features = {
            'unitroot_kpss': False,
            'histogram_mode': False,
            'diff2y_pacf5': False,
            'firstmin_ac': False,
        }
        feature_vector = TsFeatures(**disable_features).transform(TSData)
        feature_vector_round = {key: round(feature_vector[key], 6) for key in feature_vector}

        # test if there are 36 features in the feature vector now
        self.assertEqual(
            len(np.asarray(list(feature_vector.values()))) == (TsFeatures()._total_feature_len_ - len(disable_features) - 28),
            True,
        )

        # test feature vector value
        self.assertEqual(
            feature_vector_round,
            {
                'length': 144,
                'mean': 280.298611,
                'var': 14291.973331,
                'entropy': -0.428737,
                'lumpiness': 3041164.562906,
                'stability': 12303.627267,
                'flat_spots': 2,
                'hurst': -0.080233,
                'std1st_der': 27.206288,
                'crossing_points': 7,
                'binarize_mean': 0.444444,
                'heterogeneity': 126.064506,
                'linearity': 0.853638,
                'trend_strength': 0.9681,
                'seasonality_strength': 0.440863,
                'spikiness': 33.502886,
                'peak': 6,
                'trough': 3,
                'level_shift_idx': 118,
                'level_shift_size': 15.6,
                'y_acf1': 0.948047,
                'y_acf5': 3.392072,
                'diff1y_acf1': 0.302855,
                'diff1y_acf5': 0.259459,
                'diff2y_acf1': -0.191006,
                'diff2y_acf5': 0.134207,
                'y_pacf5': 1.003288,
                'diff1y_pacf5': 0.219412,
                'seas_acf1': 0.662904,
                'seas_pacf1': 0.15617,
                'firstzero_ac': 52,
                'holt_alpha': 1.0,
                'holt_beta': 0.0,
                'hw_alpha': 0.842106,
                'hw_beta': 0.052631,
                'hw_gamma': 0.157901
            }
        )

        # test selecting features
        features = [
            'var',
            'linearity',
            'spikiness',
            'trough',
            'holt_alpha',
            'hw_gamma',
            'level_shift_idx'
        ]
        feature_vector = TsFeatures(selected_features = features).transform(TSData)
        feature_vector_round = {key: round(feature_vector[key], 6) for key in feature_vector}

        # test if there are 7 features in the feature vector now
        self.assertEqual(
            len(np.asarray(list(feature_vector.values()))) == len(features),
            True,
        )

        # test feature vector value
        self.assertEqual(
            feature_vector_round,
            {
                'var': 14291.973331,
                'linearity': 0.853638,
                'spikiness': 33.502886,
                'trough': 3,
                'level_shift_idx': 118,
                'holt_alpha': 1.0,
                'hw_gamma': 0.157901
            }
        )

        # test selecting extension features
        extension_features = [
            "cusum_num",
            "cusum_conf",
            "cusum_cp_index",
            "cusum_delta",
            "cusum_llr",
            "cusum_regression_detected",
            "cusum_stable_changepoint",
            "cusum_p_value",
            "robust_num",
            "robust_metric_mean",
            "bocp_num",
            "bocp_conf_max",
            "bocp_conf_mean",
            "outlier_num",
            "trend_num",
            "trend_num_increasing",
            "trend_avg_abs_tau",
            "ROC_5",
            "MA_5",
            "MOM_5",
            "LAG_5",
            "MACD_26_5",
            "MACDsign_26_5",
            "MACDdiff_26_5",
            "seasonal_period",
            "seasonality_mag",
            "trend_mag",
            "residual_std",
        ]
        feature_vector = TsFeatures(selected_features = extension_features).transform(TSData)
        feature_vector_round = {key: round(feature_vector[key], 6) for key in feature_vector}

        # test if there are 7 features in the feature vector now
        self.assertEqual(
            len(np.asarray(list(feature_vector.values()))) == len(extension_features),
            True,
        )

        # test feature vector value
        self.assertEqual(
            feature_vector_round,
            {
                'cusum_num': 1,
                'cusum_conf': 1.0,
                'cusum_cp_index': 0.527778,
                'cusum_delta': 199.098856,
                'cusum_llr': 168.663483,
                'cusum_regression_detected': 1,
                'cusum_stable_changepoint': 1,
                'cusum_p_value': 0.0,
                'robust_num': 3,
                'robust_metric_mean': -31.866667,
                'bocp_num': 3,
                'bocp_conf_max': 0.677218,
                'bocp_conf_mean': 0.587680,
                'outlier_num': 0,
                'trend_num': 2,
                'trend_num_increasing': 0,
                'trend_avg_abs_tau': 0.821053,
                'ROC_5': 0.062858,
                'MA_5': 280.417143,
                'MOM_5': 12.841727,
                'LAG_5': 273.136691,
                'MACD_26_5': -23.360015,
                'MACDsign_26_5': -23.248081,
                'MACDdiff_26_5': -1.142731,
                'seasonal_period': 7,
                'trend_mag': 2.404464,
                'seasonality_mag': 35.0,
                'residual_std': 21.258429,
            }
        )


    def test_others(self):
        # test there is nan in feature vector because the length of TS is too short
        feature_vector = TsFeatures().transform(TSData_short)

        self.assertEqual(
            np.isnan(np.asarray(list(feature_vector.values()))).any(),
            True,
        )

    def test_errors(self):
        # test input error (time series is too short)
        self.assertRaises(
            ValueError,
            TsFeatures().transform,
            TSData_mini,
        )

    def test_IntegerArrays(self):
        df = pd.DataFrame(
            {
                'time':range(15),
                'value':[1, 4, 9, 4, 5, 5, 7, 2, 5, 1, 6, 3, 6, 5, 5]
            }
        )
        df.value = df.value.astype(dtype = pd.Int64Dtype())
        ts = TimeSeriesData(df)

        ts_features = TsFeatures(selected_features = [
            'length',
            'mean',
            'entropy',
            'hurst',
            'y_acf1',
            'seas_acf1',
            'hw_gamma',
        ])
        feats = ts_features.transform(ts)
        feats = {key: round(feats[key], 3) for key in feats}
        self.assertEqual(
            feats,
            {
                'length': 15,
                'mean': 4.533,
                'entropy': -0.765,
                'hurst': -0.143,
                'y_acf1': -0.298,
                'seas_acf1': -0.121,
                'hw_gamma': 0.947
            }
        )

    def test_transformer(self):
        # single timeseries data for testing
        metadata = {}
        line = {
            'id':'182328635174555',
            'values':'[13, 0, 1, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 1,\
             0, 0, 0, 0, 0, 1, 0, 5, 2, 0, 0, 1, 6, 0, 0, 0, 0, 0, 0, 7,\
             0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]'
        }
        # test if the default method works in transformer
        hive_transformed = next(transform(metadata, line))
        hive_transformed.pop('id')

        values = json.loads(line['values'])
        ts_obj = TimeSeriesData(pd.DataFrame(
            {'time': list(range(len(values))), 'values': values})
        )
        TsFeature_transformed = TsFeatures().transform(ts_obj)
        self.assertEqual(
            hive_transformed,
            TsFeature_transformed
        )

        # test if opt-in features works
        selected_features = [
            'trend_strength',
            'seasonality_strength',
            'diff1y_acf1',
            'holt_alpha'
        ]
        metadata = {
            'selected_features':','.join(selected_features)
        }
        hive_transformed = next(transform(metadata, line))
        hive_transformed.pop('id')

        TsFeature_transformed = TsFeatures(
            selected_features=selected_features
        ).transform(ts_obj)
        self.assertEqual(
            hive_transformed,
            TsFeature_transformed
        )

        # test if opt-in groups works
        selected_groups = [
            'special_ac',
            'stl_features',
            'statistics',
        ]
        metadata = {
            'selected_features':','.join(selected_groups)
        }
        hive_transformed = next(transform(metadata, line))
        hive_transformed.pop('id')

        TsFeature_transformed = TsFeatures(
            selected_features=selected_groups
        ).transform(ts_obj)
        self.assertEqual(
            hive_transformed,
            TsFeature_transformed
        )
