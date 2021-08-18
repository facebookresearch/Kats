# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import io
import os
import pkgutil
import re
from typing import cast, Any, Dict
from unittest import TestCase

import numpy as np
import pandas as pd
import statsmodels
from kats.consts import TimeSeriesData
from kats.tsfeatures.tsfeatures import _FEATURE_GROUP_MAPPING, TsFeatures


statsmodels_ver = float(
    re.findall("([0-9]+\\.[0-9]+)\\..*", statsmodels.__version__)[0]
)
SAMPLE_INPUT_TS_BOCPD_SCALED = pd.DataFrame(
    {
        "time": pd.date_range("2021-01-01", "2021-01-25"),
        "value": [
            -0.35010234,
            -0.40149659,
            -0.1959196,
            -0.43233314,
            -0.41177544,
            -0.44650963,
            0.0447223,
            -0.39208192,
            -0.22477185,
            -0.11754892,
            -0.45114025,
            2.31030965,
            -0.45090788,
            3.12980422,
            2.55511448,
            -0.45273205,
            -0.45397689,
            -0.44716349,
            -0.45230305,
            -0.45431129,
            -0.44282053,
            -0.44267253,
            -0.11942641,
            -0.45190004,
            -0.44805678,
        ],
    }
)


def load_data(file_name):
    ROOT = "kats"
    if "kats" in os.getcwd().lower():
        path = "data/"
    else:
        path = "kats/data/"
    data_object = pkgutil.get_data(ROOT, path + file_name)
    return pd.read_csv(io.BytesIO(data_object), encoding="utf8")


class TSfeaturesTest(TestCase):
    def setUp(self):
        DATA = load_data("air_passengers.csv")
        DATA.columns = ["time", "y"]
        self.TSData = TimeSeriesData(DATA)

        self.TSData_short = TimeSeriesData(DATA.iloc[:8, :])
        self.TSData_mini = TimeSeriesData(DATA.iloc[:2, :])

    def assertDictAlmostEqual(
        self, expected: Dict[str, Any], features: Dict[str, Any], places: int = 4
    ) -> None:
        """Compares that two dictionaries are floating-point almost equal.

        Note: the dictionaries may or may not contain floating-point values.

        Args:
          expected: the expected dictionary of values.
          actual: the actual dictionary of values.
          places: the number of decimal places for floating point comparisons.
        """

        self.assertEqual(expected.keys(), features.keys())
        for k, v in expected.items():
            if isinstance(v, float):
                if np.isnan(v):
                    self.assertTrue(np.isnan(features[k]), msg=f"{k} differ")
                else:
                    self.assertAlmostEqual(v, features[k], places=4, msg=f"{k} differ")
            else:
                self.assertEqual(v, features[k], msg=f"{k} differ")

    def test_all_feature_names_unique(self) -> None:
        features = set()
        for _, feats in _FEATURE_GROUP_MAPPING.items():
            for feat in feats:
                self.assertFalse(feat in features, f"duplicate feature name {feat}")
                features.add(feat)

    def test_tsfeatures_basic(self) -> None:
        ts = TimeSeriesData(df=SAMPLE_INPUT_TS_BOCPD_SCALED)
        features = cast(Dict[str, float], TsFeatures(hw_params=False).transform(ts))
        expected = {
            # statistics_features
            "length": 25.0,
            "mean": 0.0,
            "var": 1.0,
            "entropy": 0.8808,
            "lumpiness": 0.2423,
            "stability": 0.0148,
            "flat_spots": 1.0,
            "hurst": -1.3972,
            "std1st_der": 0.618,
            "crossing_points": 10.0,
            "binarize_mean": 0.16,
            "unitroot_kpss": 0.1567,
            "heterogeneity": 3.1459,
            "histogram_mode": -0.4543,
            "linearity": 0.0,
            # stl_features
            "trend_strength": 0.5364,
            "seasonality_strength": 0.4646,
            "spikiness": 0.0004,
            "peak": 6.0,
            "trough": 5.0,
            # level_shift_features
            "level_shift_idx": 0.0,
            "level_shift_size": 0.0046,
            # acfpacf_features
            "y_acf1": 0.2265,
            "y_acf5": 0.1597,
            "diff1y_acf1": -0.5021,
            "diff1y_acf5": 0.3465,
            "diff2y_acf1": -0.6838,
            "diff2y_acf5": 0.6092,
            "y_pacf5": 0.2144,
            "diff1y_pacf5": 0.4361,
            "diff2y_pacf5": 4.4276,
            "seas_acf1": -0.1483,
            "seas_pacf1": -0.0064,
            # special_ac
            "firstmin_ac": 4.0,
            "firstzero_ac": 4.0,
            # holt_params
            "holt_alpha": 0.0,
            "holt_beta": 0.0
            # hw_params
            # cusum_detector
            # robust_stat_detector
            # bocp_detector
            # outlier_detector
            # trend_detector
            # nowcasting
            # seasonalities
        }
        if statsmodels_ver >= 0.12:
            expected["trend_strength"] = 0.426899
            expected["seasonality_strength"] = 0.410921
            expected["spikiness"] = 0.000661
            expected["holt_alpha"] = 0.289034
        self.assertDictAlmostEqual(expected, features)

    def test_tsfeatures(self) -> None:
        feature_vector = TsFeatures().transform(self.TSData)

        feature_vector_round = {
            # pyre-fixme[6]: Expected `str` for 1st param but got `Union[Dict[str,
            #  float], str]`.
            key: round(feature_vector[key], 6)
            for key in feature_vector
        }

        # test there is no nan in feature vector
        self.assertEqual(
            # pyre-fixme[16]: `List` has no attribute `values`.
            np.isnan(np.asarray(list(feature_vector.values()))).any(),
            False,
        )

        # test there are 40 features in the feature vector now
        self.assertEqual(
            len(np.asarray(list(feature_vector.values()))) == 40,
            True,
        )

        # test feature vector value
        rounded_truth = {
            "length": 144,
            "mean": 280.298611,
            "var": 14291.973331,
            "entropy": 0.428737,
            "lumpiness": 3041164.562906,
            "stability": 12303.627267,
            "flat_spots": 2,
            "hurst": -0.080233,
            "std1st_der": 27.206288,
            "crossing_points": 7,
            "binarize_mean": 0.444444,
            "unitroot_kpss": 0.128475,
            "heterogeneity": 126.064506,
            "histogram_mode": 155.8,
            "linearity": 0.853638,
            "trend_strength": 0.9681,
            "seasonality_strength": 0.440863,
            "spikiness": 33.502886,
            "peak": 6,
            "trough": 3,
            "level_shift_idx": 118,
            "level_shift_size": 15.6,
            "y_acf1": 0.948047,
            "y_acf5": 3.392072,
            "diff1y_acf1": 0.302855,
            "diff1y_acf5": 0.259459,
            "diff2y_acf1": -0.191006,
            "diff2y_acf5": 0.134207,
            "y_pacf5": 1.003288,
            "diff1y_pacf5": 0.219412,
            "diff2y_pacf5": 0.26101,
            "seas_acf1": 0.662904,
            "seas_pacf1": 0.15617,
            "firstmin_ac": 8,
            "firstzero_ac": 52,
            "holt_alpha": 1.0,
            "holt_beta": 0.0,
            "hw_alpha": 0.842106,
            "hw_beta": 0.052631,
            "hw_gamma": 0.157901,
        }
        if statsmodels_ver >= 0.12:
            rounded_truth["trend_strength"] = 0.93833
            rounded_truth["seasonality_strength"] = 0.329934
            rounded_truth["spikiness"] = 111.697325
            feature_vector_round["holt_alpha"] = np.round(
                feature_vector_round["holt_alpha"], 1
            )
            feature_vector_round["holt_beta"] = np.round(
                feature_vector_round["holt_beta"], 1
            )
            rounded_truth["holt_alpha"] = 1.0
            rounded_truth["holt_beta"] = 0.0
            rounded_truth["hw_alpha"] = 1.0
            rounded_truth["hw_beta"] = 0.0
            rounded_truth["hw_gamma"] = 0.0
        self.assertEqual(feature_vector_round, rounded_truth)

    def test_feature_selections(self) -> None:
        # test disabling functions
        feature_vector = cast(
            Dict[str, Any],
            TsFeatures(
                unitroot_kpss=False,
                histogram_mode=False,
                diff2y_pacf5=False,
                firstmin_ac=False,
            ).transform(self.TSData),
        )
        rounded_truth = {
            "length": 144,
            "mean": 280.298611,
            "var": 14291.973331,
            "entropy": 0.428737,
            "lumpiness": 3041164.562906,
            "stability": 12303.627267,
            "flat_spots": 2,
            "hurst": -0.080233,
            "std1st_der": 27.206288,
            "crossing_points": 7,
            "binarize_mean": 0.444444,
            "heterogeneity": 126.064506,
            "linearity": 0.853638,
            "trend_strength": 0.9681,
            "seasonality_strength": 0.440863,
            "spikiness": 33.502886,
            "peak": 6,
            "trough": 3,
            "level_shift_idx": 118,
            "level_shift_size": 15.6,
            "y_acf1": 0.948047,
            "y_acf5": 3.392072,
            "diff1y_acf1": 0.302855,
            "diff1y_acf5": 0.259459,
            "diff2y_acf1": -0.191006,
            "diff2y_acf5": 0.134207,
            "y_pacf5": 1.003288,
            "diff1y_pacf5": 0.219412,
            "seas_acf1": 0.662904,
            "seas_pacf1": 0.15617,
            "firstzero_ac": 52,
            "holt_alpha": 1.0,
            "holt_beta": 0.0,
            "hw_alpha": 0.842106,
            "hw_beta": 0.052631,
            "hw_gamma": 0.157901,
        }
        if statsmodels_ver >= 0.12:
            rounded_truth["trend_strength"] = 0.93833
            rounded_truth["seasonality_strength"] = 0.329934
            rounded_truth["spikiness"] = 111.697325
            feature_vector["holt_alpha"] = np.round(feature_vector["holt_alpha"], 1)
            feature_vector["holt_beta"] = np.round(feature_vector["holt_beta"], 1)
            rounded_truth["holt_alpha"] = 1.0
            rounded_truth["holt_beta"] = 0.0
            rounded_truth["hw_alpha"] = 1.0
            rounded_truth["hw_beta"] = 0.0
            rounded_truth["hw_gamma"] = 0.0
        self.assertDictAlmostEqual(rounded_truth, feature_vector, places=6)

        # test selecting features
        features = [
            "var",
            "linearity",
            "spikiness",
            "trough",
            "holt_alpha",
            "hw_gamma",
            "level_shift_idx",
        ]
        feature_vector = cast(
            Dict[str, Any],
            TsFeatures(selected_features=features).transform(self.TSData),
        )

        # test feature vector value
        rounded_truth = {
            "var": 14291.973331,
            "linearity": 0.853638,
            "spikiness": 33.502886,
            "trough": 3,
            "level_shift_idx": 118,
            "holt_alpha": 1.0,
            "hw_gamma": 0.157901,
        }
        if statsmodels_ver >= 0.12:
            rounded_truth["spikiness"] = 111.697325
            feature_vector["holt_alpha"] = np.round(feature_vector["holt_alpha"], 1)
            rounded_truth["holt_alpha"] = 1.0
            rounded_truth["hw_gamma"] = 0.0

        self.assertDictAlmostEqual(rounded_truth, feature_vector, places=6)

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
            "nowcast_roc",
            "nowcast_ma",
            "nowcast_mom",
            "nowcast_lag",
            "nowcast_macd",
            "nowcast_macdsign",
            "nowcast_macddiff",
            "seasonal_period",
            "seasonality_mag",
            "trend_mag",
            "residual_std",
        ]
        feature_vector = cast(
            Dict[str, Any],
            TsFeatures(selected_features=extension_features).transform(self.TSData),
        )

        # test feature vector value
        rounded_truth = {
            "cusum_num": 1,
            "cusum_conf": 1.0,
            "cusum_cp_index": 0.527778,
            "cusum_delta": 199.098856,
            "cusum_llr": 168.663483,
            "cusum_regression_detected": 1,
            "cusum_stable_changepoint": 1,
            "cusum_p_value": 0.0,
            "robust_num": 3,
            "robust_metric_mean": -31.866667,
            "bocp_num": 3,
            "bocp_conf_max": 0.677218,
            "bocp_conf_mean": 0.587680,
            "outlier_num": 0,
            "trend_num": 2,
            "trend_num_increasing": 0,
            "trend_avg_abs_tau": 0.821053,
            "nowcast_roc": 0.062858,
            "nowcast_ma": 280.417143,
            "nowcast_mom": 12.841727,
            "nowcast_lag": 273.136691,
            "nowcast_macd": 11.032608,
            "nowcast_macdsign": 10.985509,
            "nowcast_macddiff": 0.527714,
            "seasonal_period": 7,
            "trend_mag": 2.404464,
            "seasonality_mag": 35.0,
            "residual_std": 21.258429,
        }
        if statsmodels_ver >= 0.12:
            rounded_truth["trend_mag"] = 2.318814
            rounded_truth["seasonality_mag"] = 36.0
            rounded_truth["residual_std"] = 29.630087
        self.assertDictAlmostEqual(rounded_truth, feature_vector, places=6)

    def test_others(self) -> None:
        # test there is nan in feature vector because the length of TS is too short
        feature_vector = TsFeatures().transform(self.TSData_short)

        self.assertEqual(
            # pyre-fixme[16]: `List` has no attribute `values`.
            np.isnan(np.asarray(list(feature_vector.values()))).any(),
            True,
        )

    def test_errors(self) -> None:
        # test input error (time series is too short)
        self.assertRaises(
            ValueError,
            TsFeatures().transform,
            self.TSData_mini,
        )
        with self.assertRaises(ValueError):
            TsFeatures(selected_features=["mango"])

    def test_IntegerArrays(self) -> None:
        if statsmodels_ver < 0.12:
            df = pd.DataFrame(
                {
                    "time": range(15),
                    "value": [1, 4, 9, 4, 5, 5, 7, 2, 5, 1, 6, 3, 6, 5, 5],
                }
            )
        elif statsmodels_ver >= 0.12:
            df = pd.DataFrame(
                {
                    "time": range(20),
                    "value": [
                        1,
                        4,
                        9,
                        4,
                        5,
                        5,
                        7,
                        2,
                        5,
                        1,
                        6,
                        3,
                        6,
                        5,
                        5,
                        6,
                        9,
                        10,
                        5,
                        6,
                    ],
                }
            )
        df["value"] = df["value"].astype(dtype=pd.Int64Dtype())
        # pyre-fixme[61]: `df` may not be initialized here.
        ts = TimeSeriesData(df)

        ts_features = TsFeatures(
            selected_features=[
                "length",
                "mean",
                "entropy",
                "hurst",
                "y_acf1",
                "seas_acf1",
                "hw_gamma",
            ]
        )
        feats = ts_features.transform(ts)
        # pyre-fixme[6]: Expected `str` for 1st param but got
        #  `Union[typing.Dict[str, float], str]`.
        feats = {key: round(feats[key], 3) for key in feats}
        if statsmodels_ver < 0.12:
            self.assertEqual(
                feats,
                {
                    "length": 15,
                    "mean": 4.533,
                    "entropy": 0.765,
                    "hurst": -0.143,
                    "y_acf1": -0.298,
                    "seas_acf1": -0.121,
                    "hw_gamma": 0.947,
                },
            )
        elif statsmodels_ver >= 0.12:
            self.assertEqual(
                feats,
                {
                    "length": 20,
                    "mean": 5.2,
                    "entropy": 0.894,
                    "hurst": -0.12,
                    "y_acf1": 0.041,
                    "seas_acf1": -0.125,
                    "hw_gamma": 0.0,
                },
            )

    def test_nowcasting_error(self) -> None:
        ts = TimeSeriesData(df=SAMPLE_INPUT_TS_BOCPD_SCALED)
        features = cast(Dict[str, float], TsFeatures(nowcasting=True).transform(ts))
        expected = {
            "trend_strength": 0.536395,
            "seasonality_strength": 0.464575,
            "spikiness": 0.000353,
            "peak": 6,
            "trough": 5,
            "level_shift_idx": 0,
            "level_shift_size": 0.004636,
            "y_acf1": 0.226546,
            "y_acf5": 0.159668,
            "diff1y_acf1": -0.502100,
            "diff1y_acf5": 0.346528,
            "diff2y_acf1": -0.683816,
            "diff2y_acf5": 0.609249,
            "y_pacf5": 0.214401,
            "diff1y_pacf5": 0.436150,
            "diff2y_pacf5": 4.427552,
            "seas_acf1": -0.148278,
            "seas_pacf1": -0.006386,
            "firstmin_ac": 4,
            "firstzero_ac": 4,
            "holt_alpha": 1.014757e-09,
            "holt_beta": 0.0,
            "hw_alpha": np.nan,
            "hw_beta": np.nan,
            "hw_gamma": np.nan,
            "length": 25,
            "mean": 1.200000e-09,
            "var": 1.0,
            "entropy": 0.880823,
            "lumpiness": 0.242269,
            "stability": 0.014825,
            "flat_spots": 1,
            "hurst": -1.397158,
            "std1st_der": 0.618019,
            "crossing_points": 10,
            "binarize_mean": 0.16,
            "unitroot_kpss": 0.156730,
            "heterogeneity": 3.145863,
            "histogram_mode": -0.454311,
            "linearity": 2.607152e-06,
            "nowcast_roc": np.nan,
            "nowcast_mom": np.nan,
            "nowcast_ma": np.nan,
            "nowcast_lag": np.nan,
            "nowcast_macd": np.nan,
            "nowcast_macdsign": np.nan,
            "nowcast_macddiff": np.nan,
        }
        if statsmodels_ver >= 0.12:
            expected["trend_strength"] = 0.426899
            expected["seasonality_strength"] = 0.410921
            expected["spikiness"] = 0.000661
            expected["holt_alpha"] = 0.289034
        self.assertDictAlmostEqual(expected, features)
