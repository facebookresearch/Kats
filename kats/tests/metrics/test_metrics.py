# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union
from unittest import TestCase

import numpy as np
import numpy.testing as npt
from kats.metrics import metrics
from parameterized.parameterized import parameterized


class MetricsTest(TestCase):
    def validate(
        self, expected: Union[float, np.ndarray], result: Union[float, np.ndarray]
    ) -> None:
        if isinstance(expected, float):
            self.assertTrue(isinstance(result, float))
            if np.isnan(expected):
                self.assertTrue(np.isnan(result), f"{result} is not nan")
            else:
                self.assertAlmostEqual(expected, result)
        else:
            self.assertTrue(isinstance(result, np.ndarray))
            npt.assert_array_almost_equal(np.array(expected), result)

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator...
    @parameterized.expand(
        [
            ("normal", [[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]),
            ("missing", [[1, 2], [2, 3]], [[1, 2], None, [2, 3]]),
            # TODO: nans
            ("empty", [], []),
        ]
    )
    def test__arrays(
        self, _name: str, expected: List[List[float]], arrs: List[List[float]]
    ) -> None:
        result = list(metrics._arrays(*arrs))
        self.assertEqual(len(expected), len(result))
        for e, a in zip(expected, result):
            npt.assert_almost_equal(np.array(e), a)

    def test__arrays_mismatched(self) -> None:
        with self.assertRaises(ValueError):
            _ = list(metrics._arrays([1], [2, 3]))

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator...
    @parameterized.expand(
        [
            ("normal", [0.25, 0.5, 2.0], [1, 2, 4], [4, 4, 2]),
            # TODO: more numerical combinations
        ]
    )
    def test__safe_divide(
        self,
        _name: str,
        exp: List[float],
        num: List[float],
        denom: Union[float, List[float]],
        negative_infinity: float = -1.0,
        positive_infinity: float = 1.0,
        indeterminate: float = 0.0,
        nan: float = np.nan,
    ) -> None:
        expected = np.array(exp)
        numerator = np.array(num)
        denominator = np.array(denom)
        npt.assert_almost_equal(
            expected,
            metrics._safe_divide(
                numerator,
                denominator,
                negative_infinity,
                positive_infinity,
                indeterminate,
                nan,
            ),
        )

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator...
    @parameterized.expand(
        [
            ("normal", [-3, -2, 2], [1, 2, 4], [4, 4, 2]),
            ("weighted", [-1.25, -0.125, 0.375], [1, 2, 3], [3.5, 2.5, 1.5], [2, 1, 1]),
            # TODO: more numerical combinations
            ("empty", [], [], []),
        ]
    )
    def test_error(
        self,
        _name: str,
        expected: List[float],
        y_true: List[float],
        y_pred: List[float],
        weights: Optional[List[float]] = None,
    ) -> None:
        result = metrics.error(y_true, y_pred, weights)
        self.validate(np.array(expected), result)

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator...
    @parameterized.expand(
        [
            ("normal", [3, 2, 2], [1, 2, 4], [4, 4, 2]),
            ("weighted", [1.25, 0.125, 0.375], [1, 2, 3], [3.5, 2.5, 1.5], [2, 1, 1]),
            # TODO: more numerical combinations
            ("empty", [], [], []),
        ]
    )
    def test_absolute_error(
        self,
        _name: str,
        expected: List[float],
        y_true: List[float],
        y_pred: List[float],
        weights: Optional[List[float]] = None,
    ) -> None:
        result = metrics.absolute_error(y_true, y_pred, weights)
        self.validate(np.array(expected), result)

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator...
    @parameterized.expand(
        [
            ("normal", [-3, -1, 0.5], [1, 2, 4], [4, 4, 2]),
            (
                "weighted",
                [-1.25, -0.0625, 0.125],
                [1, 2, 3],
                [3.5, 2.5, 1.5],
                [2, 1, 1],
            ),
            # TODO: more numerical combinations
            ("empty", [], [], []),
        ]
    )
    def test_percentage_error(
        self,
        _name: str,
        expected: List[float],
        y_true: List[float],
        y_pred: List[float],
        weights: Optional[List[float]] = None,
    ) -> None:
        result = metrics.percentage_error(y_true, y_pred, weights)
        self.validate(np.array(expected), result)

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator...
    @parameterized.expand(
        [
            ("normal", [3, 1, 0.5], [1, 2, 4], [4, 4, 2]),
            ("weighted", [1.25, 0.0625, 0.125], [1, 2, 3], [3.5, 2.5, 1.5], [2, 1, 1]),
            # TODO: more numerical combinations
            ("empty", [], [], []),
        ]
    )
    def test_absolute_percentage_error(
        self,
        _name: str,
        expected: List[float],
        y_true: List[float],
        y_pred: List[float],
        weights: Optional[List[float]] = None,
    ) -> None:
        result = metrics.absolute_percentage_error(y_true, y_pred, weights)
        self.validate(np.array(expected), result)

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator...
    @parameterized.expand(
        [
            ("normal", 2 / 9, [1, 2, 4], [4, 4, 2]),
            # TODO: more numerical combinations
            ("all_pred_missing", np.nan, [1.0, 2.0], [np.nan, np.nan]),
            ("all_true_missing", 1.0, [np.nan, np.nan], [1.0, 2.0]),
            ("empty", np.nan, [], []),
        ]
    )
    def test_continuous_rank_probability_score(
        self,
        _name: str,
        expected: float,
        y_true: List[float],
        y_pred: List[float],
    ) -> None:
        result = metrics.crps(y_true, y_pred)
        self.validate(expected, result)

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator...
    @parameterized.expand(
        [
            ("normal", 0.5, [2, 3, 4, -5], [1, 1, 1, 1], 3.0),
            ("empty", np.nan, [], [], 1.0),
        ]
    )
    def test_frequency_exceeds_relative_threshold(
        self,
        _name: str,
        expected: float,
        y_true: List[float],
        y_pred: List[float],
        threshold: float,
    ) -> None:
        result = metrics.frequency_exceeds_relative_threshold(y_true, y_pred, threshold)
        self.validate(expected, result)

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator...
    @parameterized.expand(
        [
            ("normal", 4 / 9, [1, 2, 4], [4, 4, 2]),
            # TODO: more numerical combinations
            ("all_pred_missing", np.nan, [1.0, 2.0], [np.nan, np.nan]),
            ("all_true_missing", 1.0, [np.nan, np.nan], [1.0, 2.0]),
            ("empty", np.nan, [], []),
        ]
    )
    def test_linear_error_in_probability_space(
        self,
        _name: str,
        expected: float,
        y_true: List[float],
        y_pred: List[float],
    ) -> None:
        result = metrics.leps(y_true, y_pred)
        self.validate(expected, result)

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator...
    @parameterized.expand(
        [
            ("normal", 1.5, [1, 2, 3], [3.5, 2.5, 1.5]),
            # TODO: more numerical combinations
            ("empty", np.nan, [], []),
        ]
    )
    def test_median_absolute_error(
        self,
        _name: str,
        expected: float,
        y_true: List[float],
        y_pred: List[float],
    ) -> None:
        result = metrics.mdae(y_true, y_pred)
        self.validate(expected, result)

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator...
    @parameterized.expand(
        [
            ("normal", 0.5, [1, 2, 3], [3.5, 2.5, 1.5]),
            # TODO: more numerical combinations
            ("empty", np.nan, [], []),
        ]
    )
    def test_median_absolute_percent_error(
        self,
        _name: str,
        expected: float,
        y_true: List[float],
        y_pred: List[float],
    ) -> None:
        result = metrics.mdape(y_true, y_pred)
        self.validate(expected, result)

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator...
    @parameterized.expand(
        [
            ("normal", 1.5, [1, 2, 3], [3.5, 2.5, 1.5]),
            ("weighted", 1.75, [1, 2, 3], [3.5, 2.5, 1.5], [2, 1, 1]),
            ("normal_2d", 2.25, [[1, 2, 3], [2, 4, 6]], [[3.5, 2.5, 1.5], [1, 1, 1]]),
            (
                "normal_2d_raw",
                [1.75, 1.75, 3.25],
                [[1, 2, 3], [2, 4, 6]],
                [[3.5, 2.5, 1.5], [1, 1, 1]],
                None,
                "raw_values",
            ),
            (
                "weighted_2d",
                2.458333333333,
                [[1, 2, 3], [2, 4, 6]],
                [[3.5, 2.5, 1.5], [1, 1, 1]],
                [[1, 1, 1], [1, 3, 1]],
            ),
            (
                "weighted_2d_raw",
                [1.75, 2.375, 3.25],
                [[1, 2, 3], [2, 4, 6]],
                [[3.5, 2.5, 1.5], [1, 1, 1]],
                [[1, 1, 1], [1, 3, 1]],
                "raw_values",
            ),
            # TODO: more numerical combinations
            ("empty", np.nan, [], []),
        ]
    )
    def test_mean_absolute_error(
        self,
        _name: str,
        expected: float,
        y_true: List[float],
        y_pred: List[float],
        weights: Optional[List[float]] = None,
        multioutput: Union[str, List[float]] = "uniform_average",
    ) -> None:
        result = metrics.mae(y_true, y_pred, weights, multioutput)
        self.validate(expected, result)

    def test_invalid_mean_absolute_error_multioutput(self) -> None:
        with self.assertRaises(ValueError):
            _ = metrics.mae([], [], [], "mango")

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator...
    @parameterized.expand(
        [
            ("normal", 2.125/2, [-2, 2, 3, 4], [0.5, 2.5, 1.5, 0]),
            # TODO: more numerical combinations
            ("zero_baseline", np.nan, [1, 1, 1, 1], [3.5, 2.5, 1.5, 0]),
            ("empty", np.nan, [], []),
        ]
    )
    def test_mean_absolute_scaled_error(
        self,
        _name: str,
        expected: float,
        y_true: List[float],
        y_pred: List[float],
    ) -> None:
        result = metrics.mase(y_true, y_pred)
        self.validate(expected, result)

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator...
    @parameterized.expand(
        [
            ("normal", 1.0625, [1, 2, 3, 4], [3.5, 2.5, 1.5, 0]),
            # TODO: more numerical combinations
            ("empty", np.nan, [], []),
        ]
    )
    def test_mean_absolute_percent_error(
        self,
        _name: str,
        expected: float,
        y_true: List[float],
        y_pred: List[float],
    ) -> None:
        result = metrics.mape(y_true, y_pred)
        self.validate(expected, result)

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator...
    @parameterized.expand(
        [
            ("normal", 0.625, [1, 2, 3, 4], [3.5, 2.5, 1.5, 0]),
            # TODO: more numerical combinations
            ("empty", np.nan, [], []),
        ]
    )
    def test_mean_error(
        self,
        _name: str,
        expected: float,
        y_true: List[float],
        y_pred: List[float],
    ) -> None:
        result = metrics.me(y_true, y_pred)
        self.validate(expected, result)

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator...
    @parameterized.expand(
        [
            ("normal", -0.3125, [1, 2, 3, 4], [3.5, 2.5, 1.5, 0]),
            # TODO: more numerical combinations
            ("empty", np.nan, [], []),
        ]
    )
    def test_mean_percentage_error(
        self,
        _name: str,
        expected: float,
        y_true: List[float],
        y_pred: List[float],
    ) -> None:
        result = metrics.mpe(y_true, y_pred)
        self.validate(expected, result)

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator...
    @parameterized.expand(
        [
            ("normal", 35 / 12, [1, 2, 3], [3.5, 2.5, 1.5]),
            ("weighted", 2.75, [1, 2, 3], [3.5, 2.5, 1.5], [1, 1, 2]),
            # TODO: more numerical combinations
            ("empty", np.nan, [], []),
        ]
    )
    def test_mean_squared_error(
        self,
        _name: str,
        expected: float,
        y_true: List[float],
        y_pred: List[float],
        weights: Optional[List[float]] = None,
    ) -> None:
        result = metrics.mse(y_true, y_pred, weights)
        self.validate(expected, result)

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator...
    @parameterized.expand(
        [
            ("normal", np.sqrt(35 / 12), [1, 2, 3], [3.5, 2.5, 1.5]),
            ("weighted", np.sqrt(2.75), [1, 2, 3], [3.5, 2.5, 1.5], [1, 1, 2]),
            # TODO: more numerical combinations
            ("empty", np.nan, [], []),
        ]
    )
    def test_root_mean_squared_error(
        self,
        _name: str,
        expected: float,
        y_true: List[float],
        y_pred: List[float],
        weights: Optional[List[float]] = None,
    ) -> None:
        result = metrics.rmse(y_true, y_pred, weights)
        self.validate(expected, result)

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator...
    @parameterized.expand(
        [
            ("normal", 0.5484140, [1, 2, 3], [3.5, 2.5, 1.5]),
            ("weighted", 0.5299002, [1, 2, 3], [3.5, 2.5, 1.5], [1, 1, 2]),
            # TODO: more numerical combinations
            ("empty", np.nan, [], []),
        ]
    )
    def test_root_mean_squared_log_error(
        self,
        _name: str,
        expected: float,
        y_true: List[float],
        y_pred: List[float],
        weights: Optional[List[float]] = None,
    ) -> None:
        result = metrics.rmsle(y_true, y_pred, weights)
        self.validate(expected, result)

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator...
    @parameterized.expand(
        [
            ("normal", 1.4790199, [1, 2, 3], [3.5, 2.5, 1.5]),
            ("weighted", 0.3903124, [1, 2, 3], [3.5, 2.5, 1.5], [1, 1, 2]),
            # TODO: more numerical combinations
            ("empty", np.nan, [], []),
        ]
    )
    def test_root_mean_squared_percentage_error(
        self,
        _name: str,
        expected: float,
        y_true: List[float],
        y_pred: List[float],
        weights: Optional[List[float]] = None,
    ) -> None:
        result = metrics.rmspe(y_true, y_pred, weights)
        self.validate(expected, result)

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator...
    @parameterized.expand(
        [
            ("normal", 1 / 3, [1, 2, 3], [3.5, 2.5, 1.5]),
            # TODO: more numerical combinations
            ("empty", np.nan, [], []),
        ]
    )
    def test_scaled_symmetric_mean_absolute_percentage_error(
        self,
        _name: str,
        expected: float,
        y_true: List[float],
        y_pred: List[float],
    ) -> None:
        result = metrics.scaled_smape(y_true, y_pred)
        self.validate(expected, result)

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator...
    @parameterized.expand(
        [
            ("normal", 2 / 9, [1, 2, 3], [3.5, 2.5, 1.5]),
            ("negatives", 2 / 27, [-1, 2, 3], [3.5, 2.5, -1.5]),
            # TODO: more numerical combinations
            ("empty", np.nan, [], []),
        ]
    )
    def test_symmetric_bias(
        self,
        _name: str,
        expected: float,
        y_true: List[float],
        y_pred: List[float],
    ) -> None:
        result = metrics.sbias(y_true, y_pred)
        self.validate(expected, result)

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator...
    @parameterized.expand(
        [
            ("normal", 2 / 3, [1, 2, 3], [3.5, 2.5, 1.5]),
            # TODO: more numerical combinations
            ("empty", np.nan, [], []),
        ]
    )
    def test_symmetric_mean_absolute_percentage_error(
        self,
        _name: str,
        expected: float,
        y_true: List[float],
        y_pred: List[float],
    ) -> None:
        result = metrics.smape(y_true, y_pred)
        self.validate(expected, result)

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator...
    @parameterized.expand(
        [
            ("normal", -1.0, [1, 2, 3], [3.5, 2.5, 1.5]),
            # TODO: more numerical combinations
            ("empty", np.nan, [], []),
        ]
    )
    def test_tracking_signal(
        self,
        _name: str,
        expected: float,
        y_true: List[float],
        y_pred: List[float],
    ) -> None:
        result = metrics.tracking_signal(y_true, y_pred)
        self.validate(expected, result)

    def test_metric(self) -> None:
        self.assertEqual(metrics.mae, metrics.metric("mae"))
        with self.assertRaises(ValueError):
            _ = metrics.metric("mango")

    def test_core_metric(self) -> None:
        self.assertEqual(metrics.mae, metrics.core_metric("mae"))
        with self.assertRaises(ValueError):
            _ = metrics.metric("mango")
