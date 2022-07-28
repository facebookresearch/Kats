# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest import TestCase
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import pandas as pd
from kats.consts import TimeSeriesData
from kats.data.utils import load_data
from kats.models.var import VARModel, VARParams
from parameterized.parameterized import parameterized


TEST_DATA = {
    "multivariate": {
        "ts": TimeSeriesData(load_data("multivariate_anomaly_simulated_data.csv")),
    },
    "multivariate_2": {
        "ts": TimeSeriesData(load_data("multi_ts.csv")),
    },
}


class testVARModel(TestCase):
    def test_fit_forecast(self, steps: int = 5) -> None:
        ts = TEST_DATA["multivariate_2"]["ts"]
        params = VARParams()
        train, truth = ts[:-steps], ts[-steps:]
        m = VARModel(train, params)
        m.fit()
        pred = m.predict(steps=steps)

        # check whether the time indices of each forecasted feature are the same
        index = [v.to_dataframe().time for _, v in pred.items()]
        self.assertTrue(all(x.equals(index[0]) for x in index))

        # check whether the values are close and shapes are correct
        truth = truth.to_dataframe().iloc[:, 1:]
        pred_forecast = pd.concat(
            [v["fcst"].to_dataframe().iloc[:, 1:2] for _, v in pred.items()], axis=1
        )
        pred_forecast.columns = truth.columns
        self.assertTrue(truth.subtract(pred_forecast).values.max() < 5)

    def test_invalid_params(self) -> None:
        params = VARParams()
        input_data = TimeSeriesData(pd.DataFrame())
        with self.assertRaises(ValueError):
            m = VARModel(input_data, params)
            m.fit()
            m.predict(steps=30, include_history=True)

    # pyre-fixme[56]
    @parameterized.expand([[TEST_DATA["multivariate"]["ts"]]])
    @patch("pandas.concat")
    def test_predict_exception(self, ts: TimeSeriesData, mock_obj: MagicMock) -> None:
        mock_obj.side_effect = Exception
        with self.assertRaisesRegex(
            Exception, "^Failed to generate in-sample forecasts for historical data"
        ):
            params = VARParams()
            m = VARModel(ts, params)
            m.fit()
            m.predict(steps=30, include_history=True)

    # pyre-fixme[56]
    @parameterized.expand(
        [
            [TEST_DATA["multivariate"]["ts"]],
            [TEST_DATA["multivariate_2"]["ts"]],
        ]
    )
    def test_predict_unfit(self, ts: TimeSeriesData) -> None:
        with self.assertRaises(ValueError):
            m = VARModel(ts, VARParams())
            m.predict(steps=30)

    # pyre-fixme[56]
    @parameterized.expand(
        [
            [TEST_DATA["multivariate"]["ts"]],
            [TEST_DATA["multivariate_2"]["ts"]],
        ]
    )
    def test_search_space(self, ts: TimeSeriesData) -> None:
        params = VARParams()
        params.validate_params()
        with self.assertRaises(NotImplementedError):
            VARModel.get_parameter_search_space()

    # @pytest.mark.image_compare
    # pyre-fixme[56]
    @parameterized.expand([[TEST_DATA["multivariate_2"]["ts"]]])
    def test_plot(self, ts: TimeSeriesData) -> plt.Figure:
        # Test the example from the 201 notebook.
        m = VARModel(ts, VARParams())
        m.fit()
        m.predict(steps=90)
        m.plot()
        return plt.gcf()

    # @pytest.mark.image_compare
    # pyre-fixme[56]
    @parameterized.expand([[TEST_DATA["multivariate_2"]["ts"]]])
    def test_plot_include_history(self, ts: TimeSeriesData) -> plt.Figure:
        # This shouldn't error, but currently does.
        with self.assertRaises(ValueError):
            m = VARModel(ts, VARParams())
            m.fit()
            m.predict(steps=90, include_history=True)
            m.plot()
            return plt.gcf()

    # pyre-fixme[56]
    @parameterized.expand([[TEST_DATA["multivariate"]["ts"]]])
    def test_plot_ax_not_supported(self, ts: TimeSeriesData) -> None:
        with self.assertRaises(ValueError):
            _, ax = plt.subplots()
            m = VARModel(ts, VARParams())
            m.fit()
            m.predict(steps=5)
            m.plot(ax=ax)

    # pyre-fixme[56]
    @parameterized.expand([[TEST_DATA["multivariate"]["ts"]]])
    def test_plot_unpredict(self, ts: TimeSeriesData) -> None:
        with self.assertRaises(ValueError):
            m = VARModel(ts, VARParams())
            m.plot()

    # pyre-fixme[56]
    @parameterized.expand([[TEST_DATA["multivariate"]["ts"]]])
    def test_str(self, ts: TimeSeriesData) -> None:
        result = str(VARModel(ts, VARParams()))
        self.assertEqual("VAR", result)


if __name__ == "__main__":
    unittest.main()
