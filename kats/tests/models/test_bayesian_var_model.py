# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest import TestCase

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import seaborn as sns
from kats.compat.pandas import assert_frame_equal
from kats.consts import TimeSeriesData
from kats.data.utils import load_data
from kats.models.bayesian_var import BayesianVAR, BayesianVARParams


class testBayesianVARModel(TestCase):
    def setUp(self) -> None:
        DATA_multi = load_data("multivariate_anomaly_simulated_data.csv")
        self.TSData_multi = TimeSeriesData(DATA_multi)
        self.params = BayesianVARParams()

    def test_univariate_data(self) -> None:
        univ_df = load_data("air_passengers.csv")
        univ_ts = TimeSeriesData(df=univ_df, time_col_name="ds")
        with self.assertRaises(ValueError):
            _ = BayesianVAR(univ_ts, self.params)

    def test_diff_time_name(self) -> None:
        # Create multivariate time series without 'time' as time column name
        df = load_data("air_passengers.csv")
        df["y_2"] = df.y * 2

        # Create correct DataFrame columns that model should produce
        correct_columns = ["ds", "y", "y_2"]

        # Initialize model and test
        ts = TimeSeriesData(df=df, time_col_name="ds")
        m = BayesianVAR(ts, self.params)
        self.assertEqual(list(m.data.to_dataframe().columns), correct_columns)

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    #  `pytest.mark.mpl_image_compare($parameter$remove_text = True)`.
    @pytest.mark.mpl_image_compare(remove_text=True)
    def test_plot(self) -> plt.Figure:
        params = BayesianVARParams(p=3)
        m = BayesianVAR(self.TSData_multi, params)
        m.fit()
        m.predict(steps=30, include_history=True)
        with sns.color_palette(n_colors=8):
            ax = m.plot()
            self.assertIsNotNone(ax)
        return plt.gcf()

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    #  `pytest.mark.mpl_image_compare($parameter$remove_text = True)`.
    @pytest.mark.mpl_image_compare(remove_text=True)
    def test_plot_ax(self) -> plt.Figure:
        params = BayesianVARParams(p=3)
        m = BayesianVAR(self.TSData_multi, params)
        m.fit()
        m.predict(steps=30, include_history=True)
        with sns.color_palette(n_colors=8):
            _, ax = plt.subplots(figsize=(5, 4))
            ax = m.plot(ax=ax)
            self.assertIsNotNone(ax)
        return plt.gcf()

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    #  `pytest.mark.mpl_image_compare($parameter$remove_text = True)`.
    @pytest.mark.mpl_image_compare(remove_text=True)
    def test_plot_params(self) -> plt.Figure:
        params = BayesianVARParams(p=3)
        m = BayesianVAR(self.TSData_multi, params)
        m.fit()
        m.predict(steps=30, include_history=True)
        with sns.color_palette(n_colors=8):
            ax = m.plot(figsize=(8, 5), title="Test", ls=".")
            self.assertIsNotNone(ax)
        return plt.gcf()

    def test_predict_error(self) -> None:
        m = BayesianVAR(self.TSData_multi, self.params)
        with self.assertRaises(ValueError):
            # Model needs to be fit before predict
            m.predict(10)

    def test_predict_zero(self) -> None:
        m = BayesianVAR(self.TSData_multi, self.params)
        with self.assertRaises(ValueError):
            m.predict(0)

    def test_plot_error(self) -> None:
        m = BayesianVAR(self.TSData_multi, self.params)
        with self.assertRaises(ValueError):
            # Must fit() and predict() before plot()
            m.plot()

    def test_sigma_u(self) -> None:
        # Also test verbose predict for code coverage.
        params = BayesianVARParams(p=3)
        m = BayesianVAR(self.TSData_multi, params)
        m.fit()
        m.predict(steps=30, include_history=True, verbose=True)
        self.assertEqual(3, m.k_ar)
        # fmt: off
        expected = pd.DataFrame([
        [ 4.43026366e-05,  2.05347685e-05,  1.81942688e-04,
          4.50075234e-05,  1.86048651e-05,  2.05304755e-06,
          3.55016677e-06, -9.27833097e-05,  4.99988031e-07],
        [ 2.05347685e-05,  2.65322543e-05,  3.16169643e-05,
          7.34176226e-06,  2.38786201e-05,  8.43863345e-06,
          6.53970495e-06, -8.30124023e-05, -2.33583196e-08],
        [ 1.81942688e-04,  3.16169643e-05,  1.33754111e-03,
          5.49729738e-05,  3.16310173e-05,  1.27374356e-06,
         -1.59582259e-05, -3.71395599e-04,  1.61197161e-06],
        [ 4.50075234e-05,  7.34176226e-06,  5.49729738e-05,
          4.27842799e-04,  5.83418411e-06, -2.62717772e-06,
          5.35296733e-05, -1.23409176e-05,  1.29914641e-06],
        [ 1.86048651e-05,  2.38786201e-05,  3.16310173e-05,
          5.83418411e-06,  5.89025512e-05,  1.50549994e-05,
          1.92002465e-05, -8.49113392e-05, -5.78487390e-07],
        [ 2.05304755e-06,  8.43863345e-06,  1.27374356e-06,
         -2.62717772e-06,  1.50549994e-05,  8.51511069e-05,
          5.84851727e-05, -5.63406343e-05, -1.88198418e-07],
        [ 3.55016677e-06,  6.53970495e-06, -1.59582259e-05,
          5.35296733e-05,  1.92002465e-05,  5.84851727e-05,
          1.70683186e-04, -5.78501158e-05,  5.46475324e-07],
        [-9.27833097e-05, -8.30124023e-05, -3.71395599e-04,
         -1.23409176e-05, -8.49113392e-05, -5.63406343e-05,
         -5.78501158e-05,  1.20577180e-03,  7.22586139e-07],
        [ 4.99988031e-07, -2.33583196e-08,  1.61197161e-06,
          1.29914641e-06, -5.78487390e-07, -1.88198418e-07,
          5.46475324e-07,  7.22586139e-07,  1.04254200e-07]
        ],
            columns=self.TSData_multi.value.columns,
            index=self.TSData_multi.value.columns,
        )
        # fmt: on
        assert_frame_equal(expected, m.sigma_u)

    def test_bad_params(
        self,
    ) -> None:
        for p, phi0, phi1, phi2, phi3 in [
            (0, 1, 1, 1, 1),
            (1, 0, 1, 1, 1),
            (1, 1, 0, 1, 1),
            (1, 1, 2, 1, 1),
            (1, 1, 1, 0, 1),
            (1, 1, 1, 1, 0),
        ]:
            params = BayesianVARParams()
            params.p = p
            params.phi_0 = phi0
            params.phi_1 = phi1
            params.phi_2 = phi2
            params.phi_3 = phi3
            with self.assertRaises(ValueError):
                params.validate_params()


if __name__ == "__main__":
    unittest.main()
