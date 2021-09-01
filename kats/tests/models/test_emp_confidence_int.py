# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import io
import os
import pkgutil
import unittest
from typing import Any, Dict, cast
from unittest import TestCase

import matplotlib.pyplot as plt
import pandas as pd
from kats.consts import TimeSeriesData
from kats.models.prophet import ProphetModel, ProphetParams
from kats.utils.emp_confidence_int import EmpConfidenceInt


ALL_ERRORS = ["mape", "smape", "mae", "mase", "mse", "rmse"]


def load_data(file_name):
    ROOT = "kats"
    if "kats" in os.getcwd().lower():
        path = "data/"
    else:
        path = "kats/data/"
    data_object = pkgutil.get_data(ROOT, path + file_name)
    return pd.read_csv(io.BytesIO(data_object), encoding="utf8")


def get_default_arguments(method):
    sig = inspect.signature(method)
    return {
        k: v.default
        for k, v in sig.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def test_name(funcname: str, num: int, kwargs: Dict[str, Any]):
    testname = kwargs.get("testname", "")
    return f"{funcname}_{num}_{testname}"


class testEmpConfidenceInt(TestCase):
    def setUp(self):
        DATA = load_data("air_passengers.csv")
        DATA.columns = ["time", "y"]
        self.TSData = TimeSeriesData(DATA)
        params = ProphetParams(seasonality_mode="multiplicative")
        self.params = params
        self.unfit_ci = EmpConfidenceInt(
            ALL_ERRORS,
            self.TSData,
            params,
            50,
            25,
            12,
            ProphetModel,
            confidence_level=0.9,
        )
        self.ci = EmpConfidenceInt(
            ALL_ERRORS,
            self.TSData,
            params,
            50,
            25,
            12,
            ProphetModel,
            confidence_level=0.9,
        )
        ci_plot = EmpConfidenceInt(
            ALL_ERRORS,
            self.TSData,
            params,
            50,
            25,
            12,
            ProphetModel,
            confidence_level=0.9,
        )
        self.ci_plot = ci_plot
        self.ci_diagnose_defaults = get_default_arguments(ci_plot.diagnose)
        self.ci_plot_defaults = get_default_arguments(ci_plot.plot)
        self.ci_plot.get_eci(steps=10, freq="MS")

    def test_empConfInt_Prophet(self) -> None:
        result = self.ci.get_eci(steps=10, freq="MS")
        expected = pd.DataFrame(
            data={
                "time": pd.date_range("1961-01-01", "1961-10-01", freq="MS"),
                "fcst": [
                    452.077721,
                    433.529496,
                    492.499917,
                    495.895518,
                    504.532772,
                    580.506512,
                    654.849614,
                    650.944635,
                    554.067652,
                    490.207818,
                ],
                "fcst_lower": [
                    428.329060,
                    408.808464,
                    466.806514,
                    469.229744,
                    476.894627,
                    551.895995,
                    625.266726,
                    620.389377,
                    522.540022,
                    457.707818,
                ],
                "fcst_upper": [
                    475.826382,
                    458.250528,
                    518.193320,
                    522.561292,
                    532.170918,
                    609.117028,
                    684.432501,
                    681.499894,
                    585.595281,
                    522.707819,
                ],
            }
        )
        pd.testing.assert_frame_equal(expected, result)

    def test_errors(self) -> None:
        for i, kwargs in enumerate(
            [
                {"testname": "train_too_low", "train": 0},
                {"testname": "train_too_low", "train": 101},
                {"testname": "train_too_low", "test": 0},
                {"testname": "train_too_low", "test": 101},
                {"testname": "negative_steps", "steps": -1},
            ]
        ):
            with self.subTest(msg=test_name("test_errors", i, kwargs)):
                train_pct = cast(int, kwargs.get("train", 50))
                test_pct = cast(int, kwargs.get("test", 25))
                steps = cast(int, kwargs.get("steps", 12))

                with self.assertRaises(ValueError):
                    _ = EmpConfidenceInt(
                        ALL_ERRORS,
                        self.TSData,
                        self.params,
                        train_pct,
                        test_pct,
                        steps,
                        ProphetModel,
                        confidence_level=0.9,
                    )

    def test_diagnose_error_unfit(self) -> None:
        with self.assertRaises(ValueError):
            _ = self.unfit_ci.diagnose()

    def test_diagnose(self) -> None:
        for i, kwargs in enumerate(
            [
                {"testname": "typical"},
                {"testname": "custom_ax", "ax": None},
                {"testname": "custom_figsize", "figsize": (8, 5)},
                {"testname": "no_legend", "legend": False},
                {
                    "testname": "custom",
                    "ax": None,
                    "figsize": (8, 5),
                    "linecolor": "purple",
                    "linelabel": "foo",
                    "secolor": "orange",
                    "selabel": "bar",
                    "legend": False,
                },
            ]
        ):
            with self.subTest(msg=test_name("test_diagnose", i, kwargs)):
                figsize = kwargs.get("figsize", None)
                if "ax" in kwargs:
                    _, ax = plt.subplots(figsize=(6, 4) if figsize is None else figsize)
                else:
                    ax = None
                defaults = self.ci_diagnose_defaults
                linecolor = kwargs.get("linecolor", defaults["linecolor"])
                linelabel = kwargs.get("linelabel", defaults["linelabel"])
                secolor = kwargs.get("secolor", defaults["secolor"])
                selabel = kwargs.get("selabel", defaults["selabel"])
                legend = kwargs.get("legend", defaults["legend"])
                ax = self.ci_plot.diagnose(
                    ax=ax,
                    figsize=figsize,
                    linecolor=linecolor,
                    linelabel=linelabel,
                    secolor=secolor,
                    selabel=selabel,
                    legend=legend,
                )
                self.assertIsNotNone(ax)
                # TODO: add visual diff tests

    def test_plot_error_unfit(self) -> None:
        with self.assertRaises(ValueError):
            _ = self.unfit_ci.plot()

    def test_plot(self) -> None:
        for i, kwargs in enumerate(
            [
                {"testname": "typical"},
                {"testname": "custom_ax", "ax": None},
                {"testname": "custom_figsize", "figsize": (8, 5)},
                {"testname": "no_grid", "grid": False},
                {
                    "testname": "custom",
                    "ax": None,
                    "figsize": (8, 5),
                    "linecolor": "purple",
                    "linelabel": "foo",
                    "secolor": "orange",
                    "selabel": "bar",
                    "legend": False,
                },
            ]
        ):
            with self.subTest(msg=test_name("test_plot", i, kwargs)):
                figsize = kwargs.get("figsize", None)
                if "ax" in kwargs:
                    _, ax = plt.subplots(figsize=(6, 4) if figsize is None else figsize)
                else:
                    ax = None
                defaults = self.ci_plot_defaults
                linecolor = kwargs.get("linecolor", defaults["linecolor"])
                fcstcolor = kwargs.get("fcstcolor", defaults["fcstcolor"])
                intervalcolor = kwargs.get("intervalcolor", defaults["intervalcolor"])
                intervalalpha = kwargs.get("intervalalpha", defaults["intervalalpha"])
                modelcolor = kwargs.get("modelcolor", defaults["modelcolor"])
                modelalpha = kwargs.get("modelalpha", defaults["modelalpha"])
                grid = kwargs.get("grid", defaults["grid"])
                ax = self.ci_plot.plot(
                    ax=ax,
                    figsize=figsize,
                    linecolor=linecolor,
                    fcstcolor=fcstcolor,
                    intervalcolor=intervalcolor,
                    intervalalpha=intervalalpha,
                    modelcolor=modelcolor,
                    modelalpha=modelalpha,
                    grid=grid,
                )
                self.assertIsNotNone(ax)


if __name__ == "__main__":
    unittest.main()
