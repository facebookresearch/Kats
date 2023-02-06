# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import unittest
from typing import Any, cast, Dict, List
from unittest import mock, TestCase

import kats.utils.emp_confidence_int  # noqa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kats.compat.pandas import assert_frame_equal
from kats.consts import Params, TimeSeriesData
from kats.data.utils import load_air_passengers
from kats.models.model import Model
from kats.utils.emp_confidence_int import BackTesterRollingWindow, EmpConfidenceInt


ALL_ERRORS = ["mape", "smape", "mae", "mase", "mse", "rmse"]


# pyre-fixme[2]: Parameter must be annotated.
def get_default_arguments(method) -> Dict[str, Any]:
    sig = inspect.signature(method)
    return {
        k: v.default
        for k, v in sig.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def get_name(funcname: str, num: int, kwargs: Dict[str, Any]) -> str:
    testname = kwargs.get("testname", "")
    return f"{funcname}_{num}_{testname}"


# Output from running
# ci = EmpConfidenceInt(
#    ALL_ERRORS,
#    load_air_passengers(),
#    ProphetParams(seasonality_mode="multiplicative"),
#    50,
#    25,
#    12,
#    ProphetModel,
#    confidence_level=0.9,
# )
# _ = ci.get_eci(steps=10, freq="MS")
# _FROZEN_DATA = ci.predicted
_FROZEN_DATA = pd.DataFrame(
    {
        "time": [
            pd.Timestamp("1961-01-01 00:00:00"),
            pd.Timestamp("1961-02-01 00:00:00"),
            pd.Timestamp("1961-03-01 00:00:00"),
            pd.Timestamp("1961-04-01 00:00:00"),
            pd.Timestamp("1961-05-01 00:00:00"),
            pd.Timestamp("1961-06-01 00:00:00"),
            pd.Timestamp("1961-07-01 00:00:00"),
            pd.Timestamp("1961-08-01 00:00:00"),
            pd.Timestamp("1961-09-01 00:00:00"),
            pd.Timestamp("1961-10-01 00:00:00"),
        ],
        "fcst": [
            452.07772055873204,
            433.52949637704114,
            492.49991708342753,
            495.89551818842506,
            504.53277256597954,
            580.5065115354556,
            654.8496137044091,
            650.9446353669717,
            554.0676519292672,
            490.2078181542509,
        ],
        "fcst_lower": [
            439.6423453280892,
            420.46356554973414,
            478.89099768408835,
            483.51694046072885,
            491.2561604536841,
            566.6048470292841,
            642.8619066164697,
            638.4621408959206,
            541.313705833062,
            476.5572516284846,
        ],
        "fcst_upper": [
            464.9579443844055,
            446.9729631750037,
            505.68309652107126,
            508.7417672862129,
            518.3582443565921,
            593.6895929349565,
            668.0317888909123,
            664.5665151298979,
            567.633193452028,
            502.5135100298137,
        ],
    }
)

# Obtained with
# backtester = BackTesterRollingWindow(
#    ci.error_methods,
#    ci.data,
#    ci.params,
#    ci.train_percentage,
#    ci.test_percentage,
#    ci.sliding_steps,
#    ci.model_class,
#    ci.multi,
# )
# backtester.run_backtest()
# _RAW_ERRORS = backtester.raw_errors
# fmt: off
_RAW_ERRORS: List[np.ndarray] = [
    np.array([
        1.33793031e01, 1.56117160e00, -3.43035881e-02, 1.59373281e01,
        1.00730540e01, 3.66296231e01, 4.33868022e01, 3.67005869e01,
        2.82183680e01, 2.94395101e01, 2.29291833e01, 3.36465132e01,
        3.77634219e01, 2.47023098e01, 2.67704047e01, 3.18079789e01,
        4.58371754e01, 6.17309299e01, 8.28299401e01, 6.17988444e01,
        5.98808036e01, 3.97375976e01, 3.72830104e01, 4.31723084e01,
        4.93706688e01, 3.92250383e01, 4.56086572e01, 4.97268965e01,
        6.06837536e01, 9.21593920e01, 1.06311307e02, 1.03128831e02,
        8.48049423e01, 6.34985826e01, 5.65794654e01, 5.53982669e01
    ]), np.array([
        8.11503221, 32.85179971, 38.1773251, 31.20830919, 22.83613649,
        25.18019481, 19.95353269, 31.44100063, 28.19075755, 20.84688681,
        18.42117867, 29.44474645, 37.48972786, 53.6757277, 75.20675324,
        54.29720924, 53.9564216, 33.9279106, 31.392959, 35.26362491,
        41.64895783, 35.19132276, 37.15490514, 42.05732988, 51.44437459,
        82.25785987, 95.93139254, 92.75694643, 75.95044928, 55.47231879,
        49.29311605, 46.78829892, 45.53589367, 28.87135916, 23.41573699,
        19.6015136
    ]), np.array([
        6.43628819, 8.46692472, 2.31048776, 8.87826746, 3.33150705,
        -4.69897302, -11.47274692, -3.866925, 3.54407259, 17.48006509,
        35.51531218, 8.89874617, 9.60834045, -4.83681831, -0.85771482,
        0.13613395, 6.16032569, 0.19557271, -9.9412546, -5.96883857,
        1.98659046, 23.46292865, 30.79256934, 25.32481485, 15.92345691,
        2.32690966, 3.08342995, -5.31244378, -7.91361351, -24.05883907,
        -40.14562282, -42.36867187, -28.43535616, -5.2057038, -2.13239425,
        18.54815302
    ]), np.array([
        -0.97621443, -9.77768055, -17.70556268, -9.91675413, -2.64647774,
         8.97599059, 24.73220812, -4.23069623, 1.22788942, -13.64299708,
         -11.13669419, -9.34953254, -6.03149411, -11.51804039, -22.75441027,
         -18.39575646, -10.42411953, 8.3900628, 13.17837127, 6.1880967,
         0.07831881, -13.18810358, -10.93128271, -21.19002872, -24.92990015,
         -40.81371547, -59.49776673, -61.19831502, -47.12457593, -26.74303742,
         -26.71510953, -6.17019754, -50.57903196, -41.1429546, -38.98618728,
         -62.29567193
    ]), np.array([
        -8.29654202, -0.68634186, 12.57051743, 29.68672159, 2.03042777,
        7.1764183, -8.33522202, -6.62482338, -4.15694547, -1.47075556,
        -4.82910017, -8.95704954, -12.56408943, -4.48588113, 16.14285273,
        22.543945, 16.60188572, 9.73445296, -4.65347052, -3.56958154,
        -12.80935829, -16.7233447, -28.76660973, -44.6192084, -50.95321203,
        -37.06763801, -14.8209301, -12.77194197, 8.24275652, -37.22304602,
        -29.45171176, -28.83199048, -50.71545173, -37.44934967, -46.569484,
        -41.74881161
    ]), np.array([
        29.40243384, 1.89010206, 7.24006683, -8.45259265, -7.04628643,
        -5.29116589, -3.07346362, -6.69728494, -9.88907499, -9.53726558,
        -6.14396769, 8.9945401, 21.27529524, 15.38349283, 8.75932566,
        -5.61441026, -4.59378286, -14.33694072, -18.52533181, -30.68199841,
        -45.93588853, -50.3525494, -40.19273227, -20.0559752, -15.17212749,
        5.84440392, -39.35852662, -31.31535245, -30.46043197, -52.574062,
        -39.34650964, -48.41833516, -43.43859599, -42.77277546, -25.12845727,
        -23.98293343
    ]), np.array([
        -11.09693919, -10.19931703, -10.50027015, -9.79268559, -14.97423814,
        -15.70900783, -15.2237116, -11.87084218, 2.69294037, 0.94352273,
        7.06656394, -0.52638277, -12.54825071, -11.12434467, -22.64698507,
        -27.75288247, -40.6619583, -56.06844492, -60.42365081, -50.69613356,
        -31.79236767, -34.72629314, -8.76528041, -52.6634822, -42.82177224,
        -40.4473734, -63.92084476, -50.86313044, -59.76714476, -58.08918614,
        -57.45115547, -40.69078438, -41.33920088, -40.44672867, -8.12526774,
        -46.53812807
    ]), np.array([
        -9.97839155, -12.42194859, -12.33990508, -11.7155684, -8.28210751,
        6.74562599, 5.3282428, 11.2175128, 3.1756324, -5.22326815, -5.08683361,
        -15.12787518, -24.6881203, -36.08618244, -50.38701014, -55.38680992,
        -45.83513038, -25.99070672, -27.61778658, -1.3085659, -45.17766082,
        -34.81727035, -33.78101525, -55.02442087, -44.12076223, -53.11325386,
        -50.07871059, -51.02127243, -34.70707069, -33.89844924, -30.48826259,
        2.84084501, -34.9289766, -28.59008544, -15.15099215, -28.16788562
    ]), np.array([
        -9.98094349, -6.37006441, 8.65211401, 7.65716831, 13.7018955,
        5.34661197, -3.37519534, -3.47443295, -13.16525524, -18.12164927,
        -27.5568301, -38.23648211, -52.79189579, -43.24754991, -23.62531995,
        -24.497487, 2.3314004, -41.63559343, -31.4350637, -30.5574858,
        -51.33989102, -38.50095225, -45.75418467, -31.30710373, -47.67207221,
        -31.55162359, -31.24719003, -26.65123601, 7.62360906, -29.93385307,
        -23.54446881, -10.17717866, -22.66205047, -22.12459763, -39.76804606,
        -81.19033446
    ]), np.array([
        7.98034297, 13.90850901, 5.4262051, -3.37372995, -3.50783896,
        -13.17725768, -18.10206021, -27.52973468, -38.29695448, -48.40011193,
        -37.80215288, -26.85950869, -24.37713833, 2.57513387, -41.30599573,
        -31.08239141, -30.23064295, -51.10803755, -38.40152726, -45.80562318,
        -31.81207994, -42.58548025, -23.61425884, -35.56183364, -26.81859503,
        7.87761465, -29.32633663, -22.79318029, -9.44588704, -22.1740084,
        -21.96741275, -39.94500796, -79.62625418, -20.54024838, -12.85831936,
        -29.99212219
    ]), np.array([
        -4.85353392, -4.73850058, -14.57272169, -19.59135519, -29.07178092,
        -40.67645724, -50.63883713, -40.06962156, -29.28506629, -32.71427002,
        -5.4175811, -44.86550358, -33.24816908, -32.38232471, -53.64027738,
        -40.98220932, -48.22705212, -35.44069734, -45.65852012, -26.46790868,
        -38.56285232, -38.75745067, -1.79970304, -33.34585704, -25.6010857,
        -12.54132765, -25.8923552, -25.67233828, -43.25614488, -83.82481328,
        -25.31228535, -18.22503236, -36.14701369, 4.01834411, -18.2097369,
        -32.54535836
    ]), np.array([
        -18.44823146, -27.901758, -39.24837822, -49.22234289, -38.66649663,
        -28.28769385, -31.23945923, -3.67130037, -43.35859788, -30.756412,
        -28.28216366, -47.32547612, -39.60274825, -46.82795139, -33.13069828,
        -43.42340251, -24.34981572, -37.19118123, -36.61717368, 0.70535654,
        -31.2418436, -22.886315, -6.75428849, -18.69870166, -24.12345858,
        -41.67172717, -81.67778501, -23.14325758, -15.88939984, -33.38933311,
        7.05700989, -15.07906055, -29.73410285, -7.5264136, -25.20206752,
        -28.13923221
    ])
]
# fmt:on


class FakeParams(Params):
    pass


class MyFakeModel(Model[FakeParams]):
    unfit: bool = True

    def __init__(self, data: TimeSeriesData, params: FakeParams) -> None:
        self.unfit = True

    def fit(self, *_args: Any, **_kwargs: Any) -> None:
        self.unfit = False

    # pyre-fixme[15]: `predict` overrides method defined in `Model` inconsistently.
    def predict(
        self, steps: int, include_history: bool = False, *_args: Any, **_kwargs: Any
    ) -> pd.DataFrame:
        if self.unfit:
            raise ValueError("Model hasn't been fit")
        return _FROZEN_DATA


class testEmpConfidenceInt(TestCase):
    @mock.patch("kats.utils.emp_confidence_int.BackTesterRollingWindow")
    def setUp(self, backtester: BackTesterRollingWindow) -> None:
        backtester.raw_errors = _RAW_ERRORS
        self.TSData = load_air_passengers()

        params = FakeParams()
        self.params = params
        self.unfit_ci = EmpConfidenceInt(
            ALL_ERRORS,
            self.TSData,
            params,
            50,
            25,
            12,
            MyFakeModel,
            confidence_level=0.9,
        )
        self.ci = EmpConfidenceInt(
            ALL_ERRORS,
            self.TSData,
            params,
            50,
            25,
            12,
            MyFakeModel,
            confidence_level=0.9,
        )
        ci_plot = EmpConfidenceInt(
            ALL_ERRORS,
            self.TSData,
            params,
            50,
            25,
            12,
            MyFakeModel,
            confidence_level=0.9,
        )
        self.ci_plot = ci_plot
        self.ci_diagnose_defaults = get_default_arguments(ci_plot.diagnose)
        self.ci_plot_defaults = get_default_arguments(ci_plot.plot)
        self.ci_plot.get_eci(steps=10, freq="MS")

    @mock.patch("kats.utils.emp_confidence_int.BackTesterRollingWindow")
    def test_empConfInt_Prophet(self, backtester: BackTesterRollingWindow) -> None:
        backtester.raw_errors = _RAW_ERRORS
        result = self.ci.get_eci(steps=10, freq="MS")
        expected = _FROZEN_DATA.copy()
        expected["fcst_lower"] = expected["fcst"]
        expected["fcst_upper"] = expected["fcst"]
        assert_frame_equal(expected, result, check_like=True)

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
            with self.subTest(msg=get_name("test_errors", i, kwargs)):
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
                        MyFakeModel,
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
            with self.subTest(msg=get_name("test_diagnose", i, kwargs)):
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
            with self.subTest(msg=get_name("test_plot", i, kwargs)):
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
