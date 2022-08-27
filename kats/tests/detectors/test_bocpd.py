# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing
from collections import Counter
from operator import attrgetter
from typing import Sequence
from unittest import TestCase

import numpy as np
from kats.consts import TimeSeriesData
from kats.detectors.bocpd import (
    BOCPDChangePoint,
    BOCPDetector,
    BOCPDModelType,
    NormalKnownParameters,
    PoissonModelParameters,
    TrendChangeParameters,
)
from kats.utils.simulator import Simulator
from parameterized.parameterized import parameterized


class BOCPDTest(TestCase):
    first_cp_begin = 100
    first_cp_end = 200
    second_cp_begin = 350

    sigma = 0.05  # std. dev
    num_points = 450

    def setUp(self) -> None:
        self.sim = Simulator(n=450, start="2018-01-01")

        self.cp_array_input = [
            BOCPDTest.first_cp_begin,
            BOCPDTest.first_cp_end,
            BOCPDTest.second_cp_begin,
        ]

        self.level_arr = [1.35, 1.05, 1.35, 1.2]

        ## Normal ##
        self.normal_ts = self.sim.level_shift_sim(
            random_seed=100,
            cp_arr=self.cp_array_input,
            level_arr=self.level_arr,
            noise=0.05,
            seasonal_period=7,
            seasonal_magnitude=0.0,
        )

        # build possible changepoints giving a little slack
        # algorithm can detect a few points before/after
        self.normal_cp_arr = np.concatenate(
            (
                self.normal_ts.time.values[
                    range(BOCPDTest.first_cp_begin - 5, BOCPDTest.first_cp_begin + 5)
                ],
                self.normal_ts.time.values[
                    range(BOCPDTest.first_cp_end - 5, BOCPDTest.first_cp_end + 5)
                ],
                self.normal_ts.time.values[
                    range(BOCPDTest.second_cp_begin - 5, BOCPDTest.second_cp_begin + 5)
                ],
            )
        )

        self.normal_bocpd_model = BOCPDetector(data=self.normal_ts)

        self.normal_cps = self.normal_bocpd_model.detector(
            model=BOCPDModelType.NORMAL_KNOWN_MODEL,
            changepoint_prior=0.01,
            choose_priors=False,
            agg_cp=False,
        )

        cps_params = NormalKnownParameters()
        cps_params.search_method = "gridsearch"
        self.normal_gridsearch_cps = self.normal_bocpd_model.detector(
            model=BOCPDModelType.NORMAL_KNOWN_MODEL,
            model_parameters=cps_params,
            changepoint_prior=0.01,
            choose_priors=True,
        )

        ## Normal Mutilvariate ##
        self.multnorm_ts = self.sim.level_shift_multivariate_indep_sim(
            cp_arr=self.cp_array_input,
            level_arr=self.level_arr,
            noise=0.04,
            seasonal_period=7,
            seasonal_magnitude=0.0,
            dim=3,
        )

        # build possible changepoints giving a little slack
        # algorithm can detect a few points before/after
        self.multnorm_cp_arr = np.concatenate(
            (
                self.multnorm_ts.time.values[
                    range(BOCPDTest.first_cp_begin - 5, BOCPDTest.first_cp_begin + 5)
                ],
                self.multnorm_ts.time.values[
                    range(BOCPDTest.first_cp_end - 5, BOCPDTest.first_cp_end + 5)
                ],
                self.multnorm_ts.time.values[
                    range(BOCPDTest.second_cp_begin - 5, BOCPDTest.second_cp_begin + 5)
                ],
            )
        )

        self.multnorm_bocpd_model = BOCPDetector(data=self.multnorm_ts)

        # We should have 3 change points per time series (of which there are 3)
        # However, we have set different change point priors, so we lose 3
        # and we set different thresholds, so we lose the other 3.
        self.multnorm_cps_changepointpriors_and_thresholds = self.multnorm_bocpd_model.detector(
            model=BOCPDModelType.NORMAL_KNOWN_MODEL,
            # pyre-fixme[6]: For 2nd param expected `float` but got `ndarray`.
            changepoint_prior=np.array([0.01, 0.01, 1.0]),
            # pyre-fixme[6]: For 3rd param expected `float` but got `ndarray`.
            threshold=np.array([1.0, 0.5, 0.5]),
            choose_priors=False,
            agg_cp=False,
        )

        # check if multivariate detection works in detecting all changepoints
        self.multnorm_cps = self.multnorm_bocpd_model.detector(
            model=BOCPDModelType.NORMAL_KNOWN_MODEL,
            # pyre-fixme[6]: For 2nd param expected `float` but got `ndarray`.
            changepoint_prior=np.array([0.01, 0.01, 0.01]),
            # pyre-fixme[6]: For 3rd param expected `float` but got `ndarray`.
            threshold=np.array([0.85, 0.85, 0.85]),
            choose_priors=False,
        )

        # Trend
        self.trend_sim = Simulator(n=200, start="2018-01-01")
        self.trend_ts = self.trend_sim.trend_shift_sim(
            random_seed=15,
            cp_arr=[100],
            trend_arr=[3, 28],
            intercept=30,
            noise=30,
            seasonal_period=7,
            seasonal_magnitude=0,
        )

        self.trend_bocpd_model = BOCPDetector(data=self.trend_ts)
        self.trend_cps = self.trend_bocpd_model.detector(
            model=BOCPDModelType.TREND_CHANGE_MODEL,
            model_parameters=TrendChangeParameters(
                readjust_sigma_prior=True, num_points_prior=20
            ),
            debug=True,
            threshold=0.5,
            choose_priors=False,
            agg_cp=True,
        )

        # Poisson
        self.poisson_ts = self.sim.level_shift_sim(
            random_seed=100,
            cp_arr=self.cp_array_input,
            level_arr=self.level_arr,
            noise=0.05,
            seasonal_period=7,
            seasonal_magnitude=0.0,
        )

        # check if the change points were detected
        # build possible changepoints giving a little slack
        # algorithm can detect a few points before/after
        self.poisson_cp_arr = np.concatenate(
            (
                self.poisson_ts.time.values[
                    range(BOCPDTest.first_cp_begin - 5, BOCPDTest.first_cp_begin + 5)
                ],
                self.poisson_ts.time.values[
                    range(BOCPDTest.first_cp_end - 5, BOCPDTest.first_cp_end + 5)
                ],
                self.poisson_ts.time.values[
                    range(BOCPDTest.second_cp_begin - 5, BOCPDTest.second_cp_begin + 5)
                ],
            )
        )

        self.poisson_bocpd_model = BOCPDetector(data=self.poisson_ts)

        self.poisson_cps = self.poisson_bocpd_model.detector(
            model=BOCPDModelType.POISSON_PROCESS_MODEL,
            changepoint_prior=0.01,
            model_parameters=PoissonModelParameters(beta_prior=0.01),
            choose_priors=False,
        )

    def assert_changepoints_exist(
        self, ts: TimeSeriesData, cp_arr: np.ndarray, cps: Sequence[BOCPDChangePoint]
    ) -> None:
        # check if the change points were detected
        # TODO: this check only tests that all changepoints we find should be there
        #       but not the other way around, that we find all change points.
        for t in cps:
            cp = t.start_time
            if cp == ts.time.values[0]:
                continue
            self.assertIn(cp, cp_arr)

    # Test Plots #

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator `parameter...
    @parameterized.expand(
        [
            ("normal", "normal_bocpd_model", "normal_cps"),
            ("normal", "normal_bocpd_model", "normal_gridsearch_cps"),
            ("multivariate_normal", "multnorm_bocpd_model", "multnorm_cps"),
            (
                "multivariate_normal",
                "multnorm_bocpd_model",
                "multnorm_cps_changepointpriors_and_thresholds",
            ),
            ("trend", "trend_bocpd_model", "trend_cps"),
            ("poisson", "poisson_bocpd_model", "poisson_cps"),
        ]
    )
    def test_plots(self, _: str, detector_name: str, cp_name: str) -> None:
        attrgetter(detector_name)(self).plot(attrgetter(cp_name)(self))

    # Test Normal #
    def test_normal_change_prob_len(self) -> None:

        change_prob_dict = self.normal_bocpd_model.get_change_prob()
        change_prob = list(change_prob_dict.values())[
            0
        ]  # dict only has a single element here
        self.assertEqual(change_prob.shape[0], len(self.normal_ts))

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    #  `parameterized.parameterized.parameterized.expand([("default", False, False),
    #  ("w_priors", True, False), ("w_agg_post", False, True),
    #  ("w_priors_and_agg_post", True, True)])`.
    @parameterized.expand(
        [
            ("default", False, False),
            ("w_priors", True, False),
            ("w_agg_post", False, True),
            ("w_priors_and_agg_post", True, True),
        ]
    )
    def test_normal_changepoints(
        self, _: str, choose_priors: bool, agg_cp: bool
    ) -> None:

        self.assert_changepoints_exist(
            self.normal_ts, self.normal_cp_arr, self.normal_cps
        )

    def test_normal_changepoints_gridsearch(self) -> None:
        # test the case where search method has been changed to gridsearch

        self.assert_changepoints_exist(
            self.normal_ts, self.normal_cp_arr, self.normal_gridsearch_cps
        )

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    #  `parameterized.parameterized.parameterized.expand([("default", False, False),
    #  ("w_priors", True, False), ("w_agg_post", False, True),
    #  ("w_priors_and_agg_post", True, True)])`.
    @parameterized.expand(
        [
            ("default", False, False),
            ("w_priors", True, False),
            ("w_agg_post", False, True),
            ("w_priors_and_agg_post", True, True),
        ]
    )
    def test_additional_multivariate_normal_plots(
        self, _: str, choose_priors: bool, agg_cp: bool
    ) -> None:
        # check if multivariate detection works with choosing priors
        cps = self.multnorm_bocpd_model.detector(
            model=BOCPDModelType.NORMAL_KNOWN_MODEL,
            choose_priors=choose_priors,
            agg_cp=agg_cp,
        )
        self.multnorm_bocpd_model.plot(cps)

    def test_normal_multivariate_changeprob_length(self) -> None:

        change_prob_dict = self.multnorm_bocpd_model.get_change_prob()
        change_prob_val = change_prob_dict.values()

        for prob_arr in change_prob_val:
            self.assertEqual(prob_arr.shape[0], len(self.multnorm_ts))

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    #  `parameterized.parameterized.parameterized.expand([("default", "multnorm_cps"),
    #  ("changepointpriors_and_thresholds",
    #  "multnorm_cps_changepointpriors_and_thresholds")])`.
    @parameterized.expand(
        [
            ("default", "multnorm_cps"),
            (
                "changepointpriors_and_thresholds",
                "multnorm_cps_changepointpriors_and_thresholds",
            ),
        ]
    )
    def test_normal_multivariate_changepoints(self, _: str, cps_name: str) -> None:
        cps = getattr(self, cps_name)

        for t in cps:
            cp = t.start_time
            if cp == self.multnorm_ts.time.values[0]:
                continue
            self.assertIn(cp, self.multnorm_cp_arr)

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator `parameter...
    @parameterized.expand(
        [
            ("default", "multnorm_cps", Counter(value1=3, value2=3, value3=3)),
            (
                "changepointpriors_and_thresholds",
                "multnorm_cps_changepointpriors_and_thresholds",
                Counter(value2=3),
            ),
        ]
    )
    def test_normal_multivariate_num_timeseries(
        self,
        _: str,
        cps_name: str,
        target_counter: typing.Counter[str],
    ) -> None:
        cps = getattr(self, cps_name)
        counter = Counter()
        for t in cps:
            ts_name = t.ts_name
            cp = t.start_time
            if cp == self.multnorm_ts.time.values[0]:
                continue
            counter += Counter({ts_name: 1})

        # Check we have all the time series.
        self.assertEqual(counter, target_counter)

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    #  `parameterized.parameterized.parameterized.expand([("default", "multnorm_cps",
    #  9), ("changepointpriors_and_thresholds",
    #  "multnorm_cps_changepointpriors_and_thresholds", 3)])`.
    @parameterized.expand(
        [
            ("default", "multnorm_cps", 9),
            # We should have 3 change points per time series (of which there are 3)
            # However, we have set different change point priors, so we lose 3
            # and we set different thresholds, so we lose the other 3.
            (
                "changepointpriors_and_thresholds",
                "multnorm_cps_changepointpriors_and_thresholds",
                3,
            ),
        ]
    )
    def test_normal_multivariate_changepoints_length(
        self, _: str, cps_name: str, target_len: int
    ) -> None:
        cps = getattr(self, cps_name)
        self.assertEqual(len(cps), target_len)

    def test_trend(self) -> None:
        # expect only one cp
        # test if atleast one cp is in 90:110
        start_list = [cp.start_time for cp in self.trend_cps]
        intersect = list(set(start_list) & set(self.trend_ts.time.values[90:110]))
        self.assertGreaterEqual(len(intersect), 1)

    def test_trend_confidence(self) -> None:
        # check if confidence is greater than threshold
        self.assertGreaterEqual(
            self.trend_cps[0].confidence,
            # pyre-fixme[6]: For 2nd param expected `SupportsDunderLE[Variable[_T]]`
            #  but got `float`.
            0.5,
            f"confidence should have been at least threshold 0.5, but got {self.trend_cps[0].confidence}",
        )

    def test_poisson_changepoints(self) -> None:

        self.assert_changepoints_exist(
            self.poisson_ts, self.poisson_cp_arr, self.poisson_cps
        )

    def test_time_col_name(self) -> None:

        df = self.normal_ts.to_dataframe()
        df.rename(columns={"time": "ds"}, inplace=True)
        ts = TimeSeriesData(df, time_col_name="ds")
        try:
            detector = BOCPDetector(data=ts)
            detector.detector()
        except Exception:
            self.assertTrue(False)
