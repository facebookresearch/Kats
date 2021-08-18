# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import io
import os
import pkgutil
import re
from collections import Counter
from unittest import TestCase

import numpy as np
import pandas as pd
import statsmodels
from kats.detectors.bocpd import (
    BOCPDetector,
    BOCPDModelType,
    NormalKnownParameters,
    PoissonModelParameters,
    TrendChangeParameters,
)
from kats.utils.simulator import Simulator

statsmodels_ver = float(
    re.findall("([0-9]+\\.[0-9]+)\\..*", statsmodels.__version__)[0]
)


def load_data(file_name):
    ROOT = "kats"
    if "kats" in os.getcwd().lower():
        path = "data/"
    else:
        path = "kats/data/"
    data_object = pkgutil.get_data(ROOT, path + file_name)
    return pd.read_csv(io.BytesIO(data_object), encoding="utf8")


class BOCPDTest(TestCase):
    first_cp_begin = 100
    first_cp_end = 200
    second_cp_begin = 350

    sigma = 0.05  # std. dev
    num_points = 450

    def setUp(self):
        self.sim = Simulator(n=450, start="2018-01-01")

        self.cp_array_input = [
            BOCPDTest.first_cp_begin,
            BOCPDTest.first_cp_end,
            BOCPDTest.second_cp_begin,
        ]

        self.level_arr = [1.35, 1.05, 1.35, 1.2]

    def test_normal(self) -> None:

        ts = self.sim.level_shift_sim(
            random_seed=100,
            cp_arr=self.cp_array_input,
            level_arr=self.level_arr,
            noise=0.05,
            seasonal_period=7,
            seasonal_magnitude=0.0,
        )
        bocpd_model = BOCPDetector(data=ts)

        cps = bocpd_model.detector(
            model=BOCPDModelType.NORMAL_KNOWN_MODEL,
            changepoint_prior=0.01,
            choose_priors=False,
            agg_cp=False,
        )
        bocpd_model.plot(cps)

        change_prob_dict = bocpd_model.get_change_prob()
        change_prob = list(change_prob_dict.values())[
            0
        ]  # dict only has a single element here
        self.assertEqual(change_prob.shape[0], len(ts))

        # check if the change points were detected
        # build possible changepoints giving a little slack
        # algorithm can detect a few points before/after
        cp_arr = np.concatenate(
            (
                ts.time.values[
                    range(BOCPDTest.first_cp_begin - 5, BOCPDTest.first_cp_begin + 5)
                ],
                ts.time.values[
                    range(BOCPDTest.first_cp_end - 5, BOCPDTest.first_cp_end + 5)
                ],
                ts.time.values[
                    range(BOCPDTest.second_cp_begin - 5, BOCPDTest.second_cp_begin + 5)
                ],
            )
        )

        # TODO: this check only tests that all changepoints we find should be there
        #       but not the other way around, that we find all change points.
        for t in cps:
            cp = t[0].start_time
            if cp == ts.time.values[0]:
                continue
            self.assertIn(cp, cp_arr)

        # test the case where priors are chosen automatically
        cps2 = bocpd_model.detector(
            model=BOCPDModelType.NORMAL_KNOWN_MODEL,
            changepoint_prior=0.01,
            choose_priors=True,
            agg_cp=False,
        )
        bocpd_model.plot(cps2)

        for t in cps2:
            cp = t[0].start_time
            if cp == ts.time.values[0]:
                continue
            self.assertIn(cp, cp_arr)

        # test the case where run-length posterior is aggregated
        cps3 = bocpd_model.detector(
            model=BOCPDModelType.NORMAL_KNOWN_MODEL,
            changepoint_prior=0.01,
            choose_priors=False,
            agg_cp=True,
        )
        bocpd_model.plot(cps3)

        for t in cps3:
            cp = t[0].start_time
            if cp == ts.time.values[0]:
                continue
            self.assertIn(cp, cp_arr)

        # test the case where run-length posterior is aggregated and
        # automatically tuning prior
        cps4 = bocpd_model.detector(
            model=BOCPDModelType.NORMAL_KNOWN_MODEL,
            changepoint_prior=0.01,
            choose_priors=True,
            agg_cp=True,
        )
        bocpd_model.plot(cps4)

        for t in cps4:
            cp = t[0].start_time
            if cp == ts.time.values[0]:
                continue
            self.assertIn(cp, cp_arr)

        # test the case where search method has been changed to grid
        # search
        cps5_params = NormalKnownParameters()
        cps5_params.search_method = "gridsearch"
        cps5 = bocpd_model.detector(
            model=BOCPDModelType.NORMAL_KNOWN_MODEL,
            model_parameters=cps5_params,
            changepoint_prior=0.01,
            choose_priors=True,
        )
        bocpd_model.plot(cps5)

        for t in cps5:
            cp = t[0].start_time
            if cp == ts.time.values[0]:
                continue
            self.assertIn(cp, cp_arr)

        # test to see if agg_cp=True works
        cps6 = bocpd_model.detector(
            model=BOCPDModelType.NORMAL_KNOWN_MODEL,
            changepoint_prior=0.01,
            choose_priors=True,
            agg_cp=True,
        )

        for t in cps6:
            cp = t[0].start_time
            if cp == ts.time.values[0]:
                continue
            self.assertIn(cp, cp_arr)

    def test_normal_multivariate(self) -> None:

        ts = self.sim.level_shift_multivariate_indep_sim(
            cp_arr=self.cp_array_input,
            level_arr=self.level_arr,
            noise=0.04,
            seasonal_period=7,
            seasonal_magnitude=0.0,
            dim=3,
        )

        bocpd_model = BOCPDetector(data=ts)
        cps = bocpd_model.detector(
            model=BOCPDModelType.NORMAL_KNOWN_MODEL,
            # pyre-fixme[6]: Expected `float` for 2nd param but got `ndarray`.
            changepoint_prior=np.array([0.01, 0.01, 1.0]),
            # pyre-fixme[6]: Expected `float` for 3rd param but got `ndarray`.
            threshold=np.array([1.0, 0.5, 0.5]),
            choose_priors=False,
            agg_cp=False,
        )
        bocpd_model.plot(cps)

        change_prob_dict = bocpd_model.get_change_prob()
        change_prob_val = change_prob_dict.values()

        for prob_arr in change_prob_val:
            self.assertEqual(prob_arr.shape[0], len(ts))

        # check if the change points were detected
        # build possible changepoints giving a little slack
        # algorithm can detect a few points before/after
        cp_arr = np.concatenate(
            (
                ts.time.values[
                    range(BOCPDTest.first_cp_begin - 5, BOCPDTest.first_cp_begin + 5)
                ],
                ts.time.values[
                    range(BOCPDTest.first_cp_end - 5, BOCPDTest.first_cp_end + 5)
                ],
                ts.time.values[
                    range(BOCPDTest.second_cp_begin - 5, BOCPDTest.second_cp_begin + 5)
                ],
            )
        )

        # We should have 3 change points per time series (of which there are 3)
        # However, we have set different change point priors, so we lose 3
        # and we set different thresholds, so we lose the other 3.
        self.assertEqual(len(cps), 3)

        counter = Counter()
        for t in cps:
            ts_name = t[1].ts_name
            cp = t[0].start_time
            if cp == ts.time.values[0]:
                continue
            self.assertIn(cp, cp_arr)
            counter += Counter({ts_name: 1})

        # Check we have all the time series.
        self.assertEqual(counter, Counter(value2=3))

        # check if multivariate detection works with choosing priors
        cps2 = bocpd_model.detector(
            model=BOCPDModelType.NORMAL_KNOWN_MODEL, choose_priors=True, agg_cp=False
        )
        bocpd_model.plot(cps2)

        # check if multivariate detection works with aggregating run-length
        # posterior
        cps3 = bocpd_model.detector(
            model=BOCPDModelType.NORMAL_KNOWN_MODEL, choose_priors=False
        )
        bocpd_model.plot(cps3)

        # check if multivariate detection works with aggregating run-length
        # posterior and automated tuning prior
        cps4 = bocpd_model.detector(
            model=BOCPDModelType.NORMAL_KNOWN_MODEL, choose_priors=True
        )
        bocpd_model.plot(cps4)

        # check if multivariate detection works in detecting all changepoints
        cps5 = bocpd_model.detector(
            model=BOCPDModelType.NORMAL_KNOWN_MODEL,
            # pyre-fixme[6]: Expected `float` for 2nd param but got `ndarray`.
            changepoint_prior=np.array([0.01, 0.01, 0.01]),
            # pyre-fixme[6]: Expected `float` for 3rd param but got `ndarray`.
            threshold=np.array([0.85, 0.85, 0.85]),
            choose_priors=False,
        )
        bocpd_model.plot(cps5)

        change_prob_dict = bocpd_model.get_change_prob()
        change_prob_val = change_prob_dict.values()

        for prob_arr in change_prob_val:
            self.assertEqual(prob_arr.shape[0], len(ts))

        # check if the change points were detected
        # build possible changepoints giving a little slack
        # algorithm can detect a few points before/after
        cp_arr = np.concatenate(
            (
                ts.time.values[
                    range(BOCPDTest.first_cp_begin - 5, BOCPDTest.first_cp_begin + 5)
                ],
                ts.time.values[
                    range(BOCPDTest.first_cp_end - 5, BOCPDTest.first_cp_end + 5)
                ],
                ts.time.values[
                    range(BOCPDTest.second_cp_begin - 5, BOCPDTest.second_cp_begin + 5)
                ],
            )
        )

        # With new algorithm, all changepoints should
        self.assertTrue(len(cps5) >= 9)

        counter = Counter()
        for t in cps5:
            ts_name = t[1].ts_name
            cp = t[0].start_time
            if cp == ts.time.values[0]:
                continue
            self.assertIn(cp, cp_arr)
            counter += Counter({ts_name: 1})

        # Check we have all the time series.
        self.assertEqual(counter, Counter(value1=3, value2=3, value3=3))

    def test_trend(self) -> None:
        sim = Simulator(n=200, start="2018-01-01")
        ts = sim.trend_shift_sim(
            random_seed=15,
            cp_arr=[100],
            trend_arr=[3, 28],
            intercept=30,
            noise=30,
            seasonal_period=7,
            seasonal_magnitude=0,
        )
        threshold = 0.5
        detector = BOCPDetector(data=ts)
        cps = detector.detector(
            model=BOCPDModelType.TREND_CHANGE_MODEL,
            model_parameters=TrendChangeParameters(
                readjust_sigma_prior=True, num_points_prior=20
            ),
            debug=True,
            threshold=threshold,
            choose_priors=False,
            agg_cp=True,
        )
        detector.plot(cps)

        # expect only one cp
        # test if atleast one cp is in 90:110
        start_list = [cp[0].start_time for cp in cps]
        intersect = list(set(start_list) & set(ts.time.values[90:110]))
        self.assertGreaterEqual(len(intersect), 1)

        # check if confidence is greater than threshold
        self.assertGreaterEqual(
            cps[0][0].confidence,
            threshold,
            f"confidence should have been at least threshold {threshold}, but got {cps[0][0].confidence}",
        )

    def test_poisson(self) -> None:

        ts = self.sim.level_shift_sim(
            random_seed=100,
            cp_arr=self.cp_array_input,
            level_arr=self.level_arr,
            noise=0.05,
            seasonal_period=7,
            seasonal_magnitude=0.0,
        )

        bocpd_model = BOCPDetector(data=ts)
        cps = bocpd_model.detector(
            model=BOCPDModelType.POISSON_PROCESS_MODEL,
            changepoint_prior=0.01,
            model_parameters=PoissonModelParameters(beta_prior=0.01),
            choose_priors=False,
        )
        bocpd_model.plot(cps)

        # check if the change points were detected
        # build possible changepoints giving a little slack
        # algorithm can detect a few points before/after
        cp_arr = np.concatenate(
            (
                ts.time.values[
                    range(BOCPDTest.first_cp_begin - 5, BOCPDTest.first_cp_begin + 5)
                ],
                ts.time.values[
                    range(BOCPDTest.first_cp_end - 5, BOCPDTest.first_cp_end + 5)
                ],
                ts.time.values[
                    range(BOCPDTest.second_cp_begin - 5, BOCPDTest.second_cp_begin + 5)
                ],
            )
        )

        # TODO: this check only tests that all changepoints we find should be there
        #       but not the other way around, that we find all change points.
        for t in cps:
            cp = t[0].start_time
            if cp == ts.time.values[0]:
                continue
            self.assertIn(cp, cp_arr)
