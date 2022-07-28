# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import collections
import logging
import time
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from kats.consts import TimeSeriesData
from kats.models.globalmodel.ensemble import GMEnsemble
from kats.models.globalmodel.model import GMModel
from kats.models.globalmodel.utils import fill_missing_value_na, GMParam, split
from kats.utils.backtesters import BackTesterExpandingWindow

"""
This module provides two Classes for backtest for global models:
    1) :class:`GMBackTester`: for hyper-parameter tuning and the global models are evaluted on multiple time series.
    2) :class:`GMBackTesterExpandingWindow`: for evaluating the performance of a :class:`kats.models.globalmodel.model.GMModel` object or
                a :class:`kats.models.globalmodel.ensemble.GMEnsemble` object on a single time series.
"""


class GMBackTester:
    """Backtesting class for global model.

    This class is the backtest framework for global model, which can be used for hyper-parameter tuning (i.e., finding the best :class:`GMParam` object).
    This class provides functions including run_backtest.

    Attributes:
        data: A list or a dictionary of :class:`kats.consts.TimeSeriesData` objects for training and validation.
        gmparam: A :class:`kats.models.globalmodel.utils.GMParam` object.
        backtest_timestamp: A list of strings or `pandas.Timestamp` objects representing the backtest timestamps.
        splits: Optional; An integer representing the number of sub-datasets. For each backtesting timestamp, we divide the whole training data into n=splits sub-datasets, and fit a global model for each sub-dataset. Default is 3.
        overlap: Optional; A boolean representing whether or not sub-datasets overlap with each other. For example, we have samples [ts1, ts2, ts3] and splits = 3.
                 If overlap is True, then three subsets are [[ts1], [ts2], [ts3]], i.e., each sample only appears in one sub-dataset.
                 If overlap is False, then three subsets are [[ts1, ts2], [ts2, ts3], [ts3, ts1]], i.e., each sample appears in (splits-1) sub-datasets. Default is True.
        replicate: Optional; An integer representing the number of global models for each sub-dataset, i.e., we train replicate*splits independent global models for each backtesting timestamp. Default is 1.
        multi: Optional; A boolean representing whether or not to train global models in parallel. Default is False.
        test_size: Optional; A integer representing number of test time series or a float representing the percentage of test time series. Default is 0.1.
        earliest_timestamp: Optional; A string or a `pandas.Timestamp` object representing the earliest timestamp, i.e., we discard all data before the earliest_timestamp. Default is None, and the model uses all the data for model training and validation.
        max_core: Optional; An integer representing the max number of cpu cores for parallel training. Default is None, which sets max_core as max((total_cores - 1) // 2, 1).

    Sample Usage:
        >>> gbm = GMBackTester(data, gmparam, backtest_timestamp = ['2021-02-01', '2021-05-06'])
        >>> evals = gbm.run_backtest()
    """

    def __init__(
        self,
        # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
        data: Union[List[TimeSeriesData], Dict[Any, TimeSeriesData]],
        gmparam: GMParam,
        backtest_timestamp: List[Union[str, pd.Timestamp]],
        splits: int = 3,
        overlap: bool = True,
        replicate: int = 1,
        multi: bool = False,
        test_size: Union[int, float] = 0.1,
        earliest_timestamp: Union[str, pd.Timestamp, None] = None,
        max_core: Optional[int] = None,
    ) -> None:

        if not isinstance(gmparam, GMParam):
            msg = f"gmparam should be GMParam object but receives {type(gmparam)}."
            logging.error(msg)
            raise ValueError(msg)
        self.params = gmparam
        # pyre-fixme[4]: Attribute must be annotated.
        self.max_back_test_timedelta = (
            gmparam.freq * gmparam.validation_step_num * gmparam.fcst_window * 3
        )
        # pyre-fixme[4]: Attribute must be annotated.
        self.min_train_length = (
            gmparam.input_window
            + gmparam.fcst_window
            + gmparam.min_training_step_num * gmparam.min_training_step_length
        )
        # pyre-fixme[4]: Attribute must be annotated.
        self.min_valid_length = gmparam.fcst_window

        # pyre-fixme[4]: Attribute must be annotated.
        self.min_test_length = (
            gmparam.min_warming_up_step_num * gmparam.min_training_step_length
            + gmparam.input_window
        )

        if not isinstance(backtest_timestamp, list) or len(backtest_timestamp) < 1:
            msg = "backtest_timestamp should be a non-empty list of timestamp strings."
            logging.error(msg)
            raise ValueError(msg)
        self.backtest_timestamp = backtest_timestamp

        if not isinstance(splits, int) or splits < 1:
            msg = f"splits should be a positive integer but receives {splits}."
            logging.error(msg)
            raise ValueError(msg)
        self.splits = splits

        self.overlap = overlap

        if not isinstance(replicate, int) or replicate < 1:
            msg = f"rep should be a positive integer but receives {replicate}."
            logging.error(msg)
            raise ValueError(msg)

        self.replicate = replicate

        self.multi = multi

        if (earliest_timestamp is not None) and (
            not isinstance(earliest_timestamp, str)
            and not isinstance(earliest_timestamp, pd.Timestamp)
        ):
            msg = f"earliest_timestamp should either be a str or a pd.Timestamp but receives {type(earliest_timestamp)}."
            logging.error(msg)
            raise ValueError(msg)
        self.earliest_timestamp = earliest_timestamp
        pdata = self._preprocess(data)
        # pyre-fixme[4]: Attribute must be annotated.
        self.data = pdata

        n = len(data)

        if isinstance(test_size, int) and test_size > 0 and test_size < n:
            # pyre-fixme[4]: Attribute must be annotated.
            self.test_size = test_size
        elif isinstance(test_size, float) and test_size > 0 and test_size < 1:
            self.test_size = int(len(data) * test_size)
        else:
            msg = f"test_size should be a positive integer (<={n}) or a float in (0, 1) but receives {test_size}."
            logging.error(msg)
            raise ValueError(msg)

        total_cores = cpu_count()
        if max_core is None:
            # pyre-fixme[4]: Attribute must be annotated.
            self.max_core = max((total_cores - 1) // 2, 1)
        elif isinstance(max_core, int) and max_core > 0 and max_core < total_cores:
            self.max_core = max_core
        else:
            msg = f"max_core should be a positive integer in [1, {total_cores}] but receives {max_core}."
            logging.error(msg)
            raise ValueError(msg)

        # pyre-fixme[4]: Attribute must be annotated.
        self.gm_collects = {
            bt: [GMModel(gmparam) for _ in range(int(replicate * splits))]
            for bt in backtest_timestamp
        }
        # pyre-fixme[4]: Attribute must be annotated.
        self.gm_info_collects = collections.defaultdict(list)
        # pyre-fixme[4]: Attribute must be annotated.
        self.evaluation_collects = []
        # pyre-fixme[4]: Attribute must be annotated.
        self.bt_info = {}
        # pyre-fixme[4]: Attribute must be annotated.
        self.test_ids = []

    # pyre-fixme[3]: Return annotation cannot contain `Any`.
    def _preprocess(
        self,
        # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
        data: Union[List[TimeSeriesData], Dict[Any, TimeSeriesData]],
    ) -> Dict[Any, Dict[Any, TimeSeriesData]]:
        """Preprocessing for input time series, including two steps:
            1. truncate the data before the earliest_timestamp (if earliest_timestamp is None, then skip this step).
            2. filling missing values with NA with fill_missing_value_na function.

        Args:
            data: A list or a dictionary of :class:`kats.consts.TimeSeriesData` objects representing the data to be preprocessed.

        Returns:
            A list or a dictionary of :class:`kats.consts.TimeSeriesData` objects representing the data after preprocessing.
        """

        if isinstance(data, List):
            keys = np.arange(len(data))
        elif isinstance(data, Dict):
            keys = list(data.keys())
        else:
            msg = f"data should be either a list of a dictionary of TimeSeriesData but receives {type(data)}."
            logging.error(msg)
            raise ValueError(msg)

        ans = {}
        sasonality = self.params.seasonality
        freq = self.params.freq
        for i in keys:
            ts = data[i]
            if not isinstance(ts, TimeSeriesData):
                msg = f"Each element in data should be a TimeSeriesData but receives {type(ts)}."
                logging.error(msg)
                raise ValueError(msg)
            if self.earliest_timestamp is not None:
                ts = ts[ts.time >= self.earliest_timestamp]
            ts = fill_missing_value_na(ts, sasonality, freq)
            ans[i] = ts
        return ans

    # pyre-fixme[3]: Return annotation cannot contain `Any`.
    def _filter(
        self,
        # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
        data: Union[Dict[Any, TimeSeriesData], List[TimeSeriesData]],
        backtest_ts: Union[str, pd.Timestamp],
        mode: str = "train",
    ) -> Tuple[Dict[Any, TimeSeriesData], Dict[Any, TimeSeriesData]]:
        """
        Split input data sets into training set and validation set, and discard the time series are too short for training/testing.

        :Parameters:
        data: Union[Dict[Any, TimeSeriesData], List[TimeSeriesData]]
            A list of time series.
        backtest_ts: Union[str, pd.Timestamp]
            Backtest timestamp, i.e., we use data before backtest_ts as training data and data after backtest_ts as validation set.
        mode: str = "train",
            Mode of filter. If mode!="train", then we take the filter as for testing.

        :Returns:
        train_TSs: Dict[Any, TimeSeriesData]
            A dictionary of time series for training.
        valid_TSs: Dict[Any, TimeSeriesData]
            A dictionary of time series for validation.
        """

        n = len(data)
        # get last backtesting timestamp
        if not isinstance(backtest_ts, pd.Timestamp):
            try:
                backtest_ts = pd.Timestamp(backtest_ts)
            except Exception as e:
                msg = f"Fail to convert backtesting timestamp with error message {e}."
                logging.error(msg)
                raise ValueError(msg)
        last_timestamp = backtest_ts + self.max_back_test_timedelta

        train_length = (
            self.min_train_length if mode == "train" else self.min_test_length
        )
        valid_length = self.min_valid_length if mode == "train" else 0
        min_length = train_length + valid_length

        train_TSs = {}
        valid_TSs = {}
        # processing each TS
        if isinstance(data, List):
            keys = np.arange(n)
        elif isinstance(data, Dict):
            keys = list(data.keys())
        else:
            msg = f"Collection of TSs should be either list or dictionary but receives {type(data)}."
            logging.error(msg)
            raise ValueError(msg)
        for key in keys:
            ts = data[key]
            # extract data before last_timestamp
            if len(ts) > 0:
                ts = ts[ts.time <= last_timestamp]
                if len(ts) >= min_length:
                    train = ts[ts.time < backtest_ts]
                    valid = ts[ts.time >= backtest_ts]
                    if len(train) >= train_length and len(valid) >= valid_length:
                        train_TSs[key] = train
                        valid_TSs[key] = valid
        msg = f"Processed {n} TSs and filtered out {len(train_TSs)} TSs for training and {len(valid_TSs)} TSs for validation"
        logging.info(msg)

        if len(train_TSs) == 0:
            msg = f"Found 0 valid time series for backtesting timestamp {backtest_ts}, please use a valid backtesting timestamp."
            logging.error(msg)
            raise ValueError(msg)
        return train_TSs, valid_TSs

    def _fit_single_gm(
        self,
        gmmodel: GMModel,
        # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
        train_TSs: Dict[Any, TimeSeriesData],
        # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
        valid_TSs: Optional[Dict[Any, TimeSeriesData]] = None,
        random_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Fit a global model and return training information.

        :Parameters:
        gmparam: GMModel
            GMModel object.
        train_TSs: Dict[Any, TimeSeriesData]
            Training time series.
        valid_TSs: Dict[Any, TimeSeriesData]
            Validation time series.
        random_seed: Optional[int] = None
            Random seed.
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            # to ensure performance
            torch.set_num_threads(1)
        training_info = gmmodel.train(
            train_TSs, valid_TSs, fcst_monitor=False, debug=False
        )
        return training_info

    def run_backtest(self) -> pd.DataFrame:
        """Run backtester.

        Returns:
            A `panda.DataFrame` object representing the backtest errors.
        """
        data = self.data
        n = len(data)
        # train_test split
        keys = np.array(list(data.keys())) if isinstance(data, Dict) else np.arange(n)
        np.random.shuffle(keys)
        all_test_TSs = {keys[i]: data[keys[i]] for i in range(self.test_size)}
        all_train_TSs = {keys[i]: data[keys[i]] for i in range(self.test_size, n)}
        self.test_ids = keys[: self.test_size]
        bt_info = {}
        m = int(self.replicate * self.splits)
        num_core = np.min([m, self.max_core])

        evaluation_collects = []

        for bt in self.backtest_timestamp:
            bt_train_TSs, bt_valid_TSs = self._filter(all_train_TSs, bt, "train")
            bt_test_train_TSs, bt_test_valid_TSs = self._filter(
                all_test_TSs, bt, "test"
            )
            bt_info[bt] = {
                "num_train_TSs": len(bt_train_TSs),
                "num_test_TSs": len(bt_test_train_TSs),
            }

            split_data = split(self.splits, self.overlap, bt_train_TSs, bt_valid_TSs)

            if not self.multi:
                t0 = time.time()
                i = 0
                for _ in range(self.replicate):
                    for train, valid in split_data:
                        info = self._fit_single_gm(
                            self.gm_collects[bt][i], train, valid
                        )
                        self.gm_info_collects[bt].append(info)
                        i += 1
                logging.info(
                    f"fit {self.replicate*self.splits} gm time {time.time()-t0}"
                )
            else:
                t0 = time.time()
                rds = np.random.randint(1, 10000, m)
                model_params = [
                    (
                        self.gm_collects[bt][i],
                        split_data[i % self.splits][0],
                        split_data[i % self.splits][1],
                        rds[i],
                    )
                    for i in range(m)
                ]

                pool = Pool(num_core)
                results = pool.starmap(self._fit_single_gm, model_params)
                pool.close()
                pool.join()
                self.gm_info_collects[bt] = results

                logging.info(f"fit {m} gm time {time.time()-t0}")

            bt_eval = self._evaluate(
                self.gm_collects[bt], bt_test_train_TSs, bt_test_valid_TSs
            )
            bt_eval["backtest_ts"] = bt
            evaluation_collects.append(bt_eval)
            logging.info(f"Successfully finish backtest for {bt}.")

        self.bt_info = bt_info
        return pd.concat(evaluation_collects)

    def _evaluate(
        self,
        gm_collects: List[GMModel],
        # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
        bt_test_train_TSs: Dict[Any, TimeSeriesData],
        # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
        bt_test_valid_TSs: Dict[Any, TimeSeriesData],
    ) -> pd.DataFrame:
        """
        Get forecasting errors on test sets.

        :Parameters:
        gm_collects: List[GMModel]
            A list of trained global models.
        bt_test_train_TSs: Dict[Any, TimeSeriesData]
            A dictionary of time series for testing.
        bt_test_valid_TSs: Dict[Any, TimeSeriesData]
            A dictionary of the corresponding time series for forecasting evaluation.

        :Returns:
        pd.DataFrame
            A datafram storing forecasting errors.
        """
        eval_func = gm_collects[0].build_validation_function()

        steps = np.max([len(bt_test_valid_TSs[t]) for t in bt_test_valid_TSs])

        fcst_window = self.params.fcst_window
        fcst_all = [
            gm.predict(bt_test_train_TSs, steps=steps, raw=True) for gm in gm_collects
        ]
        n = len(gm_collects)
        ans = []
        for k in bt_test_valid_TSs:
            tmp = bt_test_valid_TSs[k].value.values
            tmp_step = len(tmp) // fcst_window + int(len(tmp) % fcst_window != 0)
            tmp_fcst_step = fcst_window * tmp_step
            actuals = np.full(tmp_fcst_step, np.nan, np.float)
            actuals[: len(tmp)] = tmp
            for j in range(tmp_step):
                tmp_actuals = actuals[j * fcst_window : (j + 1) * fcst_window]
                tmp_ans = [eval_func(fcst_all[i][k][j], tmp_actuals) for i in range(n)]
                [
                    t.update({"model_num": i, "step": j, "idx": k, "type": "single"})
                    for i, t in enumerate(tmp_ans)
                ]
                ans.extend(tmp_ans)
                ensemble_fcst = np.median(
                    np.column_stack(fcst_all[i][k][j] for i in range(n)), axis=1
                )
                evl = eval_func(ensemble_fcst, tmp_actuals)
                evl["step"] = j
                evl["type"] = "ensemble"
                evl["idx"] = k
                ans.append(evl)
        return pd.DataFrame(ans)


class GMBackTesterExpandingWindow(BackTesterExpandingWindow):
    """Backtest class for global models on a single time series.

    This class evaluates the performance of a global model (ensemble) on a given time series via backtesting. This class provides function run_backtest.

    Attributes:
        error_methods: A list of strings representing the names of the error metrics. Valid error metrics including "mape", "smape", "mae", "mase", "mse" and "rmse".
        data: A :class:`kats.consts.TimeSeriesData` object representing the time series data.
        gmobject: A :class:`kats.models.globalmodel.model.GMModel` object or a :class:`kats.models.globalmodel.ensemble.GMEnsemble` object to be evaluated.
        start_train_percentage: A float representing the initial percentage of data used for training.
        end_train_percentage: A float representing the final percentage of data used for training.
        test_percentage: A float representing the percentage of data used for testing.
        expanding_steps: An integer representing the number of expanding steps (i.e. number of folds).
        multi: Optional; A boolean representing whether to use multiprocessing. Default is True.

    Sample Usage:
        >>> gmtew = GMBackTesterExpandingWindow(['smape'], data, gme, start_train_percentage = 50, end_train_percentage = 90, test_percentage = 20, expanding_steps=5)
        >>> gmtew.run_backtest()
    """

    # pyre-fixme[3]: Return type must be annotated.
    def __init__(
        self,
        error_methods: List[str],
        data: TimeSeriesData,
        gmobject: Union[GMModel, GMEnsemble],
        start_train_percentage: float,
        end_train_percentage: float,
        test_percentage: float,
        expanding_steps: int,
        # pyre-fixme[2]: Parameter must be annotated.
        multi=True,
        # pyre-fixme[2]: Parameter must be annotated.
        **kwargs,
    ):
        if data.infer_freq_robust() != gmobject.params.freq:
            msg = (
                "Test data and the global model are of different frequency"
                f"(i.e., the frequency of data is {data.infer_freq_robust()} and the frequency of gmobject is {gmobject.params.freq})."
            )
            logging.error(msg)
            raise ValueError(msg)
        self.gmobject = gmobject
        data = fill_missing_value_na(
            data, seasonality=gmobject.params.seasonality, freq=gmobject.params.freq
        )

        super(GMBackTesterExpandingWindow, self).__init__(
            error_methods=error_methods,
            data=data,
            # pyre-fixme [6]: Expected `kats.models.model.Model[typing.Any]` for 3rd parameter `model_class` to call `BackTesterExpandingWindow.__init__` but got `None`.
            model_class=None,
            # pyre-fixme[6]: Expected `Params` for 4th param but got `None`.
            params=None,
            start_train_percentage=start_train_percentage,
            start_train_percentag=start_train_percentage,
            end_train_percentage=end_train_percentage,
            test_percentage=test_percentage,
            expanding_steps=expanding_steps,
            multi=multi,
        )

    def _create_ts(
        self,
        training_data_indices: Tuple[int, int],
        testing_data_indices: Tuple[int, int],
    ) -> Tuple[TimeSeriesData, TimeSeriesData]:
        """Split original time series into training and testing time series."""

        training_data_start, training_data_end = training_data_indices
        testing_data_start, testing_data_end = testing_data_indices
        logging.info("Creating TimeSeries train test objects for split")
        logging.info(
            "Train split of {0}, {1}".format(training_data_start, training_data_end)
        )
        logging.info(
            "Test split of {0}, {1}".format(testing_data_start, testing_data_end)
        )

        if (
            training_data_start < 0
            or training_data_start > self.size
            or training_data_end < 0
            or training_data_end > self.size
            or training_data_start >= training_data_end
        ):
            logging.error(
                "Train Split of {0}, {1} was invalid".format(
                    training_data_start, training_data_end
                )
            )
            raise ValueError("Invalid training data indices in split")

        if (
            testing_data_start < 0
            or testing_data_start > self.size
            or testing_data_end < 0
            or testing_data_end > self.size
            or testing_data_end <= testing_data_start
        ):
            logging.error(
                "Test Split of {0}, {1} was invalid".format(
                    testing_data_start, testing_data_end
                )
            )
            raise ValueError("Invalid testing data indices in split")

        training_data = TimeSeriesData(
            pd.DataFrame(
                {
                    "time": self.data.time[training_data_start:training_data_end],
                    "y": self.data.value[training_data_start:training_data_end],
                }
            )
        )

        testing_data = TimeSeriesData(
            pd.DataFrame(
                {
                    "time": self.data.time[testing_data_start:testing_data_end],
                    "y": self.data.value[testing_data_start:testing_data_end],
                }
            )
        )

        return training_data, testing_data

    def _build_ts_and_get_prediction(
        self, splits: Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]
    ) -> None:
        """Build training time series and get forecasts."""

        training_splits, testing_splits = splits
        num_splits = len(training_splits)

        if self.multi:
            pool = Pool(max(1, min(cpu_count() - 1, num_splits)))
            TSs = pool.starmap(
                self._create_ts, list(zip(training_splits, testing_splits))
            )
            pool.close()
            pool.join()

        else:
            TSs = [
                self._create_ts(training_splits[i], testing_splits[i])
                for i in range(num_splits)
            ]

        training_TSs = [t[0] for t in TSs]
        test_TSs = [t[1] for t in TSs]
        steps = int(np.max([len(t) for t in test_TSs]))

        fcsts = self.gmobject.predict(training_TSs, steps=steps)

        results = []

        # Format forecasting results for function calc_error().
        for i in range(num_splits):
            train_data = training_TSs[i].value.values
            train_data = train_data[~np.isnan(train_data)]
            truth = test_TSs[i].value.values
            # pyre-fixme [6]: Expected `_SupportsIndex` for 1st positional only parameter to call `list.__getitem__` but got `str`.
            fcst = fcsts[i]["fcst_quantile_0.5"].values[: len(truth)]
            fcst = fcst[~np.isnan(truth)]
            truth = truth[~np.isnan(truth)]
            results.append((train_data, truth, None, fcst))

        self.results = results

        return

    def run_backtest(self) -> None:
        """Function for running backtest."""

        self._build_ts_and_get_prediction(self._create_train_test_splits())
        self.calc_error()
