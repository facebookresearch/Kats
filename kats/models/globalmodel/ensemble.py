# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import time
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import torch
from kats.consts import TimeSeriesData
from kats.models.globalmodel.model import GMModel, gmparam_from_string
from kats.models.globalmodel.utils import GMParam, gmpreprocess, split


class GMEnsemble:
    """A class for building the global model ensemble.

    GMEnsemble is a framework for building the ensemble of global models. It provides functions including train, predict and save_model.

    Attributes:
        gmparam: A :class:`kats.models.globalmodel.utils.GMParam` object building the for global model ensemble.
        ensemble_type: Optional; A string representing the ensemble type. Can be 'median' or 'mean'. Default is 'median'.
        splits: Optional; An positive integer representing the number of sub-datasets to be built. Default is 3.
        overlap: Optional; A boolean representing whether or not sub-datasets overlap with each other or not. For example, we have samples [ts1, ts2, ts3] and splits = 3.
                 If overlap is True, then three subsets are [[ts1], [ts2], [ts3]], i.e., each sample only appears in one sub-dataset.
                 If overlap is False, then three subsets are [[ts1, ts2], [ts2, ts3], [ts3, ts1]], i.e., each sample appears in (splits-1) sub-datasets.
                 Default is True.
        replicate: Optional; A positive integer representing the number of global models to be trained on each sub-datasets. Default is 1.
        multi: Optional; A boolean representing whether or not to use multi-processing for training and prediction. Default is False.
        max_core: Optional; A positive integer representing the number of available cpu cores. Default is None, which sets the number of cores to (total_cores - 1) // 2.

    Sample Usage:
        >>> gme = GMEnsemble(params)
        >>> # train an ensemble object and get training info (e.g., training/validation losses)
        >>> training_info = gme.train(train_TSs, valid_TSs)
        >>> # make prediction
        >>> gme.predict(train_TSs)
        >>> # save model
        >>> gme.save_model("global_model_ensemble.pickle")
        >>> # Evalute model performance on a given dataset.
        >>> evals = gme.evalute(test_train, test_test)
    """

    def __init__(
        self,
        gmparam: GMParam,
        ensemble_type: str = "median",
        splits: int = 3,
        overlap: bool = True,
        replicate: int = 1,
        multi: bool = False,
        max_core: Optional[int] = None,
    ) -> None:

        if not isinstance(gmparam, GMParam):
            msg = f"gmparam should be GMParam object but receives {type(gmparam)}."
            logging.error(msg)
            raise ValueError(msg)
        self.params = gmparam
        if ensemble_type == "median":
            # pyre-fixme[4]: Attribute must be annotated.
            self._ensemble_func = np.median
        elif ensemble_type == "mean":
            self._ensemble_func = np.mean
        else:
            msg = f"ensemble_type should be either 'mean' or 'median' but receives {ensemble_type}."
            logging.error(msg)
            raise ValueError(msg)
        self.ensemble_type = ensemble_type

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

        self.model_num = int(self.replicate * self.splits)

        self.multi = multi

        total_cores = cpu_count()
        if max_core is None:
            # pyre-fixme[4]: Attribute must be annotated.
            self.max_core = max((total_cores - 1) // 2, 1)
        elif isinstance(max_core, int) and max_core > 0 and max_core <= total_cores:
            self.max_core = max_core
        else:
            msg = f"max_core should be a positive integer in [1, {total_cores}] but receives {max_core}."
            logging.error(msg)
            raise ValueError(msg)
        # pyre-fixme[4]: Attribute must be annotated.
        self.gm_info = []
        # pyre-fixme[4]: Attribute must be annotated.
        self.gm_models = [GMModel(self.params) for _ in range(self.model_num)]
        # pyre-fixme[4]: Attribute must be annotated.
        self.test_ids = []

    def _fit_single_gm(
        self,
        gm: GMModel,
        # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
        train_TSs: Dict[Any, TimeSeriesData],
        # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
        valid_TSs: Optional[Dict[Any, TimeSeriesData]],
        random_seed: Optional[int] = None,
        # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
        test_train_TSs: Optional[Dict[Any, TimeSeriesData]] = None,
        # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
        test_valid_TSs: Optional[Dict[Any, TimeSeriesData]] = None,
    ) -> Dict[str, Any]:
        """Fit a global model and return training information.

        Args:
            gmparam: A GMParam object for global model.
            train_TSs: A dictionary representing the training time series.
            valid_TSs: A dictionary representing the corresponding validation time series.
            random_seed: Optional; An integer representing the random seed. Default is None, i.e., no random seed is set.
            test_train_TSs: Optional; A dictionary representing the training part of the test time series. Default is None.
            test_test_TSs: Optional; A dictionary representing the testing part of the test time series. Default is None.

        Returns:
            gm: A :class:`kats.models.globalmodel.model.GMModel` object representing the trained global model.
            info: A dictionary representing the training information of the global model.
        """

        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            # to ensure performance
            torch.set_num_threads(1)
        training_info = gm.train(
            train_TSs,
            valid_TSs,
            test_train_TSs,
            test_valid_TSs,
            fcst_monitor=False,
            debug=False,
        )
        return training_info

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def _predict_single_gm(self, gm, test_TSs, steps, test_batch_size=1000):
        t = time.time()
        fcst = gm.predict(
            test_TSs, steps=steps, raw=True, test_batch_size=test_batch_size
        )
        logging.info(f"fcst {len(fcst)} TSs with {time.time()-t}.")
        return fcst

    def train(
        self,
        # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
        data: Union[Dict[Any, TimeSeriesData], List[TimeSeriesData]],
        test_size: float = 0.1,
        valid_set: bool = False,
    ) -> None:
        """Train base global models.

        Args:
            data: A list or a dictionary of time series.
            test_size: Optional; A float in [0,1) representing the percentage that the test set takes up. Default is 0.1
            valid_set: Optional; A boolean specifying whether or not to have a validation set during training. Default is False.
        """

        n = len(data)
        keys = np.array(list(data.keys())) if isinstance(data, dict) else np.arange(n)

        if test_size < 0 or test_size > 1:
            msg = f"test_size should be in [0,1) but receives {test_size}."
            logging.error(msg)
            raise ValueError(msg)
        if test_size > 0:
            m = max(1, int(n * test_size))
            np.random.shuffle(keys)
            all_test_TSs = {keys[i]: data[keys[i]] for i in range(m)}
            test_train_TSs, test_valid_TSs = gmpreprocess(
                self.params, all_test_TSs, mode="test"
            )
            all_train_TSs = {keys[i]: data[keys[i]] for i in range(m, n)}
            train_TSs, valid_TSs = gmpreprocess(
                self.params, all_train_TSs, mode="train", valid_set=valid_set
            )
            self.test_ids = list(test_train_TSs.keys())
        else:
            train_TSs, valid_TSs = gmpreprocess(
                self.params, data, mode="train", valid_set=valid_set
            )
            test_train_TSs, test_valid_TSs = None, None
            self.test_ids = []

        split_data = split(self.splits, self.overlap, train_TSs, valid_TSs)
        # multi processing
        if self.multi:

            t0 = time.time()
            rds = np.random.randint(1, int(10000 * self.model_num), self.model_num)
            model_params = [
                (
                    self.gm_models[i],
                    split_data[i % self.splits][0],
                    split_data[i % self.splits][1],
                    rds[i],
                    test_train_TSs,
                    test_valid_TSs,
                )
                for i in range(self.model_num)
            ]
            pool = Pool(self.max_core)
            results = pool.starmap(self._fit_single_gm, model_params)
            pool.close()
            pool.join()
            # return results
            self.gm_info = results
            logging.info(
                f"fit {self.model_num} global models using time {time.time()-t0}"
            )
        else:
            self.gm_info = []
            t0 = time.time()
            i = 0
            for _ in range(self.replicate):
                for train, valid in split_data:
                    info = self._fit_single_gm(
                        self.gm_models[i],
                        train,
                        valid,
                        test_train_TSs=test_train_TSs,
                        test_valid_TSs=test_valid_TSs,
                    )
                    self.gm_info.append(info)
                    i += 1
            logging.info(
                f"fit {self.model_num} global models using time {time.time()-t0}"
            )
        return

    # pyre-fixme[3]: Return annotation cannot contain `Any`.
    def _combine_fcst(
        self,
        # pyre-fixme[2]: Parameter annotation cannot be `Any`.
        idx: Any,
        fcsts: List[np.ndarray],
        steps: int,
        raw: bool,
        first_timestamp: Optional[pd.Timestamp] = None,
        col_names: Optional[List[str]] = None,
    ) -> Tuple[Any, Any]:
        """Combine the forecasts from each global model."""

        fcst = [
            self._ensemble_func([fcsts[i][j] for i in range(len(fcsts))], axis=0)
            for j in range(len(fcsts[0]))
        ]
        if raw:
            return idx, fcst
        else:
            n_quantile = len(self.params.quantile)

            df = pd.DataFrame(
                np.column_stack([t.reshape(n_quantile, -1) for t in fcst]).T, copy=False
            ).iloc[:steps]

            df.columns = col_names
            df["time"] = pd.date_range(
                first_timestamp + self.params.freq, periods=steps, freq=self.params.freq
            )
            return idx, df

    # pyre-fixme[3]: Return annotation cannot contain `Any`.
    def predict(
        self,
        # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
        test_TSs: Union[
            TimeSeriesData, List[TimeSeriesData], Dict[Any, TimeSeriesData]
        ],
        steps: int,
        test_batch_size: int = 500,
        raw: bool = False,
    ) -> Dict[Any, Union[pd.DataFrame, List[np.ndarray]]]:
        """Generate forecasts for the target time series.

        Args:
            test_TSs: A TimeSeriesDdata object, list or a dictionary of time series to generate forecasts for.
            steps: An integer representing the forecast steps.
            test_batch_size: Optional; An integer representing the batch size for testing. Default is 500.
            raw: Optional; A boolean representing whether or not to return raw forecasts (i.e., `numpy.ndarray` objects). If False, the forecasts are `pandas.DataFrame` objects. Default is False.

        Returns:
            A dictionary of forecasts, whose keys are the ids for time series, and values are the corresponding forecasts.
        """

        if isinstance(test_TSs, TimeSeriesData):
            test_TSs = [test_TSs]
        elif isinstance(test_TSs, dict) or isinstance(test_TSs, list):
            pass
        else:
            msg = f"predict function only accepts a TimeSeriesData object, a dictionary or a list of TimeSeriesData objects, but receives {type(test_TSs)}"

        if steps <= 0:
            msg = f"step should be a positive integer but receives {steps}."
            logging.error(msg)
            raise ValueError(msg)

        if not isinstance(test_batch_size, int) or test_batch_size <= 0:
            msg = f"test_batch_size should be a positive integer but receives {test_batch_size}."
            logging.error(msg)
            raise ValueError(msg)
        t0 = time.time()
        if self.multi:
            pool = Pool(self.max_core)
            all_fcsts = pool.starmap(
                self._predict_single_gm,
                [(t, test_TSs, steps, test_batch_size) for t in self.gm_models],
            )
            pool.close()
            pool.join()
        else:
            all_fcsts = [
                m.predict(test_TSs, steps, raw=True, test_batch_size=test_batch_size)
                for m in self.gm_models
            ]
        logging.info(
            f"time for all global model to generate forecasts: {time.time() - t0}."
        )

        keys = (
            test_TSs.keys() if isinstance(test_TSs, dict) else np.arange(len(test_TSs))
        )

        col_names = (
            [f"fcst_quantile_{q}" for q in self.params.quantile] if (not raw) else None
        )
        if self.multi:
            cf_params = [
                (
                    k,
                    [all_fcsts[i][k] for i in range(self.model_num)],
                    steps,
                    raw,
                    test_TSs[k].time.iloc[-1],
                    col_names,
                )
                for k in keys
            ]
            pool = Pool(self.max_core)
            results = pool.starmap(self._combine_fcst, cf_params)
            pool.close()
            pool.join()
            return {t[0]: t[1] for t in results}
        else:
            ans = {}
            for k in keys:
                try:
                    ans[k] = self._combine_fcst(
                        k,
                        [all_fcsts[i][k] for i in range(self.model_num)],
                        steps,
                        raw,
                        test_TSs[k].time.iloc[-1],
                        col_names,
                    )[1]
                except Exception as e:
                    msg = f"Fail to generate forecasts with Exception {e}."
                    logging.error(msg)
                    raise ValueError(msg)
            return ans

    def save_model(self, file_name: str) -> None:
        """Save ensemble model to file.

        Args:
            file_name: A string representing the file address and file name.
        """

        if len(self.gm_models) == 0:
            msg = "Please train global models before saving GMEnsemble."
            logging.error(msg)
            raise ValueError(msg)
        try:
            # clean-up unnecessary info
            [gm._reset_nn_states() for gm in self.gm_models]
            state_dict = (
                [gm.rnn.state_dict() for gm in self.gm_models]
                if self.params.model_type == "rnn"
                else None
            )
            encoder_dict = (
                [gm.encoder.state_dict() for gm in self.gm_models]
                if self.params.model_type == "s2s"
                else None
            )
            decoder_dict = (
                [gm.decoder.state_dict() for gm in self.gm_models]
                if self.params.model_type == "s2s"
                else None
            )
            gmparam_string = self.params.to_string()
            info = {
                "state_dict": state_dict,
                "encoder_dict": encoder_dict,
                "decoder_dict": decoder_dict,
                "gmparam_string": gmparam_string,
                "gm_info": self.gm_info,
                "test_ids": self.test_ids,
                "gmensemble_params": {},
            }
            for attr in [
                "splits",
                "overlap",
                "replicate",
                "multi",
                "max_core",
                "ensemble_type",
            ]:
                info["gmensemble_params"][attr] = getattr(self, attr)
            with open(file_name, "wb") as f:
                joblib.dump(info, f)
            logging.info(f"Successfully save GMEnsemble to {file_name}.")
        except Exception as e:
            msg = f"Fail to save GMEnsemble to {file_name} with Exception {e}."
            logging.error(msg)
            raise ValueError(msg)

    def evaluate(
        self,
        # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
        test_train_TSs: Union[
            TimeSeriesData, List[TimeSeriesData], Dict[Any, TimeSeriesData]
        ],
        # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
        test_valid_TSs: Union[
            TimeSeriesData, List[TimeSeriesData], Dict[Any, TimeSeriesData]
        ],
    ) -> pd.DataFrame:
        """Evaluate the GMEnsemble object performance.

        A wrapper function to evaluate model performance on a given time series data set.

        Args:
            test_train_TSs: A list or a dictionary of :class:`kats.consts.TimeSeriesData` objects for warming-ups.
            test_valid_TSs: A list or a dictionary of :class:`kats.consts.TimeSeriesData` objects for evaluation.

        Returns:
            A `pandas.DataFrame` object representing the evaluation results.
        """

        if type(test_train_TSs) != type(test_valid_TSs):
            msg = (
                "The data type of test_train_TSs and test_valid_TSs should be the same."
            )
            logging.error(msg)
            raise ValueError(msg)

        if isinstance(test_train_TSs, TimeSeriesData):
            test_train_TSs = [test_train_TSs]
            # pyre-fixme[9]
            test_valid_TSs = [test_valid_TSs]

        if len(test_train_TSs) != len(test_valid_TSs):
            msg = "test_train_TSs and test_valid_TSs should be of the same length."
            logging.error(msg)
            raise ValueError(msg)
        keys = (
            test_train_TSs.keys()
            if isinstance(test_train_TSs, dict)
            else range(len(test_train_TSs))
        )
        if len(keys) == 0:
            msg = "The input collection of time series should not be empty."
            logging.error(msg)
            raise ValueError(msg)

        steps = np.max([len(test_valid_TSs[t]) for t in keys])

        fcst = self.predict(test_train_TSs, steps=steps, raw=True)
        logging.info(
            f"Successfully generate forecasts for all test time series with length {steps}."
        )
        eval_func = self.gm_models[0].build_validation_function()
        fcst_window = self.params.fcst_window
        ans = []
        keys = (
            test_train_TSs.keys()
            if isinstance(test_train_TSs, dict)
            else range(len(test_train_TSs))
        )
        for k in keys:
            tmp = test_valid_TSs[k].value.values
            tmp_step = len(tmp) // fcst_window + int(len(tmp) % fcst_window != 0)
            tmp_fcst_length = tmp_step * fcst_window
            actuals = np.full(tmp_fcst_length, np.nan, np.float)
            actuals[: len(tmp)] = tmp
            for j in range(tmp_step):
                tmp_actuals = actuals[j * fcst_window : (j + 1) * fcst_window]
                tmp = eval_func(fcst[k][j], tmp_actuals)
                tmp["step"] = j
                tmp["idx"] = k
                ans.append(tmp)
        return pd.DataFrame(ans, copy=False)


def load_gmensemble_from_file(file_name: str) -> GMEnsemble:
    """Load a trained :class:`GMEnsemble` object from file.

    Args:
        file_name: A string representing the file saving the :class:`GMEnsemble` object.

    Returns:
        A :class:`GMEnsemble` object loaded from the file.
    """

    try:
        info = joblib.load(open(file_name, "rb"))
        gmparam = gmparam_from_string(info["gmparam_string"])
        n = (
            len(info["state_dict"])
            if info["state_dict"] is not None
            else len(info["encoder_dict"])
        )
        gm_models = []
        for i in range(n):
            tmp_gmmodel = GMModel(gmparam)
            if gmparam.model_type == "rnn":
                tmp_gmmodel.build_rnn()
                tmp_gmmodel.rnn.load_state_dict(info["state_dict"][i])
            else:
                tmp_gmmodel.build_s2s()
                tmp_gmmodel.encoder.load_state_dict(info["encoder_dict"][i])
                tmp_gmmodel.decoder.load_state_dict(info["decoder_dict"][i])
            gm_models.append(tmp_gmmodel)
        info["gmensemble_params"]["gmparam"] = gmparam
        # ensure max_core parameter dose not mess-up model reuse.
        info["gmensemble_params"]["max_core"] = int(
            min(info["gmensemble_params"]["max_core"], cpu_count())
        )
        gmensemble = GMEnsemble(**info["gmensemble_params"])
        gmensemble.gm_models = gm_models
        gmensemble.gm_info = info["gm_info"]
    except Exception as e:
        msg = f"Fail to load GMEnsemble from {file_name} with Exception {e}."
        logging.error(msg)
        raise ValueError(msg)
    return gmensemble
