# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
MetaDetectHptSelect is a meta learner that predict best hyper parameters of chosen detection algorithm given a time series features
before predicting, user needs to train the model:
    data_x=time_series features
    data_y=best (observed) hyper parameters for those features
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kats.consts import TimeSeriesData
from kats.detectors.detector import DetectorModel
from kats.detectors.meta_learning.exceptions import (
    KatsDetectorHPTIllegalHyperParameter,
    KatsDetectorHPTModelUsedBeforeTraining,
    KatsDetectorHPTTrainError,
    KatsDetectorsUnimplemented,
)
from kats.detectors.meta_learning.metalearning_detection_model import (
    metadata_detect_preprocessor,
    PARAMS_TO_SCALE_DOWN,
)
from kats.models.metalearner.metalearner_hpt import MetaLearnHPT
from kats.tsfeatures.tsfeatures import TsFeatures


_log: logging.Logger = logging.getLogger(__name__)


def get_ts_features(ts: TimeSeriesData) -> Dict[str, float]:
    """
    Extract TSFeatures for the input time series.
    """
    # Run Kats TsFeatures
    ts_features = TsFeatures(hw_params=False)
    feats = ts_features.transform(ts)

    # Rounding features
    features = {}
    assert isinstance(feats, dict)
    for feature_name, feature_val in feats.items():
        if not math.isnan(feature_val):
            feature_val = format(feature_val, ".4f")
        features[feature_name] = feature_val

    return features


def metadata_detect_reader(
    rawdata: pd.DataFrame,
    algorithm_name: str,
    params_to_scale_down: Set[str] = PARAMS_TO_SCALE_DOWN,
) -> Dict[str, pd.DataFrame]:
    """
    A helper function:
    preprocess meta data table to get the input for class MetaDetectHptSelect
    """
    # table's format: [{"hpt_res":..., "features":..., "best_model":...}, ...]
    table = metadata_detect_preprocessor(
        rawdata=rawdata, params_to_scale_down=params_to_scale_down
    )

    metadata = {}
    metadata["data_x"] = pd.DataFrame([item["features"] for item in table])
    metadata["data_y"] = pd.DataFrame(
        [pd.Series(item["hpt_res"][algorithm_name][0]) for item in table]
    )

    return metadata


@dataclass
class NNParams:
    """
    A dataclass that saves hyper-parameters for training neural networks

    scale: A boolean to specify whether or not to normalize time series features to zero mean and unit variance. Default is False.
    loss_scale: A float to specify the hyper-parameter to scale regression loss and classification loss,
        which controls the trade-off between the accuracy of regression task and classification task.
        A larger loss_scale value gives a more accurate prediction for classification part, and a lower value gives
        a more accurate prediction for regression part. Default is 1.0.
    lr: A float for learning rate. Default is 0.001.
    n_epochs: An integer for the number of epochs. Default is 1000.
    batch_size: An integer for the batch size. Default is 128.
    method: A string for the name of optimizer. Can be 'SGD' or 'Adam'. Default is 'SGD'.
    val_size: A float for the proportion of validation set of. It should be within (0, 1). Default is 0.1.
    momentum: A fload for the momentum for SGD. Default value is 0.9.
    n_epochs_stop:An integer or a float for early stopping condition. If the number of epochs is larger than n_epochs_stop
        and there is no improvement on validation set, we stop training. One can turn off the early stopping feature by
        setting n_epochs_stop = np.inf. Default is 20.

    """

    scale: bool = False
    loss_scale: float = 1.0
    lr: float = 0.001
    n_epochs: int = 1000
    batch_size: int = 128
    method: str = "SGD"
    val_size: float = 0.1
    momentum: float = 0.9
    n_epochs_stop: Union[int, float] = 20


class MetaDetectHptSelect:
    """A class for meta-learning framework on hyper-parameters tuning for detection algorithm.

    Attributes:
        data_x: A `pandas.DataFrame` object of time series features. data_x should not be None unless load_model is True. Default is None.
        data_y: A `pandas.DataFrame` object of the corresponding best hyper-parameters. data_y should not be None unless load_model is True. Default is None.
        detector_model: Type[DetectorModel]. A detector model in Kats.

    Sample Usage:
        >>> mdhs = MetaDetectHptSelect(data_x=datax, data_y=datay, detector_model=CUSUMDetectorModel)
        >>> mdhs.train(
                num_idx=["scan_window"],
                cat_idx=["threshold_low", "threshold_high"],
                n_hidden_shared=[20],
                n_hidden_cat_combo=[[5], [5]],
                n_hidden_num=[5],
                nnparams=self.nnparams,
            )  # train NNs
        >>> mdhs.plot()  # plot loss path
        >>> # generate prediction for future TSs
        >>> res = mdhs.get_hpt_from_features(np.random.normal(0, 1, [10, 40]))
    """

    def __init__(
        self,
        data_x: pd.DataFrame,
        data_y: pd.DataFrame,
        detector_model: Type[DetectorModel],
    ) -> None:
        self.detector_model: Type[DetectorModel] = detector_model
        self._check_valid_input(data_x, data_y, self.detector_model)
        self._data_x: pd.DataFrame = data_x
        self._data_y: pd.DataFrame = data_y
        self._algorithm_name: str = self.detector_model.__name__
        self._meta_hpt_model: Optional[MetaLearnHPT] = None

        # remove parameters which are constants
        # this causes the neural network training to fail otherwise
        self.const_params_dict: Dict[str, Any] = self.get_const_params(self._data_y)
        label_cols = [
            x for x in self._data_y.columns if x not in self.const_params_dict.keys()
        ]
        self._data_y = self._data_y[label_cols]
        msg = """
            Removed parameters (columns) in data_y which are constants.
            Otherwise this causes the neural network training to fail.
        """
        _log.warning(msg=msg)

    @staticmethod
    def _check_valid_input(
        data_x: pd.DataFrame,
        data_y: pd.DataFrame,
        detector_model: Type[DetectorModel],
    ) -> None:
        # check: are all hyper parameters valid for the given algorithm
        for idx, hp in data_y.iterrows():
            try:
                # pyre-fixme[6]
                MetaDetectHptSelect._init_detection_model(detector_model, hp.to_dict())
            except Exception as e:
                raise KatsDetectorHPTIllegalHyperParameter(
                    detector_model.__name__, idx, hp.to_dict(), e
                )

    @staticmethod
    def _init_detection_model(
        detector_model: Type[DetectorModel], hyper_parameters: Dict[str, Any]
    ) -> None:
        """
        returns detection algorithm for given algorithm name initialized with given hyper parameters
        """
        hpt_to_remove = [
            "threshold_low",
            "threshold_high",
            "detection_window_sec",
            "fraction",
            "direction_1d",
        ]

        hpt_algo = {k: v for k, v in hyper_parameters.items() if k not in hpt_to_remove}

        # Try initializing the detector model
        # pyre-fixme[45]
        _ = detector_model(**hpt_algo)

    @staticmethod
    def get_const_params(df: pd.DataFrame) -> Dict[str, Any]:
        num_values_dict = df.nunique().to_dict()

        assert num_values_dict is not None and isinstance(num_values_dict, dict)
        const_params = [key for key in num_values_dict if num_values_dict[key] == 1]

        const_params_dict = {}
        for param in const_params:
            const_params_dict[param] = df[param].iloc[0]
        return const_params_dict

    def train(
        self,
        num_idx: List[str],
        cat_idx: List[str],
        n_hidden_shared: List[int],
        n_hidden_cat_combo: List[List[int]],
        n_hidden_num: List[int],
        nnparams: Optional[NNParams] = None,
    ) -> None:
        """Build a multi-task neural network and train it.

        This function builds a multi-task neural network according to given neural network structure
        (i.e., n_hidden_shared, n_hidden_cat_combo, n_hidden_num).

        Args:
            num_idx: A list of names of numerical parameters in the given data_y.
            cat_idx: A list of names of categorical parameters in the given data_y.
            n_hidden_shared: A list of numbers of hidden neurons in each shared hidden layer.
            n_hidden_cat_combo: A list of lists of task-specific hidden layers' sizes of each categorical response variables.
            n_hidden_num: A list of task-specific hidden layers' sizes of numerical response variables.
            nnparams: hyper-parameters used in training neural networks.
        """
        const_num_provided = set(num_idx) & set(self.const_params_dict.keys())
        const_cat_provided = set(cat_idx) & set(self.const_params_dict.keys())

        if len(const_num_provided) > 0:
            raise ValueError(
                f"{const_num_provided} is constant value in data_y, and we have removed that column."
            )

        if len(const_cat_provided) > 0:
            raise ValueError(
                f"{const_cat_provided} is constant value in data_y, and we have removed that column."
            )

        # n_hiddent cat_combo should have same dimensions as number
        # of categorical variables. Else, MetaLearnHPT throws an error
        # this is often difficult to set beforehand, since the data
        # may not choose different value for a param, during best HPT
        # calculation. Hence this check prevents the hpt training from
        # failing
        if len(n_hidden_cat_combo) != len(cat_idx):
            _log.info("Resetting HPT network to match cat dimensions")
            n_hidden_cat_combo = [[5]] * len(cat_idx)

        if nnparams is None:
            nnparams = NNParams()

        hpt_model = MetaLearnHPT(
            data_x=self._data_x,
            data_y=self._data_y,
            categorical_idx=cat_idx,
            numerical_idx=num_idx,
            scale=nnparams.scale,
        )

        hpt_model.build_network(
            n_hidden_shared=n_hidden_shared,
            n_hidden_cat_combo=n_hidden_cat_combo,
            n_hidden_num=n_hidden_num,
        )

        try:
            hpt_model.train(
                loss_scale=nnparams.loss_scale,
                lr=nnparams.lr,
                n_epochs=nnparams.n_epochs,
                batch_size=nnparams.batch_size,
                method=nnparams.method,
                val_size=nnparams.val_size,
                momentum=nnparams.momentum,
                n_epochs_stop=nnparams.n_epochs_stop,
            )
        except Exception as e:
            raise KatsDetectorHPTTrainError(e)

        self._meta_hpt_model = hpt_model

    def plot(self, **kwargs: Any) -> Sequence[plt.Axes]:
        model = self._meta_hpt_model
        if model is None:
            raise KatsDetectorHPTModelUsedBeforeTraining()
        return model.plot(**kwargs)

    def get_hpt_from_ts(self, ts: TimeSeriesData) -> pd.DataFrame:
        features_dict = get_ts_features(ts)
        features_array = np.asarray(features_dict.values())
        return self.get_hpt_from_features(features_array)

    def get_hpt_from_features(
        self, source_x: Union[np.ndarray, List[np.ndarray], pd.DataFrame]
    ) -> pd.DataFrame:
        assert self._meta_hpt_model is not None

        ans = self._meta_hpt_model.pred_by_feature(source_x)
        ans_df = pd.DataFrame(ans)

        # add the const params that were not predicted by the neural net back
        for k, v in self.const_params_dict.items():
            ans_df[k] = v

        return ans_df

    def save_model(self) -> None:
        raise KatsDetectorsUnimplemented()

    def load_model(self) -> None:
        raise KatsDetectorsUnimplemented()
