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
    change_dtype,
    change_str_to_dict,
    NUM_SECS_IN_DAY,
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

    rawdata_features = rawdata["features"].map(change_str_to_dict)
    rawdata_features = rawdata_features.map(change_dtype)
    rawdata_hpt_res = rawdata["hpt_res"].map(change_str_to_dict)

    metadata = {}
    metadata["data_x"] = pd.DataFrame(rawdata_features.tolist())

    algorithm_names = (
        rawdata_hpt_res.map(lambda kv: list(kv.keys())).explode().unique().tolist()
    )

    metadata["data_y_dict"] = {}
    for a in algorithm_names:
        metadata["data_y_dict"][a] = (
            rawdata_hpt_res.map(lambda kv: kv[a][0])
            .map(
                lambda kv: {
                    k: v if k not in params_to_scale_down else v / NUM_SECS_IN_DAY
                    for k, v in kv.items()
                }
            )
            .apply(pd.Series)  # expend dict to columns
            .convert_dtypes(convert_integer=False)
            .reset_index(drop=True)
        )

    return {
        "data_x": metadata["data_x"].copy(),
        "data_y": metadata["data_y_dict"][algorithm_name].copy(),
    }


class MetaDetectHptSelect:
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
        scale: bool = False,
        loss_scale: float = 1.0,
        lr: float = 0.001,
        n_epochs: int = 1000,
        batch_size: int = 128,
        method: str = "SGD",
        val_size: float = 0.1,
        momentum: float = 0.9,
        n_epochs_stop: Union[int, float] = 20,
    ) -> None:
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

        hpt_model = MetaLearnHPT(
            data_x=self._data_x,
            data_y=self._data_y,
            categorical_idx=cat_idx,
            numerical_idx=num_idx,
            scale=scale,
        )

        hpt_model.build_network(
            n_hidden_shared=n_hidden_shared,
            n_hidden_cat_combo=n_hidden_cat_combo,
            n_hidden_num=n_hidden_num,
        )

        try:
            hpt_model.train(
                loss_scale=loss_scale,
                lr=lr,
                n_epochs=n_epochs,
                batch_size=batch_size,
                method=method,
                val_size=val_size,
                momentum=momentum,
                n_epochs_stop=n_epochs_stop,
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
