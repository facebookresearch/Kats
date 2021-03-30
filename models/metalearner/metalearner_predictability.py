#!/usr/bin/env python3

import ast
import logging
from collections import Counter
from typing import Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from infrastrategy.kats.consts import TimeSeriesData
from infrastrategy.kats.tsfeatures.tsfeatures import TsFeatures
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


class MetaLearnPredictability:
    """
    Meta learning model to predict whether a time series is predictable or not (
        we define the time series with error metrics less than a user defined threshold as predictable).

    This framework uses classification algorithms with time series features.

    After training a classifier, it can predict whether a time series is predictable or not.

    :Parameters:
    metadata: Optional[List]
        A list of Dict[str, Any] (or None if one wants to load a trained MetaLearnPredictablity model).
        The dict is meta data from outputs of class GetMetaData, and it must contain 3 components, hpt_res, features and best_model.
        hpt_res represents best hyper-params for each candidate model and its corresponding error, features are time series features,
        and best_model means the best candidate model for a given time series data.

    threshold: float
        Threshold for the error metric that decides whether a time series is predictable or not. The time series with a error metric higher
        than the threshold is considered as unpredictable.

    load_model: bool
        Whether one wants to load a trained model from a saved file.
        If so, then initial metadata list is not necessary and data check will not be performed.

    :Example:
    >>> from infrastrategy.kats.models.metalearner.metalearner_predictability import MetaLearnPredictablity
    >>> # create an object
    >>> mlp = MetaLearnPredictability(data)
    >>> # train a model
    >>> mlp.train()
    >>> # save the trained model
    >>> mlp.save_model()
    >>> # create a new object to load the trained model
    >>> mlp2 = MetaLearnPredictability(load_model=True)
    >>> mlp2.load_model()
    >>> # predict a time series
    >>> mlp.pred(TSdata)
    >>> # predict using time series features
    >>> mlp.pred_by_feature(TSfeatures)


    :Methods:
    """

    def __init__(
        self, metadata: Optional[List] = None, threshold: float = 0.2, load_model=False
    ) -> None:
        if load_model:
            msg = "Initialize this class without meta data, and a pretrained model should be loaded using .load_model() method."
            logging.info(msg)
        else:
            if metadata is None:
                msg = "Please input meta data to initialize this class."
                logging.error(msg)
                raise ValueError(msg)
            if len(metadata) <= 30:
                msg = "Dataset is too small to train a meta learner!"
                logging.error(msg)
                raise ValueError(msg)

            if "hpt_res" not in metadata[0]:
                msg = "Missing best hyper-params, not able to train a meta learner!"
                logging.error(msg)
                raise ValueError(msg)

            if "features" not in metadata[0]:
                msg = "Missing time series features, not able to train a meta learner!"
                logging.error(msg)
                raise ValueError(msg)

            if "best_model" not in metadata[0]:
                msg = "Missing best models, not able to train a meta learner!"
                logging.error(msg)
                raise ValueError(msg)

            self.metadata = metadata
            self.threshold = threshold
            self._reorganize_data()
            self._validate_data()
            self.rescale = False
            self.clf = None
            self.clf_threshold = None

    def _reorganize_data(self) -> None:
        """
        Reorganize raw input data into features and labels
        """
        metadata = self.metadata

        self.features = []
        self.labels = []

        for i in range(len(metadata)):
            try:
                if isinstance(metadata[i]["hpt_res"], str):
                    hpt = ast.literal_eval(metadata[i]["hpt_res"])
                else:
                    hpt = metadata[i]["hpt_res"]

                if isinstance(metadata[i]["features"], str):
                    feature = ast.literal_eval(metadata[i]["features"])
                else:
                    feature = metadata[i]["features"]

                self.features.append(feature)
                self.labels.append(hpt[metadata[i]["best_model"]][1])
            except Exception as e:
                logging.exception(e)
        self.labels = (np.array(self.labels) > self.threshold).astype(int)
        self.features = pd.DataFrame(self.features)
        return

    def _validate_data(self) -> None:
        """
        Validate input data and we check two aspects:
            1) whether input data contain both positive and negative instances.
            2) whether training data size is at least 30.
        """
        if len(np.unique(self.labels)) == 1:
            msg = "Only one type of time series data and cannot train a classifier!"
            logging.error(msg)
            raise ValueError(msg)
        if len(self.features) <= 30:
            msg = "Dataset is too small to train a meta learner!"
            logging.error(msg)
            raise ValueError(msg)

    def count_category(self) -> Dict[int, int]:
        """
        Count number of positive and negative instances.
        """
        return Counter(self.labels)

    def preprocess(self) -> None:
        """
        Rescale the input-features.
        """
        self.rescale = True
        self.features_mean = np.average(self.features.values, axis=0)
        self.features_std = np.std(self.features.values, axis=0)

        self.features_std[self.features_std == 0] = 1.0

        features = (self.features.values - self.features_mean) / self.features_std
        self.features = pd.DataFrame(features, columns=self.features.columns)

    def train(
        self,
        method: str = "RandomForest",
        valid_size: float = 0.1,
        test_size: float = 0.1,
        recall_threshold: float = 0.7,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Train a classifier with time series features to forecast predictability.

        :Parameters:
        method: str
            Classification algorithm. Default algorithm is RandomForest. Currently we support RandomForest, GBDT, KNN and NaiveBayes.
        valid_size: float
            Size of validation set for parameter tunning and should be within (0, 1).
        test_size: float
            Size of test set and should be within [0., 1-valid_size) (if not, test_size will be set as 0).
        recall_threshold: float
            Control the recall score on the classifier. We tune the classifier by selecting the one with recall score
            no less than recall_threshold on the validation set.

        :Returns:
        A dictionary stores the classifier performance on the test set (if test_size is valid).
        """
        if method not in ["RandomForest", "GBDT", "KNN", "NaiveBayes"]:
            msg = "Only support RandomForest, GBDT, KNN, and NaiveBayes method."
            logging.error(msg)
            raise ValueError(msg)

        if valid_size <= 0.0 or valid_size >= 1.0:
            msg = "valid size should be in (0.0, 1.0)"
            logging.error(msg)
            raise ValueError(msg)

        if test_size <= 0.0 or test_size >= 1.0:
            msg = f"invalid test_size={test_size} and reset the test_size as 0."
            test_size = 0.0
            logging.warning(msg)

        n = len(self.features)
        x_train, x_valid, y_train, y_valid = train_test_split(
            self.features, self.labels, test_size=int(n * valid_size)
        )

        if test_size > 0 and test_size < (1 - valid_size):
            x_train, x_test, y_train, y_test = train_test_split(
                x_train, y_train, test_size=int(n * test_size)
            )
        elif test_size == 0:
            x_train, y_train = self.features, self.labels
            x_test, y_test = None, None
        else:
            msg = "Invalid test_size and re-set test_size as 0."
            logging.info(msg)
            x_train, y_train = self.features, self.labels
            x_test, y_test = None, None
        if method == "NaiveBayes":
            clf = GaussianNB(**kwargs)
        elif method == "GBDT":
            clf = GradientBoostingClassifier(**kwargs)
        elif method == "KNN":
            kwargs["n_neighbors"] = kwargs.get("n_neighbors", 5)
            clf = KNeighborsClassifier(**kwargs)
        else:
            kwargs["n_estimators"] = kwargs.get("n_estimators", 500)
            kwargs["class_weight"] = kwargs.get("class_weight", "balanced_subsample")
            clf = RandomForestClassifier(**kwargs)

        clf.fit(x_train, y_train)
        pred_valid = clf.predict_proba(x_valid)[:, 1]
        p, r, threshold = precision_recall_curve(y_valid, pred_valid)
        try:
            clf_threshold = threshold[np.where(p == np.max(p[r >= recall_threshold]))][
                -1
            ]
        except Exception as e:
            msg = f"Fail to get a proper threshold for recall {recall_threshold}, use 0.5 as threshold instead. Exception message is: {e}"
            logging.warning(msg)
            clf_threshold = 0.5

        if x_test is not None:
            pred_test = clf.predict_proba(x_test)[:, 1] > clf_threshold
            precision_test, recall_test, f1_test, _ = precision_recall_fscore_support(
                y_test, pred_test, average="binary"
            )
            accuracy = np.average(pred_test == y_test)
            ans = {
                "accuracy": accuracy,
                "precision": precision_test,
                "recall": recall_test,
                "f1": f1_test,
            }
        else:
            ans = {}
        self.clf = clf
        self.clf_threshold = clf_threshold
        return ans

    def pred(self, source_ts: TimeSeriesData, ts_rescale: bool = True) -> bool:
        """
        Predict whether a time series is predicable or not.

        :Parameters:
        source_ts: TimeSeriesData
            The time series that one wants to predict.
        ts_rescale: bool
            Whether to scale time series data before calculating features.
        """
        ts = TimeSeriesData(source_ts.to_dataframe().copy())
        if self.clf is None:
            msg = "No model trained yet, please train the model first."
            logging.error(msg)
            raise ValueError(msg)
        if ts_rescale:
            ts.value /= ts.value.max()
            msg = "Successful scaled! Each value of TS has been divided by the max value of TS."
            logging.info(msg)
        features = TsFeatures().transform(ts)
        x = np.array(list(features.values()))
        if np.sum(np.isnan(x)) > 0:
            msg = (
                "Features of ts contain NaN, please consider preprocessing ts. Features are: "
                f"{features}. Return False by default."
            )
            logging.warning(msg)
            return False
        return self.pred_by_feature([x])[0]

    def pred_by_feature(
        self, source_x: Union[np.ndarray, List[np.ndarray], pd.DataFrame]
    ) -> np.ndarray:
        """
        Predict whether a list of time series are predicable or not with their features.

        :Parameters:
        source_x: Union[np.ndarray, List, pd.DataFrame]
            The time series features.
        """
        if self.clf is None:
            msg = "No model trained yet, please train the model first."
            logging.error(msg)
            raise ValueError(msg)
        if isinstance(source_x, List):
            x = np.row_stack(source_x)
        else:
            x = source_x.copy()
        if self.rescale:
            x = (x - self.features_mean) / self.features_std
        pred = (self.clf.predict_proba(x)[:, 1] < self.clf_threshold).astype(int)
        return pred

    def save_model(
        self, file_path: str = "", file_name: str = "MetaLearnPredictability.pkl"
    ):
        """
        Save trained model.

        :Parameters:
        file_path: str
            Path to save trained model.
        file_name: str
            File name for the trained model.
        """
        if self.clf is None:
            msg = "Please train the model first!"
            logging.error(msg)
            raise ValueError(msg)
        joblib.dump(self.__dict__, file_path + file_name)
        logging.info(f"Successfully save the model: {file_path + file_name}.")

    def load_model(
        self, file_path: str = "", file_name: str = "MetaLearnPredictability.pkl"
    ):
        """
        Load a pre-trained model.

        :Parameters:
        file_path: str
            Path to load pre-trained model.
        file_name: str
            File name for the pre-trained model.
        """
        try:
            self.__dict__ = joblib.load(file_path + file_name)
            logging.info(f"Successfully load the model: {file_path + file_name}.")
        except Exception as e:
            msg = f"Fail to load model with Exception msg: {e}"
            logging.exception(msg)
            raise ValueError(msg)
