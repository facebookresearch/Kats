#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import ast
import logging
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from infrastrategy.kats.consts import TimeSeriesData
from infrastrategy.kats.tsfeatures.tsfeatures import TsFeatures
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class MetaLearnModelSelect:
    """
    Meta-learning framework on forecasting model selection.

    This framework is using classification algorithms with time series features as inputs, and best model as outputs.

    After training a classifier, it can directly predict best forecasting model for a new time series data.

    :Parameters:
    metadata: Optional[List]
        A list of Dict[str, Any]. The dict is meta data from outputs of class GetMetaData, and it must contain 3 components, hpt_res, features and best_model.
        hpt_res represents best hyper-params for each candidate model and its corresponding error, features are time series features, and best_model means the best candidate model for a given time series data.
        metadata can be None when load_model == True.
    load_model: bool
        Whether one wants to load a trained model from a saved file. If so, then initial metadata list is not necessary and data check will not be performed.
    :Example:
    >>> from infrastrategy.kats.models.metalearner.metalearner_modelselect import MetaLearnModelSelect
    >>> # create an object of class MetaLearnModelSelect
    >>> mlms = MetaLearnModelSelect(data)
    >>> # train a model
    >>> mlms.train(n_trees=200, test_size=0.1, eval_method='mean')
    >>> # predict forecasting model for a new time series data
    >>> mlms.pred(TSdata)
    >>> # save trained model
    >>> mlms.save_model("mlms.pkl")
    >>> # create a new object, and then load a pre-trained model
    >>> mlms2 = MetaLearnModelSelect(metadata=None, load_model=True)
    >>> mlms2.load_model("mlms.pkl")
    >>> # get the model with the highest predictive probabilty
    >>> mlms2.pred(TSdata)
    >>> # get the best 3 models
    >>> mlms2.pred(TSdata, n_top=3)
    >>> # get the best models (i.e. the best and the second best model if they are comparable)
    >>> mlms2.pred_fuzzy(TSdata)


    :Methods:
    """

    def __init__(
        self, metadata: Optional[List] = None, load_model: bool = False
    ) -> None:
        if not load_model:
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
            self._reorganize_data()
            self._validate_data()

            self.scale = False
            self.clf = None

    def _reorganize_data(self) -> None:
        hpt_list = []
        metadataX_list = []
        metadataY_list = []
        for i in range(len(self.metadata)):
            if isinstance(self.metadata[i]["hpt_res"], str):
                hpt_list.append(ast.literal_eval(self.metadata[i]["hpt_res"]))
            else:
                hpt_list.append(self.metadata[i]["hpt_res"])

            if isinstance(self.metadata[i]["features"], str):
                metadataX_list.append(
                    list(ast.literal_eval(self.metadata[i]["features"]).values())
                )
            else:
                metadataX_list.append(list(self.metadata[i]["features"].values()))

            metadataY_list.append(self.metadata[i]["best_model"])

        self.col_namesX = list(self.metadata[0]["features"].keys())  # Feature names
        self.hpt = pd.DataFrame({"hpt": hpt_list}).hpt
        self.metadataX = pd.DataFrame(metadataX_list)
        self.metadataX.columns = self.col_namesX
        self.metadataY = pd.DataFrame({"y": metadataY_list}).y

    def _validate_data(self):
        num_class = self.metadataY.nunique()
        if num_class == 1:
            msg = "Only one class in the label column (best_model), not able to train a classifier!"
            logging.error(msg)
            raise ValueError(msg)

        local_count = list(self.count_category().values())
        if min(local_count) * num_class < 30:
            msg = "Not recommend to do downsampling! Dataset will be too small after downsampling!"
            logging.info(msg)
        elif max(local_count) > min(local_count) * 5:
            msg = "Number of obs in majority class is much greater than in minority class. Downsampling is recommended!"
            logging.info(msg)
        else:
            msg = "No significant data imbalance problem, no need to do downsampling."
            logging.info(msg)

    def count_category(self) -> None:
        """
        Count number of observations in each class of Y.
        """
        return Counter(self.metadataY)

    def preprocess(self, downsample: bool = True, scale: bool = False) -> None:
        """
        Pre-process meta data before training a classifier.

        There are 2 options in this function: whether to downsample meta data, and whether to scale meta data.

        :Parameters:
        downsample: bool
            To fix data imbalance problem. True by default.
        scale: bool
            To scale metadataX. Center to the mean and component wise scale to unit variance. False by default.
        """
        if downsample:
            self.hpt, self.metadataX, self.metadataY = RandomDownSampler(
                self.hpt, self.metadataX, self.metadataY
            ).fit_resample()
            logging.info("Successfully applied random downsampling!")

        if scale:
            self.scale = True
            self.x_mean = np.average(self.metadataX.values, axis=0)
            self.x_std = np.std(self.metadataX.values, axis=0)

            self.metadataX = (self.metadataX - self.x_mean) / self.x_std
            logging.info(
                "Successfully scaled data by centering to the mean and component-wise scaling to unit variance!"
            )

    def plot_feature_comparison(self, i: int, j: int) -> None:
        """
        Generate features comparison plot.

        :Parameters:
        i: int
            Index of one feature vector from feature matrix to be compared.
        j: int
            Index of the other one feature vector from feature matrix to be compared.
        """
        combined = pd.concat([self.metadataX.iloc[i], self.metadataX.iloc[j]], axis=1)
        combined.columns = [
            str(self.metadataY.iloc[i]) + " model",
            str(self.metadataY.iloc[j]) + " model",
        ]
        combined.plot(kind="bar", figsize=(12, 6))

    def get_corr_mtx(self) -> pd.DataFrame:
        """
        Calculate correlation matrix of feature matrix.
        """
        return self.metadataX.corr()

    def plot_corr_heatmap(self, camp: str = "RdBu_r") -> None:
        """
        Generate heat-map for correlation matrix of feature matrix.

        :Parameters:
        camp: str
            Color bar used to generate heat-map.
        """
        fig, _ = plt.subplots(figsize=(8, 6))
        _ = sns.heatmap(
            self.get_corr_mtx(),
            cmap=camp,
            yticklabels=self.metadataX.columns,
            xticklabels=self.metadataX.columns,
        )

    def train(
        self,
        method: str = "RandomForest",
        eval_method: str = "mean",
        test_size: float = 0.1,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Train a meta-learner (classifier) with time series features as inputs, and best model as outputs.

        :Parameters:
        method: str
            Classification algorithm. Now we are supporting Random Forest model, gradient boosting decision tree model (GBDT), support vector machine (SVM),
            k-nearest neighbors model (KNN), and  Naive Bayes model.
        eval_method: str
            We will calculate errors for all TSs in both training and test sets for meta-learning method and pre-selected methods.
            Hence, we could calculate mean or median of errors as final evaluation of all methods.
        test_size: float
            Proportion of test set, which should be within (0, 1).
        n_trees: optional
            Number of trees in random forest model.
        n_neighbors: optional
            Number of neighbors in KNN model.
        """
        if method not in ["RandomForest", "GBDT", "SVM", "KNN", "NaiveBayes"]:
            msg = "Only support RandomForest, GBDT, SVM, KNN, and NaiveBayes method."
            logging.error(msg)
            raise ValueError(msg)

        if eval_method not in ["mean", "median"]:
            msg = "Only support mean and median as evaluation method."
            logging.error(msg)
            raise ValueError(msg)

        if test_size <= 0 or test_size >= 1:
            msg = "Illegal test set."
            logging.error(msg)
            raise ValueError(msg)

        x_train, x_test, y_train, y_test, hpt_train, hpt_test = train_test_split(
            self.metadataX, self.metadataY, self.hpt, test_size=test_size
        )

        if method == "RandomForest":
            n_trees = kwargs.get("n_trees", 500)
            clf = RandomForestClassifier(n_estimators=n_trees)
        elif method == "GBDT":
            clf = GradientBoostingClassifier()
        elif method == "SVM":
            clf = make_pipeline(StandardScaler(), SVC(gamma="auto"))
        elif method == "KNN":
            n_neighbors = kwargs.get("n_neighbors", 5)
            clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        else:
            clf = GaussianNB()

        clf.fit(x_train, y_train)
        y_fit = clf.predict(x_train)
        y_pred = clf.predict(x_test)

        # calculate model errors
        fit_error, pred_error = {}, {}

        # evaluate method
        em = np.mean if eval_method == "mean" else np.median

        # meta learning erros
        fit_error["meta-learn"] = em(
            [hpt_train.iloc[i][c][-1] for i, c in enumerate(y_fit)]
        )
        pred_error["meta-learn"] = em(
            [hpt_test.iloc[i][c][-1] for i, c in enumerate(y_pred)]
        )

        # pre-selected model errors, for all candidate models
        for label in self.metadataY.unique():
            fit_error[label] = em(
                [hpt_train.iloc[i][label][-1] for i in range(len(hpt_train))]
            )
            pred_error[label] = em(
                [hpt_test.iloc[i][label][-1] for i in range(len(hpt_test))]
            )

        self.clf = clf
        return {
            "fit_error": fit_error,
            "pred_error": pred_error,
            "clf_accuracy": metrics.accuracy_score(y_test, y_pred),
        }

    def save_model(self, file_name: str) -> None:
        """
        Save trained model.

        :Parameters:
        file_name: str
            File name for the trained model.
        """
        if self.clf is None:
            msg = "Haven't trained a model."
            logging.error(msg)
            raise ValueError(msg)
        else:
            joblib.dump(self.__dict__, file_name)
            logging.info("Successfully saved the trained model!")

    def load_model(self, file_name: str) -> None:
        """
        Load a pre-trained model.

        :Parameters:
        file_path: str
            Path to load pre-trained model.
        file_name: str
            File name for the pre-trained model.
        """
        try:
            self.__dict__ = joblib.load(file_name)
            logging.info("Successfully loaded a pre-trained model!")
        except Exception:
            msg = "No existing pre-trained model. Please change file path or train a model first!"
            logging.error(msg)
            raise ValueError(msg)

    def pred(
        self, source_ts: TimeSeriesData, ts_scale: bool = True, n_top: int = 1
    ) -> str:
        """
        Predict a forecasting model for a new time series data.

        :Parameters:
        source_ts: TimeSeriesData
            A new time series data.
        ts_scale: bool
            Whether to scale time series data before calculating features.
        n_top: int
            return the best n_top models.
        """
        ts = TimeSeriesData(source_ts.to_dataframe().copy())
        if self.clf is None:
            msg = "Haven't trained a model. Please train a model or load a model before predicting."
            logging.error(msg)
            raise ValueError(msg)

        if ts_scale:
            # scale time series to make ts features more stable
            ts.value /= ts.value.max()
            msg = "Successful scaled! Each value of TS has been divided by the max value of TS."
            logging.info(msg)

        new_features = TsFeatures().transform(ts)
        new_features_vector = np.asarray(list(new_features.values()))
        if np.sum(np.isnan(new_features_vector)) > 0:
            msg = (
                "Features of the test time series contains NaN value, consider processing it. Features are: "
                f"{new_features}. Return Prophet by default"
            )
            logging.warning(msg)
            if n_top == 1:
                return "prophet"
            else:
                return ["prophet"] * n_top
        return self.pred_by_feature([new_features_vector], n_top=n_top)[0]

    def pred_by_feature(
        self,
        source_x: Union[np.ndarray, List[np.ndarray], pd.DataFrame],
        n_top: int = 1,
    ) -> np.ndarray:
        """
        Predict a forecasting model for a list/dataframe of time series features

        :Parameters:
        source_x: Union[np.ndarray, List, pd.DataFrame]
            features of the time series that one wants to predict, can be a np.ndarray, a list of np.ndarray or a pandas dataframe.
        n_top: int
            return the best n_top models.

        :Returns:
            if n_top=1, then a 1d np.ndarray will be returned. Otherwise, a 2d np.ndarray will be returned.
        """
        if self.clf is None:
            msg = "Haven't trained a model. Please train a model or load a model before predicting."
            logging.error(msg)
            raise ValueError(msg)
        if isinstance(source_x, List):
            x = np.row_stack(source_x)
        else:
            x = source_x.copy()
        if self.scale:
            x = (x - self.x_mean) / self.x_std

        if n_top == 1:
            return self.clf.predict(x)
        prob = self.clf.predict_proba(x)
        order = np.argsort(-prob, axis=1)
        classes = np.array(self.clf.classes_)
        return classes[order][:, :n_top]

    def _bootstrap(self, data: np.array, rep: int = 200) -> float:
        """
        Helper function for bootstrap test and returns the pvalue.

        :Parameters:
        data: np.array
            data used to generate bootstrap sample
        rep: int
            size of bootstrap replicates
        """
        diff = data[:, 0] - data[:, 1]
        n = len(diff)
        idx = np.random.choice(np.arange(n), n * rep)
        sample = diff[idx].reshape(-1, n)
        bs = np.average(sample, axis=1)
        pvalue = np.average(bs < 0)
        return pvalue

    def pred_fuzzy(
        self, source_ts: TimeSeriesData, ts_scale: bool = True, sig_level: float = 0.2
    ) -> Dict[str, List]:
        """
        Predict a forecasting model for a new time series data. The best candiate model and the second best model will be
        returned if there is no statistically significant difference between them. The test is based on the bootstrapping
        samples drawn from the fitted random forest model.
        (Only for random forest model)

        :Parameters:
        source_ts: TimeSeriesData
            A new time series data.
        ts_scale: bool
            Whether to scale time series data before calculating features.
        sig_level: float
            significance level for bootstrap test. If pvalue>=sig_level, then we deem there is no difference between
            the best and the second best model.
        """
        ts = TimeSeriesData(source_ts.to_dataframe().copy())
        if ts_scale:
            # scale time series to make ts features more stable
            ts.value /= ts.value.max()
        new_features_vector = np.asarray(list(TsFeatures().transform(ts).values()))
        if self.scale:
            new_features_vector = (
                new_features_vector - np.asarray(self.x_mean)
            ) / np.asarray(self.x_std)
        test = new_features_vector.reshape([1, -1])
        m = len(self.clf.estimators_)
        data = np.array(
            [self.clf.estimators_[i].predict_proba(test)[0] for i in range(m)]
        )
        prob = self.clf.predict_proba(test)[0]
        idx = np.argsort(-prob)[:2]
        pvalue = self._bootstrap(data[:, idx[:2]])
        if pvalue >= sig_level:
            label = self.clf.classes_[idx[:2]]
            prob = prob[idx[:2]]
        else:
            label = self.clf.classes_[idx[:1]]
            prob = prob[idx[:1]]
        ans = {"label": label, "probability": prob, "pvalue": pvalue}
        return ans

    def __str__(self):
        return "MetaLearnModelSelect"


class RandomDownSampler:
    """
    An assistant class for class MetaLearnModelSelect to do random downsampling.

    :Parameters:
    hpt: pd.Series
        Best hyper-parameters for each model, and corresponding errors.
    dataX: pd.DataFrame,
        Time series features matrix.
    dataY: pd.Series,
        Best models.
    :Example:
    >>> from infrastrategy.kats.models.metalearner.metalearner_modelselect import RandomDownSampler
    >>> sampled_hpt, sampled_metadataX, sampled_metadataY = RandomDownSampler(hpt, metadataX, metadataY).fit_resample()

    :Methods:
    """

    def __init__(self, hpt: pd.Series, dataX: pd.DataFrame, dataY: pd.Series) -> None:
        self.hpt = hpt
        self.dataX = dataX
        self.dataY = dataY
        self.col_namesX = self.dataX.columns

    def fit_resample(self) -> Tuple:
        """Apply random downsampling technique."""
        resampled_x, resampled_y, resampled_hpt = [], [], []
        # naive down-sampler technique for data imbalance problem
        min_n = min(Counter(self.dataY).values())

        idx_dict = defaultdict(list)
        for i, c in enumerate(self.dataY):
            idx_dict[c].append(i)

        for key in idx_dict:
            idx_dict[key] = np.random.choice(idx_dict[key], size=min_n, replace=False)
            resampled_x += self.dataX.iloc[np.asarray(idx_dict[key]), :].values.tolist()
            resampled_y += list(self.dataY.iloc[np.asarray(idx_dict[key])])
            resampled_hpt += list(self.hpt.iloc[np.asarray(idx_dict[key])])

        resampled_x = pd.DataFrame(resampled_x)
        resampled_x.columns = self.col_namesX

        resampled_y = pd.DataFrame({"y": resampled_y}).y
        resampled_hpt = pd.DataFrame({"hpt": resampled_hpt}).hpt
        return resampled_hpt, resampled_x, resampled_y
