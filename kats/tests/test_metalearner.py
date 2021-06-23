# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import collections
import logging
from unittest import TestCase

import numpy as np
import pandas as pd
from ax.modelbridge.registry import Models, SearchSpace
from ax.service.utils.instantiation import parameter_from_json
from kats.consts import TimeSeriesData
from kats.models.arima import ARIMAModel
from kats.models.holtwinters import HoltWintersModel
from kats.models.metalearner.get_metadata import GetMetaData
from kats.models.metalearner.metalearner_hpt import MetaLearnHPT
from kats.models.metalearner.metalearner_modelselect import (
    MetaLearnModelSelect,
)
from kats.models.metalearner.metalearner_predictability import (
    MetaLearnPredictability,
)
from kats.models.prophet import ProphetModel
from kats.models.sarima import SARIMAModel
from kats.models.stlf import STLFModel
from kats.models.theta import ThetaModel
from kats.tsfeatures.tsfeatures import TsFeatures


DATA = pd.DataFrame(
    {
        "time": pd.date_range("2020-05-06", periods=60, freq="D"),
        "y": np.arange(1, 61),
    }
)
TSData = TimeSeriesData(DATA)

# TS which is too short
TSData_short = TimeSeriesData(DATA.iloc[:8, :])

# TS which has constant values only
DATA_const = DATA.copy()
DATA_const["y"] = 1
TSData_const = TimeSeriesData(DATA_const)

# TS which has NAN values
DATA_nan = DATA.copy()
DATA_nan.iloc[10, 1] = np.nan
TSData_nan = TimeSeriesData(DATA_nan)

# TS which has INF values
DATA_inf = DATA.copy()
DATA_inf.iloc[10, 1] = np.inf
TSData_inf = TimeSeriesData(DATA_inf)

# TS which doesn't have constant frequency
DATA_gap = DATA.copy()
DATA_gap = DATA_gap.drop([3, 4])
TSData_gap = TimeSeriesData(DATA_gap)

# TS which is not univariate
DATA_multi = pd.DataFrame(
    {
        "time": pd.date_range("2020-05-06", periods=60, freq="D"),
        "y": np.arange(1, 61),
        "z": np.random.randn(60),
    }
)
TSData_multi = TimeSeriesData(DATA_multi)

# Base Models
base_models = {
    "arima": ARIMAModel,
    "holtwinters": HoltWintersModel,
    "sarima": SARIMAModel,
    "prophet": ProphetModel,
    "stlf": STLFModel,
    "theta": ThetaModel,
}


def generate_test_ts():
    # time series with negative data, which contains Nan for TsFeatures
    time = pd.date_range("2020-05-06", "2020-11-17", freq="D")
    ts = pd.DataFrame(np.random.randn(len(time)), columns=["value"])
    ts["time"] = time
    ts1 = TimeSeriesData(ts)
    # predictable time series
    ts = pd.DataFrame(np.abs(np.random.randn(len(time))), columns=["value"])
    ts["time"] = time
    ts2 = TimeSeriesData(ts)
    return (ts1, ts2)


def generate_meta_data(n):
    # generate meta data to initialize MetaLearnModelSelect
    spaces = {m: base_models[m].get_parameter_search_space() for m in base_models}

    m = len(base_models)
    res = np.abs(np.random.uniform(0, 1.0, n * m)).reshape(n, -1)
    features = np.random.randn(n * 40).reshape(n, -1)
    generators = {
        m: Models.UNIFORM(
            SearchSpace([parameter_from_json(item) for item in spaces[m]])
        )
        for m in spaces
    }
    models = list(base_models.keys())
    ans = []
    for i in range(n):
        hpt = {}
        j = 0
        for m in base_models:
            hpt[m] = (generators[m].gen(1).arms[0].parameters, res[i, j])
            j += 1
        ans.append(
            {
                "hpt_res": hpt,
                "best_model": np.random.choice(models),
                "features": {str(k): features[i, k] for k in range(features.shape[1])},
            }
        )
    return ans


def generate_meta_data_by_model(model, n, d=40):
    model = model.lower()
    if model in base_models:
        model = base_models[model]
    space = model.get_parameter_search_space()
    generator = Models.UNIFORM(
        SearchSpace([parameter_from_json(item) for item in space])
    )
    x = np.random.randn(n * d).reshape(n, -1)
    x = pd.DataFrame(x)
    y = [generator.gen(1).arms[0].parameters for i in range(n)]
    y = pd.DataFrame(y)
    return x, y


def equals(v1, v2):
    # check whether v1 and v2 are equal
    try:
        if isinstance(v1, pd.DataFrame):
            return v1.equals(v2)
        elif isinstance(v1, np.ndarray):
            return np.array_equal(v1, v2)
        elif isinstance(v1, list) and (len(v1) == len(v2)):
            comp = [equals(v1[i], v2[i]) for i in range(len(v1))]
            return np.sum(comp) == len(comp)
        else:
            return False
    except Exception as e:
        msg = "fail to compare the inputs and exception message is " + e
        raise ValueError(msg)


class testMetaLearner(TestCase):
    def test_get_meta_data(self) -> None:
        # test GetMetaData using a simple case
        metadata = GetMetaData(data=TSData, num_trials=2, num_arms=1)
        res = metadata.get_meta_data()

        # test meta data output
        self.assertEqual(
            list(res.keys()),
            ["hpt_res", "features", "best_model", "search_method", "error_method"],
        )

        # test meta data output - HPT part
        self.assertEqual(
            list(res["hpt_res"].keys()),
            ["arima", "holtwinters", "prophet", "theta", "stlf", "sarima"],
        )

    def test_inputdata_errors(self) -> None:
        # test input data error (time series' type is not TimeSeriesData)
        self.assertRaises(ValueError, GetMetaData, DATA)

        # test input data error (time series is not univariate)
        self.assertRaises(ValueError, GetMetaData, TSData_multi)

        # test input data error (time series is too short)
        self.assertRaises(ValueError, GetMetaData, TSData_short)

        # test input data error (time series only contains constant value)
        self.assertRaises(ValueError, GetMetaData, TSData_const)

        # test input data error (time series contains nan)
        self.assertRaises(ValueError, GetMetaData, TSData_nan)

        # test input data error (time series contains inf)
        self.assertRaises(ValueError, GetMetaData, TSData_inf)

        # test input data error (time series doesn't have constant freq)
        self.assertRaises(ValueError, GetMetaData, TSData_gap)


class MetaLearnModelSelectTest(TestCase):
    def test_initialize(self) -> None:

        self.assertRaises(ValueError, MetaLearnModelSelect, [])

        self.assertRaises(ValueError, MetaLearnModelSelect, [{}] * 40)

        self.assertRaises(ValueError, MetaLearnModelSelect, [{"hpt_res": [None]}] * 40)

        self.assertRaises(
            ValueError,
            MetaLearnModelSelect,
            [{"hpt_res": [None], "features": [None]}] * 40,
        )

        self.assertRaises(
            ValueError,
            MetaLearnModelSelect,
            [{"hpt_res": [1.0], "features": {"f": 1.0}, "best_model": "best"}] * 40,
        )

    def test_model(self) -> None:
        samples = generate_meta_data(n=35)
        mlms = MetaLearnModelSelect(samples)

        # Test preprocess
        mlms.preprocess(downsample=True, scale=True)

        # Test rescale
        mtx = mlms.metadataX.values

        # test variable-wise zero-mean
        self.assertEqual(
            np.sum(np.abs(np.average(mtx, axis=0)) < 1e-10),
            mtx.shape[1],
            f"After rescaling, each variable should have zero-mean. with {np.average(mtx, axis=0)}",
        )

        # test variable-wise unit std
        self.assertEqual(
            np.sum(np.abs(np.std(mtx, axis=0) - 1) < 1e-8),
            mtx.shape[1],
            "After rescaling, each variable should have unit standard deviation.",
        )

        # Test subsample
        if len(np.unique(list(collections.Counter(mlms.metadataY).values()))) > 1:
            msg = "RandomDownSample fails."
            logging.error(msg)
            raise ValueError(msg)

        # Test train
        mlms.train(method="RandomForest")
        # Test prediction consistency
        t1, t2 = generate_test_ts()
        t2_df = t2.to_dataframe().copy()
        pred = mlms.pred(t2)
        pred_fuzzy = mlms.pred_fuzzy(t2)
        pred_all = mlms.pred(t2, n_top=2)
        if pred != pred_fuzzy["label"][0] or pred != pred_all[0]:
            msg = f"Prediction is not consistent! Results are: self.pred: {pred}, self.pred_fuzzy: {pred_fuzzy}, self.pred(, n_top=2): {pred_all}"
            logging.error(msg)
            raise ValueError(msg)
        # Test case for time series with nan features
        _ = mlms.pred(t1)
        # Test pred_by_feature and its consistency
        feature = np.random.randn(3 * mlms.metadataX.shape[1]).reshape(3, -1)
        feature2 = feature.copy()
        pred = mlms.pred_by_feature(feature)
        pred_all = mlms.pred_by_feature(feature, n_top=2)

        if np.sum(pred != pred_all[:, 0]) > 0:
            msg = f"pred_by_feature method is not consistent. Results are: self.pred_by_feature: {pred}, self.pred_by_feature(, n_top=2): {pred_all}"
            logging.error(msg)
            raise ValueError(msg)
        # Test if the target TimeSeriesData keeps its original value
        equals(t2.to_dataframe(), t2_df)
        # Test if the features keep their original values
        equals(feature, feature2)


class MetaLearnPredictabilityTest(TestCase):
    def test_initialize(self) -> None:
        self.assertRaises(ValueError, MetaLearnPredictability)

        self.assertRaises(ValueError, MetaLearnPredictability, metadata=[])

        MetaLearnPredictability(load_model=True)

    def test_model(self) -> None:
        # Train a model
        data = generate_meta_data(40)
        mlp = MetaLearnPredictability(data)
        mlp.preprocess()
        mlp.train()
        # Test prediction for ts
        t1, t2 = generate_test_ts()
        t2_df = t2.to_dataframe().copy()
        # Test case for time series with nan features
        ts_pred = mlp.pred(t1)
        self.assertTrue(
            isinstance(ts_pred, bool),
            f"The output of MetaLearnPredictability should be a boolean but receives {type(ts_pred)}.",
        )

        mlp.pred(t2)
        features = np.random.randn(3 * mlp.features.shape[1]).reshape(3, -1)
        features2 = features.copy()
        mlp.pred_by_feature(features)
        # Test if the target TimeSeriesData keeps its original value
        equals(t2.to_dataframe(), t2_df)
        # Test if the features keep their original values
        equals(features, features2)


class MetaLearnHPTTest(TestCase):
    def test_default_models(self) -> None:
        t1, t2 = generate_test_ts()
        t2_df = t2.to_dataframe().copy()
        feature1 = np.random.randn(3 * 40).reshape(3, -1)
        feature2 = [np.random.randn(40), np.random.randn(40)]
        feature3 = pd.DataFrame(np.random.randn(3 * 40).reshape(3, -1))
        feature1_copy, feature2_copy, feature3_copy = (
            feature1.copy(),
            list(feature2),
            feature3.copy(),
        )
        for model in ["prophet", "arima", "sarima", "theta", "stlf", "holtwinters"]:
            x, y = generate_meta_data_by_model(model, 150, 40)
            # Check default models initialization and training
            mlhpt = MetaLearnHPT(x, y, default_model=model)
            mlhpt.get_default_model()
            self.assertRaises(ValueError, mlhpt.build_network, [20])
            mlhpt.build_network()
            mlhpt.train()
            # Test case for time series with nan features
            _ = (mlhpt.pred(t1).parameters[0],)
            mlhpt.pred(t2)
            mlhpt.pred_by_feature(feature1)
            mlhpt.pred_by_feature(feature2)
            mlhpt.pred_by_feature(feature3)
            # Check prediction consistency:
            dict1 = mlhpt.pred(t2).parameters[0]
            t2.value /= t2.value.max()
            dict2 = mlhpt.pred_by_feature(pd.DataFrame([TsFeatures().transform(t2)]))[0]
            for elm in dict1:
                if (
                    isinstance(dict1[elm], float)
                    and abs(dict1[elm] - dict2[elm]) <= 1e-5
                ):
                    pass
                elif dict1[elm] == dict2[elm]:
                    pass
                else:
                    msg = (
                        "Predictions given by .pred and .pred_by_feature are different! The predictions are: .pred:"
                        f"{dict1[elm]}, .pred_by_feature: {dict2[elm]}."
                    )
                    logging.error(msg)
                    raise ValueError(msg)
        # Test if the target TimeSeriesData keeps its original value
        equals(t2_df, t2.to_dataframe())
        # Test if the features keep their original values
        equals(feature1, feature1_copy)
        equals(feature2, feature2_copy)
        equals(feature3, feature3_copy)

    def test_initialize(self) -> None:
        x, y = generate_meta_data_by_model("arima", 150, 40)
        self.assertRaises(ValueError, MetaLearnHPT, x, y)
        # Test load model method
        MetaLearnHPT(load_model=True)
        # Test customized initialization
        MetaLearnHPT(x, y, ["p"], ["d", "q"])
        self.assertRaises(ValueError, MetaLearnHPT, x, y, categorical_idx=["p"])
        self.assertRaises(ValueError, MetaLearnHPT, x, y, numerical_idx=["p"])
        self.assertRaises(
            ValueError, MetaLearnHPT, x, y, categorical_idx=["p"], default_model="arima"
        )

    def test_customized_models(self) -> None:
        t1, t2 = generate_test_ts()
        x, y = generate_meta_data_by_model("arima", 150, 40)
        # Test customized model
        mlhpt = MetaLearnHPT(x, y, ["p"], ["d", "q"])
        self.assertRaises(ValueError, mlhpt.build_network)
        # pyre-fixme[6]: Expected `Optional[typing.List[int]]` for 2nd param but got
        #  `List[typing.List[int]]`.
        mlhpt.build_network([40], [[5]], [10, 20])
        mlhpt.train()
        mlhpt.pred(t2)
