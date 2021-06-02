#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from unittest import TestCase

import numpy as np
import pandas as pd
from kats.consts import TimeSeriesData
from kats.transformers.t2v.consts import T2VParam
from kats.transformers.t2v.t2vbatch import T2VBatch
from kats.transformers.t2v.t2vnn import T2VNN
from kats.transformers.t2v.t2vpreprocessing import T2VPreprocessing
from kats.transformers.t2v.utils import Normalize, Standardize


class test_preprocessing(TestCase):
    def test_regression(self) -> None:
        # Create dummy time series
        ts = pd.DataFrame([np.arange(12), np.arange(12)]).transpose()
        # pyre-fixme[16]: `DataFrame` has no attribute `columns`.
        ts.columns = ["time", "value"]
        # pyre-fixme[16]: `test_preprocessing` has no attribute `ts`.
        self.ts = [TimeSeriesData(ts)]

        # Test Normalize normalization function
        param = T2VParam(mode="regression", normalizer=Normalize)

        preprocessor = T2VPreprocessing(
            param=param,
            data=self.ts,
        )
        preprocessed = preprocessor.transform()
        self.assertTrue(preprocessed.label == [11])
        self.assertTrue(
            np.array_equal(
                preprocessed.seq[0], ((np.arange(11)) / 10).reshape([11, 1])
            )
        )
        self.assertTrue(preprocessed.output_size == 1)
        self.assertTrue(preprocessed.window == 11)
        param = T2VParam(mode="regression", normalizer=Standardize)

        preprocessor = T2VPreprocessing(
            param=param,
            data=self.ts,
        )
        preprocessed = preprocessor.transform()
        self.assertTrue(
            np.array_equal(
                preprocessed.seq[0],
                (
                    (np.arange(11) - np.mean(np.arange(11))) / np.std(np.arange(11))
                ).reshape([11, 1]),
            )
        )
        self.assertFalse(preprocessed.batched)

        # Test regression with user defined label
        param = T2VParam(mode="regression", normalizer=Normalize)
        preprocessor = T2VPreprocessing(param=param, data=self.ts, label=[1.25])
        preprocessed = preprocessor.transform()
        self.assertTrue(preprocessed.label == [1.25])
        self.assertTrue(
            np.array_equal(
                preprocessed.seq[0], ((np.arange(12)) / 11).reshape([12, 1])
            )
        )
        self.assertTrue(preprocessed.output_size == 1)
        self.assertTrue(preprocessed.window == 12)

        # Test when feeding more labels than should, should raise ValueError
        param = T2VParam(mode="regression", normalizer=Normalize)

        preprocessor = T2VPreprocessing(param=param, data=self.ts, label=[1, 2])
        self.assertRaises(ValueError, preprocessor.transform)

    def test_classification(self) -> None:
        # Create dummy time series
        ts = pd.DataFrame([np.arange(12), np.arange(12)]).transpose()
        # pyre-fixme[16]: `DataFrame` has no attribute `columns`.
        ts.columns = ["time", "value"]
        # pyre-fixme[16]: `test_preprocessing` has no attribute `ts`.
        self.ts = [TimeSeriesData(ts)]

        # Test feeding float number as class label, should raise ValueError
        param = T2VParam(mode="classification", normalizer=Normalize)

        preprocessor = T2VPreprocessing(param=param, data=self.ts, label=[1.55])
        self.assertRaises(ValueError, preprocessor.transform)

        param = T2VParam(mode="classification", normalizer=Normalize)
        preprocessor = T2VPreprocessing(param=param, data=self.ts, label=[2])
        preprocessed = preprocessor.transform()
        self.assertTrue(preprocessed.label == [2])
        self.assertTrue(
            np.array_equal(
                preprocessed.seq[0], ((np.arange(12)) / 11).reshape([12, 1])
            )
        )
        self.assertTrue(preprocessed.output_size == 3)

        # Test classification mode with Standardize
        param = T2VParam(mode="classification", normalizer=Standardize)

        preprocessor = T2VPreprocessing(param=param, data=self.ts, label=[0])
        preprocessed = preprocessor.transform()
        self.assertTrue(preprocessed.label == [0])
        self.assertTrue(
            np.array_equal(
                preprocessed.seq[0],
                (
                    (np.arange(12) - np.mean(np.arange(12))) / np.std(np.arange(12))
                ).reshape([12, 1]),
            )
        )
        self.assertTrue(preprocessed.output_size == 1)

        # Test when not feeding label for classification mode, should raise ValueError
        param = T2VParam(mode="classification", normalizer=Normalize)

        preprocessor = T2VPreprocessing(
            param=param,
            data=self.ts,
        )
        self.assertRaises(ValueError, preprocessor.transform)


class test_batch(TestCase):
    def test_case_1(self) -> None:
        TS = []
        for _ in range(100):
            ts = pd.DataFrame(
                [np.arange(12), np.random.randint(0, 100, 12)]
            ).transpose()
            # pyre-fixme[16]: `DataFrame` has no attribute `columns`.
            ts.columns = ["time", "value"]
            ts = TimeSeriesData(ts)
            TS.append(ts)
        logging.info("Time series data simulated.")

        # preprocessing time series data
        param = T2VParam(
            normalizer=Normalize,
            batch_size=32,
        )
        preprocessor = T2VPreprocessing(
            param=param,
            data=TS,
        )
        preprocessed = preprocessor.transform()
        logging.info("Time series data preprocessed.")

        batched = T2VBatch(preprocessed, param).transform()
        logging.info("Time series data batched.")
        self.assertTrue(len(batched.seq) == 100)
        self.assertTrue(len(batched.batched_tensors[0]) == 32)
        self.assertTrue(len(batched.batched_tensors[-1]) == 4)
        self.assertTrue(batched.batch_size == 32)
        # testing if T2VBatched inherited all attributes from T2VProcessed
        self.assertTrue(np.array_equal(batched.label, preprocessed.label))
        self.assertTrue(np.array_equal(batched.seq, preprocessed.seq))
        self.assertTrue(batched.window == preprocessed.window)
        self.assertTrue(batched.batched)
        self.assertFalse(preprocessed.batched)
        # test if the value matches
        self.assertTrue(
            np.array_equal(
                batched.batched_tensors[0][0][0].numpy(),
                preprocessed.seq[0].reshape([batched.window, 1]),
            )
        )
        self.assertTrue(
            np.array_equal(
                batched.batched_tensors[0][-1][0].numpy(),
                preprocessed.seq[31].reshape([batched.window, 1]),
            )
        )
        self.assertTrue(
            batched.batched_tensors[0][0][1].item() == preprocessed.label[0]
        )
        self.assertTrue(
            np.array_equal(
                batched.batched_tensors[-1][-1][0].numpy(),
                preprocessed.seq[-1].reshape([batched.window, 1]),
            )
        )
        self.assertTrue(
            batched.batched_tensors[-1][-1][1].item() == preprocessed.label[-1]
        )

    def test_case_2(self) -> None:
        # simulate dummy time series data
        TS = []
        for _ in range(160):
            ts = pd.DataFrame(
                [np.arange(10), np.random.randint(0, 100, 10)]
            ).transpose()
            # pyre-fixme[16]: `DataFrame` has no attribute `columns`.
            ts.columns = ["time", "value"]
            ts = TimeSeriesData(ts)
            TS.append(ts)
        logging.info("Time series data simulated.")

        # Test Normalize normalization function
        param = T2VParam(
            normalizer=Normalize,
            batch_size=16,
        )
        preprocessor = T2VPreprocessing(
            param=param,
            data=TS,
        )
        preprocessed = preprocessor.transform()
        logging.info("Time series data preprocessed.")

        batched = T2VBatch(preprocessed, param).transform()
        logging.info("Time series data batched.")

        self.assertTrue(len(batched.batched_tensors[-1]) == 16)
        self.assertTrue(len(batched.batched_tensors) == 10)
        self.assertTrue(
            np.array_equal(
                batched.batched_tensors[0][-1][0].numpy(),
                preprocessed.seq[15].reshape([batched.window, 1]),
            )
        )
        self.assertTrue(
            np.array_equal(
                batched.batched_tensors[-1][-1][0].numpy(),
                preprocessed.seq[-1].reshape([batched.window, 1]),
            )
        )
        self.assertTrue(
            np.array_equal(
                batched.batched_tensors[2][5][0].numpy(),
                preprocessed.seq[37].reshape([batched.window, 1]),
            )
        )
class test_t2vnn(TestCase):

    def test_regression(self) -> None:
        # data simulation and initialization
        TS = []
        for _ in range(64):
            ts = pd.DataFrame(
                [np.arange(30), np.random.randint(0, 11, 30).astype(float)]
            ).transpose()
            # pyre-fixme[16]: `DataFrame` has no attribute `columns`.
            ts.columns = ["time", "value"]
            ts = TimeSeriesData(ts)
            TS.append(ts)

        # Turn testing data into a list of time series
        test_TS = []
        for _ in range(64):
            ts = pd.DataFrame(
                [np.arange(30), np.random.randint(0, 11, 30).astype(float)]
            ).transpose()
            ts.columns = ["time", "value"]
            ts = TimeSeriesData(ts)
            test_TS.append(ts)

        # Preprocessing training data into sequences
        param = T2VParam(
            mode="regression",
            normalizer=Normalize,
            batch_size=32,
            vector_length=16,
            learning_rate=0.001,
            hidden=[64],
            dropout=0.2,
            epochs=2,
        )
        preprocessor = T2VPreprocessing(
            param=param,
            data=TS,
        )
        preprocessed = preprocessor.transform()
        # Batching traing data
        batched = T2VBatch(preprocessed, param).transform()

        ###
        # Preprocessing testing data into sequences
        test_preprocessor = T2VPreprocessing(
            param=param,
            data=test_TS,
        )
        test_preprocessed = test_preprocessor.transform()

        # training
        t2vnn = T2VNN(batched, param)
        t2vnn.train()

        # validation output
        val_output = t2vnn.val(preprocessed)
        self.assertTrue(list(val_output.keys())[0] == "mae")

        # embedding translation
        test_embeddings = t2vnn.translate(test_preprocessed)
        self.assertTrue(len(test_embeddings[0]) == 16)

    def test_classification(self) -> None:
        # data simulation and initialization
        TS = []
        for _ in range(32):
            ts = pd.DataFrame(
                [np.arange(30), np.random.randint(0, 11, 30).astype(float)]
            ).transpose()
            # pyre-fixme[16]: `DataFrame` has no attribute `columns`.
            ts.columns = ["time", "value"]
            ts = TimeSeriesData(ts)
            TS.append(ts)

        class_labels = np.random.randint(0, 2, 32)

        # Turn testing data into a list of time series
        test_TS = []
        for _ in range(32):
            ts = pd.DataFrame(
                [np.arange(30), np.random.randint(0, 11, 30).astype(float)]
            ).transpose()
            ts.columns = ["time", "value"]
            ts = TimeSeriesData(ts)
            test_TS.append(ts)

        param = T2VParam(
            mode="classification",
            normalizer=Normalize,
            batch_size=16,
            vector_length=32,
            learning_rate=0.001,
            hidden=[64],
            dropout=0.2,
            epochs=2,
        )
        preprocessor = T2VPreprocessing(
            param=param,
            data=TS,
            label=class_labels,
        )
        preprocessed = preprocessor.transform()

        batched = T2VBatch(preprocessed, param).transform()

        ###
        # Preprocessing testing data into sequences
        test_preprocessor = T2VPreprocessing(
            param=param, data=test_TS, dummy_label=True
        )
        test_preprocessed = test_preprocessor.transform()

        # Batching testing data
        test_batched = T2VBatch(test_preprocessed, param).transform()

        # training batched data and translate
        t2vnn = T2VNN(batched, param)
        train_translated = t2vnn.train(translate=True)
        self.assertTrue(
            train_translated["labels"][0] == batched.batched_tensors[0][0][1].item()
        )

        # translate only
        test_embeddings = t2vnn.translate(test_batched)
        self.assertTrue(len(test_embeddings[0]) == 32)

        # training on preprocessed data
        t2vnn = T2VNN(preprocessed, param)
        t2vnn.train()

        test_embeddings = t2vnn.translate(test_batched)
        self.assertTrue(len(test_embeddings[0]) == 32)
