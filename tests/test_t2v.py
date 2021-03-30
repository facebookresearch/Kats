import logging
from unittest import TestCase

import numpy as np
import pandas as pd
from infrastrategy.kats.consts import TimeSeriesData
from infrastrategy.kats.transformers.t2v.consts import T2VParam
from infrastrategy.kats.transformers.t2v.t2vbatch import T2VBatch
from infrastrategy.kats.transformers.t2v.t2vpreprocessing import T2VPreprocessing
from infrastrategy.kats.transformers.t2v.utils import MinMax, Standardize


class test_preprocessing(TestCase):
    def test_regression(self):
        # Create dummy time series
        ts = pd.DataFrame([np.arange(12), np.arange(12)]).transpose()
        ts.columns = ["time", "value"]
        self.ts = [TimeSeriesData(ts)]

        # Test MinMax normalization function
        param = T2VParam(mode="regression", normalizer=MinMax)

        preprocessor = T2VPreprocessing(
            param=param,
            data=self.ts,
        )
        preprocessed = preprocessor.transform()
        self.assertTrue(preprocessed.label == [11])
        self.assertTrue(np.array_equal(preprocessed.seq[0], (np.arange(11) - 0) / 10))
        self.assertTrue(preprocessed.output_size == 1)
        self.assertTrue(preprocessed.window == 11)

        # Test Standardize normalization function
        param = T2VParam(mode="regression", normalizer=Standardize)

        preprocessor = T2VPreprocessing(
            param=param,
            data=self.ts,
        )
        preprocessed = preprocessor.transform()
        self.assertTrue(
            np.array_equal(
                preprocessed.seq[0],
                (np.arange(11) - np.mean(np.arange(11))) / np.std(np.arange(11)),
            )
        )
        self.assertFalse(preprocessed.batched)

        # Test regression with user defined label
        param = T2VParam(mode="regression", normalizer=MinMax)

        preprocessor = T2VPreprocessing(param=param, data=self.ts, label=[1.25])
        preprocessed = preprocessor.transform()
        self.assertTrue(preprocessed.label == [1.25])
        self.assertTrue(np.array_equal(preprocessed.seq[0], (np.arange(12) - 0) / 11))
        self.assertTrue(preprocessed.output_size == 1)
        self.assertTrue(preprocessed.window == 12)

        # Test when feeding more labels than should, should raise ValueError
        param = T2VParam(mode="regression", normalizer=MinMax)

        preprocessor = T2VPreprocessing(param=param, data=self.ts, label=[1, 2])
        self.assertRaises(ValueError, preprocessor.transform)

    def test_classification(self):
        # Create dummy time series
        ts = pd.DataFrame([np.arange(12), np.arange(12)]).transpose()
        ts.columns = ["time", "value"]
        self.ts = [TimeSeriesData(ts)]

        # Test feeding float number as class label, should raise ValueError
        param = T2VParam(mode="classification", normalizer=MinMax)

        preprocessor = T2VPreprocessing(param=param, data=self.ts, label=[1.55])
        self.assertRaises(ValueError, preprocessor.transform)

        # Test classification mode with MinMax
        param = T2VParam(mode="classification", normalizer=MinMax)

        preprocessor = T2VPreprocessing(param=param, data=self.ts, label=[2])
        preprocessed = preprocessor.transform()
        self.assertTrue(preprocessed.label == [2])
        self.assertTrue(np.array_equal(preprocessed.seq[0], (np.arange(12) - 0) / 11))
        self.assertTrue(preprocessed.output_size == 3)

        # Test classification mode with Standardize
        param = T2VParam(mode="classification", normalizer=Standardize)

        preprocessor = T2VPreprocessing(param=param, data=self.ts, label=[0])
        preprocessed = preprocessor.transform()
        self.assertTrue(preprocessed.label == [0])
        self.assertTrue(
            np.array_equal(
                preprocessed.seq[0],
                (np.arange(12) - np.mean(np.arange(12))) / np.std(np.arange(12)),
            )
        )
        self.assertTrue(preprocessed.output_size == 1)

        # Test when not feeding label for classification mode, should raise ValueError
        param = T2VParam(mode="classification", normalizer=MinMax)

        preprocessor = T2VPreprocessing(
            param=param,
            data=self.ts,
        )
        self.assertRaises(ValueError, preprocessor.transform)


class test_batch(TestCase):
    def test_case_1(self):
        # simulate dummy time series data
        TS = []
        for _ in range(100):
            ts = pd.DataFrame(
                [np.arange(12), np.random.randint(0, 100, 12)]
            ).transpose()
            ts.columns = ["time", "value"]
            ts = TimeSeriesData(ts)
            TS.append(ts)
        logging.info("Time series data simulated.")

        # preprocessing time series data
        param = T2VParam()
        preprocessor = T2VPreprocessing(
            param=param,
            data=TS,
        )
        preprocessed = preprocessor.transform()
        logging.info("Time series data preprocessed.")

        batched = T2VBatch(preprocessed, 32).transform()
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

    def test_case_2(self):
        # simulate dummy time series data
        TS = []
        for _ in range(160):
            ts = pd.DataFrame(
                [np.arange(10), np.random.randint(0, 100, 10)]
            ).transpose()
            ts.columns = ["time", "value"]
            ts = TimeSeriesData(ts)
            TS.append(ts)
        logging.info("Time series data simulated.")

        # Test MinMax normalization function
        param = T2VParam()
        preprocessor = T2VPreprocessing(
            param=param,
            data=TS,
        )
        preprocessed = preprocessor.transform()
        logging.info("Time series data preprocessed.")

        batched = T2VBatch(preprocessed, 16).transform()
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
