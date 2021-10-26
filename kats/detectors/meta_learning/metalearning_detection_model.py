# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import List, Dict

import pandas as pd
from kats.models.metalearner.metalearner_modelselect import MetaLearnModelSelect

NUM_SECS_IN_DAY = 3600 * 24
PARAMS_TO_SCALE_DOWN = {"n_control", "n_test", "historical_window", "scan_window"}


def change_dtype(d):
    for elm in d:
        d[elm] = float(d[elm])
    return d


class MetaDetectModelSelect(object):
    def __init__(self, df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            msg = "Dataset is not in form of a dataframe!"
            logging.error(msg)
            raise ValueError(msg)

        if len(df) <= 30:
            msg = "Dataset is too small to train a meta learner!"
            logging.error(msg)
            raise ValueError(msg)

        if "hpt_res" not in df:
            msg = "Missing best hyper-params, not able to train a meta learner!"
            logging.error(msg)
            raise ValueError(msg)

        if "features" not in df:
            msg = "Missing features, not able to train a meta learner!"
            logging.error(msg)
            raise ValueError(msg)

        if "best_model" not in df:
            msg = "Missing best models, not able to train a meta learner!"
            logging.error(msg)
            raise ValueError(msg)

        self.df = df
        self.results = None

    def preprocess(self) -> List:
        # prepare the training data
        # Create training data table
        table = [
            {
                "hpt_res": self.df["hpt_res"][i],
                "features": self.df["features"][i],
                "best_model": self.df["best_model"][i],
            }
            for i in range(len(self.df))
        ]

        # Change dtype of TSFeatures for compatibility
        for t in table:
            t["features"] = change_dtype(t["features"])

        # Scaling down certain params by num_secs_in_day to make models easier to converge
        for ts_data in table:
            for hpt_vals in ts_data["hpt_res"].values():
                params = hpt_vals[0]
                for param in params.keys():
                    if param in PARAMS_TO_SCALE_DOWN:
                        params[param] = params[param] / NUM_SECS_IN_DAY
        return table

    def train(self) -> Dict[str, Dict[str, float]]:
        # call the train() method of MetaLearnModelSelect
        mlms = MetaLearnModelSelect(self.preprocess())
        self.results = mlms.train()
        return self.results

    def report_metrics(self) -> pd.DataFrame:
        # report the summary, as in the notebook N1154788
        if self.results is None:
            self.results = self.train()
        summary = pd.DataFrame([self.results["fit_error"], self.results["pred_error"]])
        summary["type"] = ["fit_error", "pred_error"]
        summary["error_metric"] = "Inverted F-score"
        return summary

    def predict(self, TimeSeriesData):
        # for a given timeseries data, predicts the best model
        # this can be omitted, for the bootcamp task (add later)
        raise ValueError("Predict method hasn't been implemented yet.")
