#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals
from datainfra.dataswarm.common.hivetransformer import HiveTransformer
from infrastrategy.kats.tsfeatures.tsfeatures import TsFeatures
from infrastrategy.kats.consts import TimeSeriesData
import pandas as pd
import numpy as np
import logging
import json


def transform(metadata, line):
    """
    Spark transformer function calling TsFeatures

    Parameters
    ----------
    metadata : a dict of command line arguments to the script

    line : a dict of {column_name: value} representing one row from input table

    Returns
    -------
    output : a dict of {column_name : value} representing one row in the output table
    """
    # extract data from input
    key = int(line['id'])
    values = json.loads(line['values'])  # a list of floats

    # create TimeSeriesData object from input data
    ts_obj = TimeSeriesData(pd.DataFrame(
        {'time': list(range(len(values))), 'values': values})
    )

    # parsing selected features
    selected_features = metadata.get('selected_features', None)
    if selected_features:
        selected_features = selected_features.split(',')

    # initiate time series featurizer
    featurizer = TsFeatures(
        # spectral frequency for entropy calculation
        spectral_freq=int(metadata.get('spectral_freq', 1)),
        # window size for rolling functions
        window_size=int(metadata.get('window_size', 20)),
        # seasonal period
        stl_period=int(metadata.get('stl_period', 7)),
        # number of bins
        nbins=int(metadata.get('nbins', 10)),
        # lag size
        lag_size=int(metadata.get('lag_size', 30)),
        # lag size for AC/PAC features
        acfpacf_lag=int(metadata.get('acfpacf_lag', 6)),
        # time series decomposition (used in outlier detection)
        decomp=str(metadata.get('decomp', 'additive')),
        # multiplier to inter quartile range used in outlier detection
        iqr_mult=float(metadata.get('iqr_mult', 3.0)),
        # threshold for trend intensity
        threshold=float(metadata.get('threshold', 0.8)),
        # window size for nowcasting features
        window=int(metadata.get('window', 5)),
        # n_fast parameter for nowcasting MACD features
        n_fast=int(metadata.get('n_fast', 26)),
        # n_slow parameter for nowcasting MACD features
        n_slow=int(metadata.get('n_slow', 5)),
        # which features/groups to opt-in for calculation
        selected_features=selected_features
    )

    # extract output "id" column
    output = {'id': key}

    # feature names (after opt-in selection)
    feature_names = list(featurizer.final_filter.keys())

    # calculate time series features and append them to output
    try:
        output_features = featurizer.transform(ts_obj)
    except Exception as e:
        if len(ts_obj) < 5:
            # raise error when length of time series is less than 5, see tsfeatures.py
            msg = "Length of time series is too short. Unable to calculate features for this time series!"
        else:
            msg = f"Failed to featurize time series: {e}"
        logging.error(msg)
        output_features = {name: np.nan for name in feature_names}

    output.update(output_features)

    yield output


if __name__ == '__main__':
    HiveTransformer(transform_func=transform).run()
