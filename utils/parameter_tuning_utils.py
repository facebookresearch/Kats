#!/usr/bin/env python3
from typing import Dict, List

import numpy as np


def get_default_prophet_parameter_search_space() -> List[Dict[str, object]]:
    return [
        {
            "name": "seasonality_prior_scale",
            "type": "choice",
            "value_type": "float",
            "values": list(np.logspace(-2, 1, 10, endpoint=True)),
            "is_ordered": True,
        },
        {
            "name": "yearly_seasonality",
            "type": "choice",
            "value_type": "bool",
            "values": [True, False],
        },
        {
            "name": "weekly_seasonality",
            "type": "choice",
            "value_type": "bool",
            "values": [True, False],
        },
        {
            "name": "daily_seasonality",
            "type": "choice",
            "value_type": "bool",
            "values": [True, False],
        },
        {
            "name": "seasonality_mode",
            "type": "choice",
            "value_type": "str",
            "values": ["additive", "multiplicative"],
        },
        {
            "name": "changepoint_prior_scale",
            "type": "choice",
            "value_type": "float",
            "values": list(np.logspace(-3, 0, 10, endpoint=True)),
            "is_ordered": True,
        },
        {
            "name": "changepoint_range",
            "type": "choice",
            "value_type": "float",
            "values": list(np.arange(0.8, 0.96, 0.01)),  # last value is 0.95
            "is_ordered": True,
        },
    ]


def get_default_arnet_parameter_search_space() -> List[Dict[str, object]]:
    return [
        {
            "name": "input_size",
            "type": "choice",
            "values": list(range(3, 14)),
            "value_type": "int",
            "is_ordered": True,
        },
        {
            "name": "output_size",
            "type": "choice",
            "values": list(range(3, 14)),
            "value_type": "int",
            "is_ordered": True,
        },
        {
            "name": "batch_size",
            "type": "choice",
            "values": list(range(5, 20)),
            "value_type": "int",
            "is_ordered": True,
        },
    ]


def get_default_stlf_parameter_search_space() -> List[Dict[str, object]]:
    return [
        {
            "name": "method",
            "type": "choice",
            "value_type": "str",
            "values": ["linear", "quadratic", "theta", "prophet"],
        },
        {
            "name": "m",
            "type": "choice",
            # The number of periods in this seasonality
            # (e.g. 7 periods for daily data would be used for weekly seasonality)
            "values": [4, 7, 10, 14, 24, 30],
            "value_type": "int",
            "is_ordered": True,
        },
    ]


def get_default_arima_parameter_search_space() -> List[Dict[str, object]]:
    return [
        {
            "name": "p",
            "type": "choice",
            "values": list(range(1, 6)),
            "value_type": "int",
            "is_ordered": True,
        },
        {
            "name": "d",
            "type": "choice",
            "values": list(range(1, 3)),
            "value_type": "int",
            "is_ordered": True,
        },
        {
            "name": "q",
            "type": "choice",
            "values": list(range(1, 6)),
            "value_type": "int",
            "is_ordered": True,
        },
    ]


def get_default_holtwinters_parameter_search_space() -> List[Dict[str, object]]:
    return [
        {
            "name": "trend",
            "type": "choice",
            "value_type": "str",
            "values": ["additive", "multiplicative"],
        },
        {
            "name": "damped",
            "type": "choice",
            "value_type": "bool",
            "values": [True, False],
        },
        {
            "name": "seasonal",
            "type": "choice",
            "value_type": "str",
            "values": ["additive", "multiplicative"],
        },
        {
            "name": "seasonal_periods",
            "type": "choice",
            # The number of periods in this seasonality
            # (e.g. 7 periods for daily data would be used for weekly seasonality)
            "values": [4, 7, 10, 14, 24, 30],
            "value_type": "int",
            "is_ordered": True,
        },
    ]


def get_default_sarima_parameter_search_space() -> List[Dict[str, object]]:
    return [
        {
            "name": "p",
            "type": "choice",
            "values": list(range(1, 6)),
            "value_type": "int",
            "is_ordered": True,
        },
        {
            "name": "d",
            "type": "choice",
            "values": list(range(1, 3)),
            "value_type": "int",
            "is_ordered": True,
        },
        {
            "name": "q",
            "type": "choice",
            "values": list(range(1, 6)),
            "value_type": "int",
            "is_ordered": True,
        },
        {
            "name": "seasonal_order",
            "type": "choice",
            "values": [(1, 0, 1, 7), (1, 0, 2, 7), (2, 0, 1, 7), (2, 0, 2, 7), (1, 1, 1, 7), (0, 1, 1, 7)],
            # Note: JSON representation must be 'int', 'float', 'bool' or 'str'.
            # so we use 'str' here instead of 'Tuple'
            # when doing HPT, we need to convert it back to tuple
            "value_type": "str",
        },
        {
            "name": "trend",
            "type": "choice",
            "values": ['n', 'c', 't', 'ct'],
            "value_type": "str",
        },
    ]


def get_default_theta_parameter_search_space() -> List[Dict[str, object]]:
    return [
        {
            "name": "m",
            "type": "choice",
            # Number of observations before the seasonal pattern repeats
            # e.g. m=12 for montly data with yearly seasonality
            "values": list(range(1, 31)),
            "value_type": "int",
            "is_ordered": True,
        },
    ]


def get_default_var_parameter_search_space() -> List[Dict[str, object]]:
    # TODO: remove raise error, then implement the default parameter
    # space definition.
    raise NotImplementedError(
        "get_parameter_search_space() method has not been implemented for "
        "VAR model."
    )
