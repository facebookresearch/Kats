# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Collection of methods that return default search spaces for their relevant models.

This module has a collection of functions. Each function is dedicated for a model. It
returns default search space for hyperparameter tuning that pertains to the model.
They are called by hyperparemeter tuning module, time_series_parameter_tuning.py.

  Typical usage example:

  SearchMethodFactory.create_search_method(get_default_prophet_parameter_search_space(), ...)
  SearchMethodFactory.create_search_method(get_default_arnet_parameter_search_space(), ...)
"""


from typing import Dict, List, Union

import numpy as np


def get_default_prophet_parameter_search_space() -> List[
    Dict[str, Union[str, list, bool]]
]:
    """Generates default search space as a list of dictionaries and returns it for prophet model.

    Each dictionary in the list corresponds to a hyperparameter, having properties
    defining that hyperparameter. Properties are name, type, value_type, values,
    is_ordered. Hyperparameters that are included: seasonality_prior_scale, yearly_seasonality,
    weekly_seasonality, daily_seasonality, seasonality_mode, changepoint_prior_scale,
    changepoint_range.

    Args:
        N/A

    Returns:
        As described above

    Raises:
        N/A
    """

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


def get_default_arnet_parameter_search_space() -> List[
    Dict[str, Union[str, list, bool]]
]:
    """Generates default search space as a list of dictionaries and returns it for arnet.

    Each dictionary in the list corresponds to a hyperparameter, having properties
    defining that hyperparameter. Properties are name, type, value_type, values,
    is_ordered. Hyperparameters that are included: input_size, output_size, batch_size.

    Args:
        N/A

    Returns:
        As described above

    Raises:
        N/A
    """

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


def get_default_stlf_parameter_search_space() -> List[
    Dict[str, Union[str, list, bool]]
]:
    """Generates default search space as a list of dictionaries and returns it for stfl.

    Each dictionary in the list corresponds to a hyperparameter, having properties
    defining that hyperparameter. Properties are name, type, value_type, values,
    is_ordered. Hyperparameters that are included: method, m.

    Args:
        N/A

    Returns:
        As described above

    Raises:
        N/A
    """

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


def get_default_arima_parameter_search_space() -> List[
    Dict[str, Union[str, list, bool]]
]:
    """Generates default search space as a list of dictionaries and returns it for arima.

    Each dictionary in the list corresponds to a hyperparameter, having properties
    defining that hyperparameter. Properties are name, type, value_type, values,
    is_ordered. Hyperparameters that are included: p, d, q.

    Args:
        N/A

    Returns:
        As described above

    Raises:
        N/A
    """

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


def get_default_holtwinters_parameter_search_space() -> List[
    Dict[str, Union[str, list, bool]]
]:
    """Generates default search space as a list of dictionaries and returns it for holtwinters.

    Each dictionary in the list corresponds to a hyperparameter, having properties
    defining that hyperparameter. Properties are name, type, value_type, values,
    is_ordered. Hyperparameters that are included: trend, damped, seasonal, seasonal_periods.

    Args:
        N/A

    Returns:
        As described above

    Raises:
        N/A
    """

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


def get_default_sarima_parameter_search_space() -> List[
    Dict[str, Union[str, list, bool]]
]:
    """Generates default search space as a list of dictionaries and returns it for sarima.

    Each dictionary in the list corresponds to a hyperparameter, having properties
    defining that hyperparameter. Properties are name, type, value_type, values,
    is_ordered. Hyperparameters that are included: p, d, q, seasonal_order, trend.

    Args:
        N/A

    Returns:
        As described above

    Raises:
        N/A
    """

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
            "values": [
                (1, 0, 1, 7),
                (1, 0, 2, 7),
                (2, 0, 1, 7),
                (2, 0, 2, 7),
                (1, 1, 1, 7),
                (0, 1, 1, 7),
            ],
            # Note: JSON representation must be 'int', 'float', 'bool' or 'str'.
            # so we use 'str' here instead of 'Tuple'
            # when doing HPT, we need to convert it back to tuple
            "value_type": "str",
        },
        {
            "name": "trend",
            "type": "choice",
            "values": ["n", "c", "t", "ct"],
            "value_type": "str",
        },
    ]


def get_default_theta_parameter_search_space() -> List[
    Dict[str, Union[str, list, bool]]
]:
    """Generates default search space as a list of dictionaries and returns it for theta.

    Each dictionary in the list corresponds to a hyperparameter, having properties
    defining that hyperparameter. Properties are name, type, value_type, values,
    is_ordered. Hyperparameters that are included: m.

    Args:
        N/A

    Returns:
        As described above

    Raises:
        N/A
    """

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


def get_default_var_parameter_search_space() -> List[Dict[str, Union[str, list, bool]]]:
    """Generates default search space as a list of dictionaries and returns it for var.

    Each dictionary in the list corresponds to a hyperparameter, having properties
    defining that hyperparameter. Properties are name, type, value_type, values,
    is_ordered. Hyperparameters that are included: To be filled.

    Args:
        N/A

    Returns:
        As described above

    Raises:
        NotImplementedError: The method is to be implemented in the future. This
        error will then be removed.
    """

    # TODO: remove raise error, then implement the default parameter
    # space definition.
    raise NotImplementedError(
        "get_parameter_search_space() method has not been implemented for " "VAR model."
    )
