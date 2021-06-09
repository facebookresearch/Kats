#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This is a file with functions which turn time series into ML features.

Typical use case is to create various features for the nowcasting model.
The features are rolling, i.e. they are the times series as well.

  Typical usage example:

  >>> df = ROC(df, 5)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional

def ROC(df: pd.DataFrame, n: int, column: str = 'y') -> pd.DataFrame:
    """Adds another column indicating return comparing to step n back.

    Args:
        df: a pandas dataframe.
        n: an integer on how many steps looking back.
        column: Optional. If column is provided, will calculate based on provided column
            otherwise the column named y will be the target.

    Returns:
        A dataframe with all the columns from input df, and the added column.
    """

    M = df[column].diff(n - 1)
    N = df[column].shift(n - 1)
    if column == 'y':
        ROC = pd.Series(M / N, name = 'ROC_' + str(n))
    else:
        ROC = pd.Series(M / N, name = column +'_ROC_' + str(n))
    df = df.join(ROC)
    return df

def MOM(df: pd.DataFrame, n: int, column: str = 'y') -> pd.DataFrame:
    """Adds another column indicating momentum: difference of current value and n steps back.

    Args:
        df: a pandas dataframe.
        n: an integer on how many steps looking back.
        column: Optional. If column is provided, will calculate based on provided column
            otherwise the column named y will be the target.

    Returns:
        A dataframe with all the columns from input df, and the added column.
    """

    if column == 'y':
        M = pd.Series(df[column].diff(n), name = 'MOM_' + str(n))
    else:
        M = pd.Series(df[column].diff(n), name = column +'_MOM_' + str(n))
    df = df.join(M)
    return df

def MA(df: pd.DataFrame, n: int, column: str = 'y') -> pd.DataFrame:
    """Adds another column indicating moving average in the past n steps.

    Args:
        df: a pandas dataframe.
        n: an integer on how many steps looking back.
        column: Optional. If column is provided, will calculate based on provided column
            otherwise the column named y will be the target.

    Returns:
        A dataframe with all the columns from input df, and the added column.
    """

    if column == 'y':
        MA = pd.Series(df[column].rolling(n).mean(), name = 'MA_' + str(n))
    else:
        MA = pd.Series(df[column].rolling(n).mean(), name = column +'_MA_' + str(n))
    df = df.join(MA)
    return df

def LAG(df: pd.DataFrame, n: int, column: str = 'y') -> pd.DataFrame:
    """Adds another column indicating lagged value at the past n steps.

    Args:
        df: a pandas dataframe.
        n: an integer on how many steps looking back.
        column: Optional. If column is provided, will calculate based on provided column
            otherwise the column named y will be the target.

    Returns:
        A dataframe with all the columns from input df, and the added column.
    """

    N = df[column].shift(n)
    if column == 'y':
        LAG = pd.Series(N, name = 'LAG_' + str(n))
    else:
        LAG = pd.Series(N, name = column + '_LAG_' + str(n))
    df = df.join(LAG)
    return df

def MACD(df: pd.DataFrame, n_fast: int =12, n_slow: int =21, column: str = 'y') -> pd.DataFrame:
    """Adds three columns indicating MACD: https://www.investopedia.com/terms/m/macd.asp.

    Args:
        df: a pandas dataframe
        n_fast: an integer on how many steps looking back fast.
        n_slow: an integer on how many steps looking back slow.
        column: Optional. If column is provided, will calculate based on provided column
            otherwise the column named y will be the target.

    Returns:
        A dataframe with all the columns from input df, and the added 3 columns.
    """

    EMAfast = pd.Series(df[column].ewm( span = n_fast, min_periods = n_slow - 1).mean())
    EMAslow = pd.Series(df[column].ewm( span = n_slow, min_periods = n_slow - 1).mean())
    if column == 'y':
        MACD = pd.Series(EMAfast - EMAslow, name = 'MACD_' + str(n_fast) + '_' + str(n_slow))
        MACDsign = pd.Series(MACD.ewm( span = 9, min_periods = 8).mean(), name = 'MACDsign_' + str(n_fast) + '_' + str(n_slow))
        MACDdiff = pd.Series(MACD - MACDsign, name = 'MACDdiff_' + str(n_fast) + '_' + str(n_slow))
    else:
        MACD = pd.Series(EMAfast - EMAslow, name = column + '_MACD_' + str(n_fast) + '_' + str(n_slow))
        MACDsign = pd.Series(MACD.ewm( span = 9, min_periods = 8).mean(), name = column + '_MACDsign_' + str(n_fast) + '_' + str(n_slow))
        MACDdiff = pd.Series(MACD - MACDsign, name = column + '_MACDdiff_' + str(n_fast) + '_' + str(n_slow))
    df = df.join(MACD)
    df = df.join(MACDsign)
    df = df.join(MACDdiff)
    return df



    def _ewma(
        arr: np.ndarray,
        span: int,
        min_periods: int
    ):
        """
        Exponentialy weighted moving average specified by a decay ``window``
        to provide better adjustments for small windows via:
            y[t] = (x[t] + (1-a)*x[t-1] + (1-a)^2*x[t-2] + ... + (1-a)^n*x[t-n]) /
                   (1 + (1-a) + (1-a)^2 + ... + (1-a)^n).

        Args:
        arr : np.ndarray; A single dimenisional numpy array.
        span : int; The decay window, or 'span'.
        min_periods: int; Minimum amount of data points we'd like to include in the output.

        Returns:
            A np.ndarray. The exponentially weighted moving average of the array.
        """
        output_array = np.empty(arr.shape[0], dtype=np.float64)
        output_array[:] = np.NaN

        arr = arr[~np.isnan(arr)]
        n = arr.shape[0]
        ewma = np.empty(n, dtype=np.float64)
        alpha = 2 / float(span + 1)
        w = 1
        ewma_old = arr[0]
        ewma[0] = ewma_old
        for i in range(1, n):
            w += (1-alpha)**i
            ewma_old = ewma_old*(1-alpha) + arr[i]
            ewma[i] = ewma_old / w

        output_subset = ewma[(min_periods-1):]
        output_array[-len(output_subset):] = output_subset
        return output_array

    def _get_nowcasting_np(
        x: np.ndarray,
        window: int = 5,
        n_fast: int = 12,
        n_slow: int = 21,
        extra_args: Optional[Dict[str, bool]] = None,
        default_status: bool = True,
    ):
        """
        Internal function for actually performing feature engineering using the same procedure as
        nowcasting feature_extraction under kats/models.

        Args:
            x: The univariate time series array in the form of 1d numpy array.
            window: int; Length of window size for all Nowcasting features. Default value is 5.
            n_fast: int; length of "fast" or short period exponential moving average in the MACD
                algorithm in the nowcasting features. Default value is 12.
            n_slow: int; length of "slow" or long period exponential moving average in the MACD
                algorithm in the nowcasting features. Default value is 21.
            extra_args: A dictionary containing information for disabling calculation
                of a certain feature. Default value is None, i.e. no feature is disabled.
            default_status: Default status of the switch for calculate the features or not.
                Default value is True.

        Returns:
            A list containing extracted nowcast features.
        """

        # initializing the outputs
        nowcasting_features = [np.nan for _ in range(7)]

        # ROC: indicating return comparing to step n back.
        if extra_args is not None and extra_args.get("nowcast_roc", default_status):
            M = x[(window-1):] - x[:-(window-1)]
            N = x[:-(window-1)]
            arr = M / N
            nowcasting_features[0] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).mean()

        # MOM: indicating momentum: difference of current value and n steps back.
        if extra_args is not None and extra_args.get("nowcast_mom", default_status):
            M = x[window:] - x[:-window]
            nowcasting_features[1] = np.nan_to_num(M, nan=0.0, posinf = 0.0, neginf=0.0).mean()

        # MA: indicating moving average in the past n steps.
        if extra_args is not None and extra_args.get("nowcast_ma", default_status):
            ret = np.cumsum(x, dtype=float)
            ret[window:] = ret[window:] - ret[:-window]
            ma = ret[window - 1:] / window
            nowcasting_features[2] = np.nan_to_num(ma, nan=0.0, posinf=0.0, neginf=0.0).mean()

        # LAG: indicating lagged value at the past n steps.
        if extra_args is not None and extra_args.get("nowcast_lag", default_status):
            N = x[:-window]
            nowcasting_features[3] = np.nan_to_num(N, nan=0.0, posinf=0.0, neginf=0.0).mean()

        # MACD: https://www.investopedia.com/terms/m/macd.asp.
        ema_fast = _ewma(x, n_fast, n_slow-1)
        ema_slow = _ewma(x, n_slow, n_slow-1)
        MACD = ema_fast - ema_slow
        if extra_args is not None and extra_args.get("nowcast_macd", default_status):
            nowcasting_features[4] = np.nan_to_num(np.nanmean(MACD), nan=0.0, posinf=0.0, neginf=0.0)

        MACDsign = _ewma(MACD, 9, 8)
        if extra_args is not None and extra_args.get("nowcast_macdsign", default_status):
            nowcasting_features[5] = np.nan_to_num(np.nanmean(MACDsign), nan=0.0, posinf=0.0, neginf=0.0)

        MACDdiff = MACD - MACDsign
        if extra_args is not None and extra_args.get("nowcast_macddiff", default_status):
            nowcasting_features[6] = np.nan_to_num(np.nanmean(MACDdiff), nan=0.0, posinf=0.0, neginf=0.0)

        return nowcasting_features
