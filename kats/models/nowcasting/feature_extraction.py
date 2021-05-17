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
