import pandas as pd

def ROC(df, n, column = 'y'):
    M = df[column].diff(n - 1)
    N = df[column].shift(n - 1)
    if column == 'y':
        ROC = pd.Series(M / N, name = 'ROC_' + str(n))
    else:
        ROC = pd.Series(M / N, name = column +'_ROC_' + str(n))
    df = df.join(ROC)
    return df

def MOM(df, n, column = 'y'):
    if column == 'y':
        M = pd.Series(df[column].diff(n), name = 'MOM_' + str(n))
    else:
        M = pd.Series(df[column].diff(n), name = column +'_MOM_' + str(n))
    df = df.join(M)
    return df

def MA(df, n, column = 'y'):
    if column == 'y':
        MA = pd.Series(df[column].rolling(n).mean(), name = 'MA_' + str(n))
    else:
        MA = pd.Series(df[column].rolling(n).mean(), name = column +'_MA_' + str(n))
    df = df.join(MA)
    return df

def LAG(df, n, column = 'y'):
    N = df[column].shift(n)
    if column == 'y':
        LAG = pd.Series(N, name = 'LAG_' + str(n))
    else:
        LAG = pd.Series(N, name = column + '_LAG_' + str(n))
    df = df.join(LAG)
    return df

def MACD(df, n_fast=12, n_slow=21, column = 'y'):
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
