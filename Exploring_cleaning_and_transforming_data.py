import numpy as np
import pandas as pd
from collections import Counter

def simple_moving_average(x: pd.Series, num_of_past_candles) -> pd.Series:
    return x.rolling(window=num_of_past_candles, min_periods=1).mean()

def exponential_moving_average(x: pd.Series, param) -> pd.Series:
    alpha = 2 / (param + 1)
    return x.ewm(alpha=alpha, adjust=False).mean()

def relative_strength_index(x: pd.Series, param: int=14) -> pd.Series:
    delta = x.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/param, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/param, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def chaikin_money_flow(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, n: int) -> pd.Series:
    high_low = high - low
    clv = ((close - low) - (high - close)) / high_low.replace(0, np.nan)
    mf = clv * volume
    cmf = mf.rolling(window=n, min_periods=1).sum() / volume.rolling(window=n, min_periods=1).sum()
    return cmf

def typical_price(high: pd.Series, low: pd.Series, close: pd.Series):
    return (high + low + close) / 3

def money_flow_index(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, n: int):
    tp = typical_price(high, low, close)
    mf = tp * volume
    pos = mf.where(tp > tp.shift(1), 0.0)
    neg = mf.where(tp < tp.shift(1), 0.0)
    sum_pos = pos.rolling(window=n, min_periods=1).sum()
    sum_neg = neg.rolling(window=n, min_periods=1).sum()
    mfr = sum_pos / sum_neg
    mfi = 100 - (100 / (1 + mfr))
    return mfi

def volume_weighted_moving_average(volume: pd.Series, n_of_past_candles, close: pd.Series):
    num = (close * volume).rolling(window=n_of_past_candles, min_periods=1).sum()
    den = volume.rolling(window=n_of_past_candles, min_periods=1).sum()
    return num / den

def slope(series: pd.Series, window: int):
    log_series = np.log(series.clip(lower=1e-8))
    def _slope(y):
        x = np.arange(len(y))
        return np.polyfit(x, y, 1)[0]

    return log_series.rolling(window=window, min_periods=window).apply(_slope, raw=True)

def creating_datasets(filename_without_csv_pref, future_candles, r):

    """
    :param filename_without_csv_pref: name of the file we want to store transformed data,
    :param future_candles: amount of  periods at most we are waiting for price to hit take profit or stop loss
    :param r: risk parameter
    :return: dataset of indicators for given financial asset
    """


    df = pd.read_csv(f"{filename_without_csv_pref}.csv", skiprows=14, index_col='date')
    print(df.head())

    # parameters to be set (the most important: names of columns in dataframe)
    volume = 'volume'
    close_price = 'close'
    open_price = 'open'
    highest_price = 'high'
    lowest_price = 'low'

    print("Risk ratio: ", r)

    log_close = np.log(df[close_price].clip(lower=1e-8))
    log_volume = np.log(df[volume].clip(lower=1e-8))

    # slopes for price and volume
    slopes_price_5 = slope(log_close, 5).iloc[:-future_candles]
    slopes_price_10 = slope(log_close, 10).iloc[:-future_candles]
    slopes_price_20 = slope(log_close, 20).iloc[:-future_candles]
    slopes_vols_5 = slope(log_volume, 5).iloc[:-future_candles]
    slopes_vols_10 = slope(log_volume, 10).iloc[:-future_candles]
    slopes_vols_20 = slope(log_volume, 20).iloc[:-future_candles]

    #Standard Moving Average
    SMA_5 = simple_moving_average(log_close, 5)
    SMA_5_price_diff = (log_close - SMA_5).iloc[:-future_candles]
    SMA_5_delta_3 = slope(SMA_5, 3).iloc[:-future_candles]
    SMA_10 = simple_moving_average(log_close, 10)
    SMA_10_price_diff = (log_close - SMA_10).iloc[:-future_candles]
    SMA_10_delta_5 = slope(SMA_10, 10).iloc[:-future_candles]
    SMA_20 = simple_moving_average(log_close, 20)
    SMA_20_price_diff = (log_close - SMA_20).iloc[:-future_candles]
    SMA_20_delta_10 = slope(SMA_20, 10).iloc[:-future_candles]

    #Exponential Moving Average
    EMA_5 = exponential_moving_average(log_close, 5)
    EMA_5_price_diff = (log_close - EMA_5).iloc[:-future_candles]
    EMA_5_delta_3 = slope(EMA_5, 3).iloc[:-future_candles]
    EMA_10 = exponential_moving_average(log_close, 10)
    EMA_10_price_diff = (log_close - EMA_10).iloc[:-future_candles]
    EMA_10_delta_5 = slope(EMA_10, 5).iloc[:-future_candles]
    EMA_20 = exponential_moving_average(log_close, 20)
    EMA_20_price_diff = (log_close - EMA_20).iloc[:-future_candles]
    EMA_20_delta_10 = slope(EMA_10, 10).iloc[:-future_candles]

    #RSI
    RSI_14 = relative_strength_index(df[close_price], 14).iloc[:-future_candles]

    #CMF
    CMF_5 = chaikin_money_flow(df[highest_price], df[lowest_price], df[close_price], df[volume], 5).iloc[:-future_candles]
    CMF_5_delta_3 = CMF_5.diff(3)
    CMF_10 = chaikin_money_flow(df[highest_price], df[lowest_price], df[close_price], df[volume], 10).iloc[:-future_candles]
    CMF_10_delta_5 = CMF_10.diff(5)
    CMF_20 = chaikin_money_flow(df[highest_price], df[lowest_price], df[close_price], df[volume], 20).iloc[:-future_candles]
    CMF_20_delta_10 = CMF_20.diff(10)

    #MFI
    MFI_5 = money_flow_index(df[highest_price], df[lowest_price], df[close_price], df[volume], 5).iloc[:-future_candles]
    MFI_5_delta_3 = MFI_5.diff(3)
    MFI_10 = money_flow_index(df[highest_price], df[lowest_price], df[close_price], df[volume], 10).iloc[:-future_candles]
    MFI_10_delta_5 = MFI_10.diff(5)
    MFI_20 = money_flow_index(df[highest_price], df[lowest_price], df[close_price], df[volume], 20).iloc[:-future_candles]
    MFI_20_delta_10 = MFI_20.diff(10)

    #VWMA
    VWMA_5 = volume_weighted_moving_average(df[volume], 5, df[close_price])
    VWMA_5_price_diff = (np.log(VWMA_5).clip(lower=1e-8) - log_close).iloc[:-future_candles]
    VWMA_5_slope_3 = slope(VWMA_5, 3).iloc[:-future_candles]
    VWMA_10 = volume_weighted_moving_average(df[volume], 10, df[close_price])
    VWMA_10_price_diff = (np.log(VWMA_10).clip(lower=1e-8) - log_close).iloc[:-future_candles]
    VWMA_10_slope_5 = slope(VWMA_10, 5).iloc[:-future_candles]
    VWMA_20 = volume_weighted_moving_average(df[volume], 20, df[close_price])
    VWMA_20_price_diff = (np.log(VWMA_20).clip(lower=1e-8) - log_close).iloc[:-future_candles]
    VWMA_20_slope_10 = slope(VWMA_20, 10).iloc[:-future_candles]

    labels = np.zeros(len(df[close_price])-future_candles)

    for i in range(1, labels.size):
        future_cndls = df.iloc[i: i+future_candles].reset_index(drop=True)

        # creating labels -1 if short 0 if consolidation 1 if long
        for idx in range(future_candles):
            if future_cndls.loc[idx, highest_price] > df.iloc[i - 1][close_price]*(1 + r) and future_cndls.loc[idx, lowest_price] < df.iloc[i - 1][close_price]*(1 - r):
                break
            elif future_cndls.loc[idx, highest_price] > df.iloc[i - 1][open_price] * (1 + r):
                labels[i-1] = 1
                break
            elif future_cndls.loc[idx, lowest_price] < df.iloc[i-1][open_price] * (1 - r):
                labels[i-1] = -1
                break
    labels_counter = Counter(labels)

    # Dataset with too much zeros in labels won't be considered
    if labels_counter[np.float64(0.0)] >= 0.7*labels.size:
        return 0

    print('How many in each class in train dataset: ', labels_counter)

    indicators_df1 = pd.DataFrame({
        "Price Trend": slopes_price_5,
        "Volume Trend": slopes_vols_5,
        "SMA Distance from the Price": SMA_5_price_diff,
        "SMA Trend": SMA_5_delta_3,
        "EMA Distance from the Price": EMA_5_price_diff,
        "EMA Trend": EMA_5_delta_3,
        "RSI": RSI_14,
        "CMF": CMF_5,
        "CMF Trend": CMF_5_delta_3,
        "MFI": MFI_5,
        "MFI Trend": MFI_5_delta_3,
        "VWMA Distance from the Price": VWMA_5_price_diff,
        "VWMA Trend": VWMA_5_slope_3,
        "Label": labels})

    indicators_df1 = indicators_df1.dropna()

    indicators_df1.to_csv(f'data_files/{filename_without_csv_pref}_short_term_r={r*100}f={future_candles}.csv', na_rep="NaN")

    indicators_df2 = pd.DataFrame({
        "Price Trend": slopes_price_10,
        "Volume Trend": slopes_vols_10,
        "SMA Distance from the Price": SMA_10_price_diff,
        "SMA Trend": SMA_10_delta_5,
        "EMA Distance from the Price": EMA_10_price_diff,
        "EMA Trend": EMA_10_delta_5,
        "RSI": RSI_14,
        "CMF": CMF_10,
        "CMF Trend": CMF_10_delta_5,
        "MFI": MFI_10,
        "MFI Trend": MFI_10_delta_5,
        "VWMA Distance from the Price": VWMA_10_price_diff,
        "VWMA Trend": VWMA_10_slope_5,
        "Label": labels})

    indicators_df2 = indicators_df2.dropna()

    indicators_df2.to_csv(f'data_files/{filename_without_csv_pref}_medium_term_r={r*100}f={future_candles}.csv')

    indicators_df3 = pd.DataFrame({
        "Price Trend": slopes_price_20,
        "Volume Trend": slopes_vols_20,
        "SMA Distance from the Price": SMA_20_price_diff,
        "SMA Trend": SMA_20_delta_10,
        "EMA Distance from the Price": EMA_20_price_diff,
        "EMA Trend": EMA_20_delta_10,
        "RSI": RSI_14,
        "CMF": CMF_20,
        "CMF Trend": CMF_20_delta_10,
        "MFI": MFI_20,
        "MFI Trend": MFI_20_delta_10,
        "VWMA Distance from the Price": VWMA_20_price_diff,
        "VWMA Trend": VWMA_20_slope_10,
        "Label": labels})

    indicators_df3 = indicators_df3.dropna()

    indicators_df3.to_csv(f'data_files/{filename_without_csv_pref}_long_term_r={r*100}f={future_candles}.csv')


filename_list = ['GOOGL', 'ISRG', 'NVDA', 'TSLA']
future_candles_list = [5, 10, 15, 20]
risk_ratio_list = [0.01, 0.03, 0.05, 0.08, 0.1]

for filename in filename_list:
    for f in future_candles_list:
        for r in risk_ratio_list:
            creating_datasets(filename, f, r)