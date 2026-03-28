import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
df_branch = pd.read_csv("NVIDIA_price_data_branch.csv", index_col="Date")
df_branch = df_branch.iloc[:-1]
nvda = yf.Ticker("NVDA")

hist = nvda.history(period="5d", rounding=True).iloc[-2:]
close = hist["Close"]
open = hist["Open"]
high = hist["High"]
low = hist["Low"]
volume = hist["Volume"]
date = hist.index
date = date.tz_convert('UTC').tz_localize(None)
date = date.astype("datetime64[ns]")
date = date.strftime("%Y-%m-%d")
new_data = pd.DataFrame({"Close/Last": close, "Volume": volume, "Open": open, "High": high, "Low": low}, index=date)
df = pd.read_csv("NVIDIA_price_data_main.csv", index_col="Date")

df.index = df.index.astype("datetime64[ns]")
df.index = df.index.strftime("%Y-%m-%d")

df = pd.concat([df, new_data.iloc[[-2]]])
df_branch = pd.concat([df_branch, new_data.iloc[[-2]]])
df.to_csv("NVIDIA_price_data_main.csv")
df_branch = pd.concat([df_branch, new_data.iloc[[-1]]])
df_branch.to_csv("NVIDIA_price_data_branch.csv")

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



# parameters to be set (the most important: names of columns in dataframe)
volume = 'Volume'
close_price = 'Close/Last'
open_price = 'Open'
highest_price = 'High'
lowest_price = 'Low'

log_close = np.log(df_branch[close_price].clip(lower=1e-8))
log_volume = np.log(df_branch[volume].clip(lower=1e-8))



# slopes for price and volume
slopes_price_5 = slope(log_close, 5)
slopes_vols_5 = slope(log_volume, 5)

# Standard Moving Average
SMA_5 = simple_moving_average(log_close, 5)
SMA_5_price_diff = (log_close - SMA_5)
SMA_5_delta_3 = slope(SMA_5, 3)

# Exponential Moving Average
EMA_5 = exponential_moving_average(log_close, 5)
EMA_5_price_diff = (log_close - EMA_5)
EMA_5_delta_3 = slope(EMA_5, 3)

# RSI
RSI_14 = relative_strength_index(df_branch[close_price], 14)

# CMF
CMF_5 = chaikin_money_flow(df_branch[highest_price], df_branch[lowest_price], df_branch[close_price], df_branch[volume], 5)
CMF_5_delta_3 = CMF_5.diff(3)

# MFI
MFI_5 = money_flow_index(df_branch[highest_price], df_branch[lowest_price], df_branch[close_price], df_branch[volume], 5)
MFI_5_delta_3 = MFI_5.diff(3)

# VWMA
VWMA_5 = volume_weighted_moving_average(df_branch[volume], 5, df_branch[close_price])
VWMA_5_price_diff = (np.log(VWMA_5).clip(lower=1e-8) - log_close)
VWMA_5_slope_3 = slope(VWMA_5, 3)


indicators_df1 = pd.DataFrame({
    "Price Trend": slopes_price_5.iloc[[-1]],
    "Volume Trend": slopes_vols_5.iloc[[-1]],
    "SMA Distance from the Price": SMA_5_price_diff.iloc[[-1]],
    "SMA Trend": SMA_5_delta_3.iloc[[-1]],
    "EMA Distance from the Price": EMA_5_price_diff.iloc[[-1]],
    "EMA Trend": EMA_5_delta_3.iloc[[-1]],
    "RSI": RSI_14.iloc[[-1]],
    "CMF": CMF_5.iloc[[-1]],
    "CMF Trend": CMF_5_delta_3.iloc[[-1]],
    "MFI": MFI_5.iloc[[-1]],
    "MFI Trend": MFI_5_delta_3.iloc[[-1]],
    "VWMA Distance from the Price": VWMA_5_price_diff.iloc[[-1]],
    "VWMA Trend": VWMA_5_slope_3.iloc[[-1]]}
)

indicators_df1 = indicators_df1.dropna()
indicators_df1.to_csv("NVIDIA_indicators_dataset_temp.csv", na_rep="NaN")


