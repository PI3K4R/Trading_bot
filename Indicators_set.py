from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import pandas as pd
from collections import Counter
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

def PnL_scorer(y_test, y_pred):
    conf_mtrx = confusion_matrix(y_test, y_pred, labels=[1, 0, -1])
    total_trades = conf_mtrx[:,[0,2]].sum()
    if total_trades == 0:
        return 0.0

    TA = (conf_mtrx[0,0] + conf_mtrx[2,2])/total_trades
    coverage = total_trades/y_test.size
    return coverage*(2*TA - 1)


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



df = pd.read_csv("C:/Users/Jakub Piekarski/OneDrive/Pulpit/HistoricalData_1771071541863.csv", index_col='Date')
df = df.iloc[::-1]

df.index = df.index.astype("datetime64[ns]")
# parameters to be set (the most important: names of columns in dataframe)
volume = 'Volume'
close_price = 'Close/Last'
open_price = 'Open'
highest_price = 'High'
lowest_price = 'Low'
r = 0.03
future_candles = 10


df[close_price] = df[close_price].str.replace(r"^\$", "", regex=True)
df[open_price] = df[open_price].str.replace(r"^\$", "", regex=True)
df[highest_price] = df[highest_price].str.replace(r"^\$", "", regex=True)
df[lowest_price] = df[lowest_price].str.replace(r"^\$", "", regex=True)
df[[close_price, highest_price, lowest_price, open_price, volume]] = df[[close_price, highest_price, lowest_price, open_price, volume]].astype("float64")
labels = np.zeros(len(df[close_price]) - future_candles)

df.tail(15).to_csv("NVIDIA_price_data_branch.csv", na_rep="NaN")
df.to_csv("NVIDIA_price_data_main.csv", na_rep="NaN")
log_close = np.log(df[close_price].clip(lower=1e-8))
log_volume = np.log(df[volume].clip(lower=1e-8))


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
RSI_14 = relative_strength_index(df[close_price], 14).iloc[:-future_candles]

# CMF
CMF_5 = chaikin_money_flow(df[highest_price], df[lowest_price], df[close_price], df[volume], 5)
CMF_5_delta_3 = CMF_5.diff(3)

# MFI
MFI_5 = money_flow_index(df[highest_price], df[lowest_price], df[close_price], df[volume], 5)
MFI_5_delta_3 = MFI_5.diff(3)

# VWMA
VWMA_5 = volume_weighted_moving_average(df[volume], 5, df[close_price])
VWMA_5_price_diff = (np.log(VWMA_5).clip(lower=1e-8) - log_close).iloc[:-future_candles]
VWMA_5_slope_3 = slope(VWMA_5, 3).iloc[:-future_candles]


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
    "VWMA Trend": VWMA_5_slope_3})

indicators_df1 = indicators_df1.dropna()
indicators_df1.to_csv('NVIDIA_indicators_dataset.csv', na_rep="NaN")
quit()

X = indicators_df1.iloc[:, :-1]
y = indicators_df1.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
spl = TimeSeriesSplit(n_splits=10, gap=10).split(X_train, y_train)

pipe_svc = make_pipeline(
        StandardScaler(),
        SVC(kernel='linear', C= 0.8)
    )


scores = []
for train, test in spl:
    pipe_svc.fit(X_train.iloc[train], y_train.iloc[train])
    y_pred = pipe_svc.predict(X_train.iloc[test])
    score = PnL_scorer(y_test=y_train.iloc[test], y_pred=y_pred)
    print(score)
    scores.append(score)


y_pred = pipe_svc.predict(X_test)
print(PnL_scorer(y_test=y_test, y_pred=y_pred))
