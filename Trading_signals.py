from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
import numpy as np


def PnL_scorer(y_test, y_pred):
    conf_mtrx = confusion_matrix(y_test, y_pred, labels=[1, 0, -1])
    total_trades = conf_mtrx[:,[0,2]].sum()
    if total_trades == 0:
        return 0.0

    TA = (conf_mtrx[0,0] + conf_mtrx[2,2])/total_trades
    coverage = total_trades/y_test.size
    return coverage*(2*TA - 1)


indicators_data = pd.read_csv("NVIDIA_indicators_dataset.csv", index_col='Date')
signal_data = pd.read_csv("NVIDIA_indicators_dataset_temp.csv", index_col="Date")
indicators_df = indicators_data.iloc[:-10]
price_data = pd.read_csv("NVIDIA_price_data_main.csv", index_col="Date")
price_data_branch = pd.read_csv("NVIDIA_price_data_branch.csv", index_col="Date")

labels = np.zeros(len(indicators_data)-10)

for i in range(1, labels.size):
    future_cndls = price_data.iloc[i: i+10].reset_index(drop=True)

    # creating labels -1 if short 0 if consolidation 1 if long
    for idx in range(10):
        if future_cndls.loc[idx, "High"] > price_data.iloc[i - 1]["Close/Last"]*(1.03) and future_cndls.loc[idx, "Low"] < price_data.iloc[i - 1]["Close/Last"]*(0.97):
            break
        elif future_cndls.loc[idx, "High"] > price_data.iloc[i - 1]["Open"] * (1.03):
            labels[i-1] = 1
            break
        elif future_cndls.loc[idx, "Low"] < price_data.iloc[i-1]["Open"] * (0.97):
            labels[i-1] = -1
            break

indicators_df["Label"] = labels

X = indicators_df.iloc[:, :-1]
y = indicators_df.iloc[:, -1]
spl = TimeSeriesSplit(n_splits=10, gap=10).split(X, y)

pipe_svc = make_pipeline(
        StandardScaler(),
        SVC(kernel='linear', C= 0.8)
    )


scores = []
for train, test in spl:
    pipe_svc.fit(X.iloc[train], y.iloc[train])
    y_pred = pipe_svc.predict(X.iloc[test])
    score = PnL_scorer(y_test=y.iloc[test], y_pred=y_pred)
    print(score)
    scores.append(score)

close = price_data_branch.iloc[-1]["Close/Last"]

signal = pipe_svc.predict(signal_data)[0]

if signal == 1.0:
    print("LONG")
    SL = round(close*0.97, 2)
    TP = round(close*1.03, 2)
    print("Stop Loss: ", SL, "\n", "Take Profit: ", TP)

elif signal == -1.0:
    print("SHORT")
    SL = round(close*1.03, 2)
    TP = round(close*0.97, 2)
    print("Stop Loss: ", SL, "\nTake Profit: ", TP)

else:
    print("DO NOTHING")

