from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline


def PnL_scorer(y_test, y_pred):
    conf_mtrx = confusion_matrix(y_test, y_pred, labels=[1, 0, -1])
    total_trades = conf_mtrx[:,[0,2]].sum()
    if total_trades == 0:
        return 0.0

    TA = (conf_mtrx[0,0] + conf_mtrx[2,2])/total_trades
    coverage = total_trades/y_test.size
    return coverage*(2*TA - 1)


indicators_df = pd.read_csv("NVIDIA_price_data_main.csv", index_col="Date")

labels = np.zeros(len(df[close_price]) - future_candles)

for i in range(1, labels.size):
    future_cndls = df.iloc[i: i + future_candles].reset_index(drop=True)

    # creating labels -1 if short 0 if consolidation 1 if long
    for idx in range(future_candles):
        if future_cndls.loc[idx, highest_price] > df.iloc[i - 1][close_price] * (1 + r) and future_cndls.loc[
            idx, lowest_price] < df.iloc[i - 1][close_price] * (1 - r):
            break
        elif future_cndls.loc[idx, highest_price] > df.iloc[i - 1][open_price] * (1 + r):
            labels[i - 1] = 1
            break
        elif future_cndls.loc[idx, lowest_price] < df.iloc[i - 1][open_price] * (1 - r):
            labels[i - 1] = -1
            break
X = indicators_df.iloc[:, :-1]
y = indicators_df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
spl = TimeSeriesSplit(n_splits=5, gap=10).split(X_train, y_train)

pipe_svc = make_pipeline(
        StandardScaler(),
        SVC(kernel='linear', C=0.8)
    )


scores = []
for train, test in spl:
    pipe_svc.fit(X_train.iloc[train], y_train.iloc[train])
    y_pred = pipe_svc.predict(X_train.iloc[test])
    score = PnL_scorer(y_test=y_train.iloc[test], y_pred=y_pred)
    print(score)
    scores.append(score)





