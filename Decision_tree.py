from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

def PnL_scorer(y_test, y_pred):
    conf_mtrx = confusion_matrix(y_test, y_pred, labels=[1, 0, -1])
    total_trades = conf_mtrx[:,[0,2]].sum()
    if total_trades == 0:
        return 0.0

    TA = (conf_mtrx[0,0] + conf_mtrx[2,2])/total_trades
    coverage = total_trades/y_test.size
    return coverage*(2*TA - 1)

df = pd.read_csv("data_files\TSLA_short_term_r=3.0f=15.csv", index_col='date')
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
single_decision_tree = DecisionTreeClassifier(criterion="gini", max_depth=10)
tscv = TimeSeriesSplit(n_splits=10, gap=15)
spl = list(tscv.split(X_train, y_train))

scores = []
for k, (train, test) in enumerate(spl):
    single_decision_tree.fit(X_train.iloc[train], y_train.iloc[train])
    y_pred = single_decision_tree.predict(X_train.iloc[test])
    score = PnL_scorer(y_test=y_train.iloc[test], y_pred=y_pred)
    scores.append(score)

mean_acc = np.mean(scores)
print("Train accuracy single tree: ", mean_acc)

y_test_pred_single = single_decision_tree.predict(X_test)
print("Test accuracy single tree: ", PnL_scorer(y_test=y_test, y_pred=y_test_pred_single))

cum_pnl = np.cumsum(y_test_pred_single * y_test)
plt.plot(cum_pnl)
plt.title("Cumulative PnL - Test")
plt.show()

forest = RandomForestClassifier(100, criterion='gini')


scores = []
for k, (train, test) in enumerate(spl):
    forest.fit(X_train.iloc[train], y_train.iloc[train])
    y_pred = forest.predict(X_train.iloc[test])
    score = PnL_scorer(y_test=y_train.iloc[test], y_pred=y_pred)
    scores.append(score)

mean_acc = np.mean(scores)
print("Train accuracy forest: ", mean_acc)

y_test_pred_forest = forest.predict(X_test)
print("Test accuracy forest: ", PnL_scorer(y_test=y_test, y_pred=y_test_pred_forest))
