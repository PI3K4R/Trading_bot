from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

def PnL_scorer(y_test, y_pred):
    conf_mtrx = confusion_matrix(y_test, y_pred, labels=[1, 0, -1])
    total_trades = conf_mtrx[:,[0,2]].sum()
    if total_trades == 0:
        return 0.0

    TA = (conf_mtrx[0,0] + conf_mtrx[2,2])/total_trades
    coverage = total_trades/y_test.size
    return coverage*(2*TA - 1)

df = pd.read_csv("data_files\TSLA_short_term_r=3.0f=15.csv", index_col="date")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
knn = KNeighborsClassifier(n_neighbors=100, metric='euclidean')
cols_to_std_scaling = df.columns.difference(["RSI", "MFI", "Label"]).tolist()
spl = TimeSeriesSplit(n_splits=10, gap=15).split(X_train, y_train)
preprocessor = ColumnTransformer([
    ("StandardScaler", StandardScaler(), cols_to_std_scaling),
    ("MinMaxScaler", MinMaxScaler(feature_range=(-1, 1)), ["RSI", "MFI"])]
)
pipe_knn = make_pipeline(
    preprocessor,
    knn
)

scores = []
for k, (train, test) in enumerate(spl):
    pipe_knn.fit(X_train.iloc[train], y_train.iloc[train])
    y_pred = pipe_knn.predict(X_train.iloc[test])
    score = PnL_scorer(y_test=y_train.iloc[test], y_pred=y_pred)
    scores.append(score)

mean_acc = np.mean(scores)
print("Train accuracy: ", mean_acc)

y_test_pred = pipe_knn.predict(X_test)
print("Test accuracy: ", PnL_scorer(y_test=y_test, y_pred=y_test_pred))