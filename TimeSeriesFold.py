from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.compose import ColumnTransformer
from pathlib import Path

def PnL_scorer(y_test, y_pred):
    conf_mtrx = confusion_matrix(y_test, y_pred, labels=[1, 0, -1])
    total_trades = conf_mtrx[:,[0,2]].sum()
    if total_trades == 0:
        return 0.0

    TA = (conf_mtrx[0,0] + conf_mtrx[2,2])/total_trades
    coverage = total_trades/y_test.size
    return coverage*(2*TA - 1)

file_name = []
file_score_train = []
file_score_test = []


data_dir = Path("data_files")

for file in data_dir.iterdir():
    file_name.append(file)
    df = pd.read_csv(file, index_col="date")
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    spl = TimeSeriesSplit(n_splits=10, gap=10).split(X_train, y_train)

    cols_to_std_scaling = df.columns.difference(["RSI", "MFI", "Label"]).tolist()
    preprocessor = ColumnTransformer([
        ("StandardScaler", StandardScaler(), cols_to_std_scaling),
        ("MinMaxScaler", MinMaxScaler(feature_range=(-1, 1)), ["RSI", "MFI"])]
    )
    pipe_svc = make_pipeline(
        preprocessor,
        SVC(kernel='linear', C= 1.0)
    )


    scores = []
    for k, (train, test) in enumerate(spl):
        pipe_svc.fit(X_train.iloc[train], y_train.iloc[train])
        y_pred = pipe_svc.predict(X_train.iloc[test])
        score = PnL_scorer(y_test=y_train.iloc[test], y_pred=y_pred)
        scores.append(score)

    mean_acc = np.mean(scores)
    file_score_train.append(mean_acc)

    y_pred_test = pipe_svc.predict(X_test)
    file_score_test.append(PnL_scorer(y_test=y_test, y_pred=y_pred_test))

results_df = pd.DataFrame({
    'file_name': file_name,
    'score_train': file_score_train,
    'score_test': file_score_test
})

print(results_df.sort_values('score_train', ascending=False).head(30))
print(results_df.sort_values('score_train', ascending=False).tail(30))

