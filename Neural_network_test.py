import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
def PnL_scorer_nn(y_test, y_pred):
    conf_mtrx = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    total_trades = conf_mtrx[:, [0, 2]].sum()
    if total_trades == 0:
        return 0.0

    TA = (conf_mtrx[0, 0] + conf_mtrx[2, 2]) / total_trades
    coverage = total_trades / y_test.size
    return coverage * (2 * TA - 1)


indicators_data = pd.read_csv("NVIDIA_indicators_dataset.csv", index_col='Date')
indicators_df = indicators_data.iloc[:-10]
price_data = pd.read_csv("NVIDIA_price_data_main.csv", index_col="Date")

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

X_train_std = sc.fit_transform(X_train)
X_test_std  = sc.transform(X_test)

y_train_1 = y_train + 1
y_test_1 = y_test + 1
n_train = 100
x_train_tensor = torch.tensor(X_train_std, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_1.to_numpy(), dtype=torch.long)
x_valid = torch.tensor(X_test_std, dtype=torch.float32)
y_valid = torch.tensor(y_test_1.to_numpy(), dtype=torch.long)

train_ds = TensorDataset(x_train_tensor, y_train_tensor)
batch_size = 64
train_dl = DataLoader(train_ds, batch_size, shuffle=False)

model = nn.Sequential(
    nn.Linear(13, 16),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(16, 3)
)

loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([2.0, 1.0, 2.0]))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 200


def train(model, num_epochs, train_dl, x_valid, y_valid):
    loss_hist_train = []
    loss_hist_valid = []
    pnl_hist_train = []
    pnl_hist_valid = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_pnl = 0

        for x_batch, y_batch in train_dl:
            optimizer.zero_grad()

            logits = model(x_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            y_pred = torch.argmax(logits, dim=1)
            train_pnl += PnL_scorer_nn(
                y_test=y_batch.detach().cpu().numpy(),
                y_pred=y_pred.detach().cpu().numpy()
            )

        model.eval()
        with torch.no_grad():
            logits_val = model(x_valid)
            val_loss = loss_fn(logits_val, y_valid)

            y_pred_val = torch.argmax(logits_val, dim=1)
            val_pnl = PnL_scorer_nn(
                y_test=y_valid.detach().cpu().numpy(),
                y_pred=y_pred_val.detach().cpu().numpy()
            )

        loss_hist_train.append(train_loss / len(train_dl))
        loss_hist_valid.append(val_loss.item())
        pnl_hist_train.append(train_pnl / len(train_dl))
        pnl_hist_valid.append(val_pnl)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | "
                  f"train_loss={loss_hist_train[-1]:.4f} | "
                  f"train_score={train_pnl / len(train_dl):.4f} | "
                  f"val_loss={loss_hist_valid[-1]:.4f} | "
                  f"val_score={val_pnl:.4f}")

    return loss_hist_train, loss_hist_valid, pnl_hist_train, pnl_hist_valid


training_data = train(model, num_epochs, train_dl, x_valid, y_valid)

print("\n\n\n\n\n")
x_arr = np.arange(len(training_data[0])) + 1

fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(1, 2, 1)
ax.plot(x_arr, training_data[0], label='Train loss')
ax.plot(x_arr, training_data[1], label='Validation loss')
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Loss', size=15)
ax.legend(fontsize=15)
ax = fig.add_subplot(1, 2, 2)
ax.plot(x_arr, training_data[2], label='Train acc.')
ax.plot(x_arr, training_data[3], label='Validation acc.')
ax.legend(fontsize=15)
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Accuracy', size=15)
plt.show()
