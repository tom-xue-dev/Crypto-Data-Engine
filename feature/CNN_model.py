import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from read_large_files import map_and_load_pkl_files, select_assets
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, r2_score

class CryptoDataset(Dataset):
    def __init__(self, asset_data, window_size=60):
        self.data = asset_data[['open', 'high', 'low', 'close', 'volume']].values
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        window = self.data[idx:idx + self.window_size]
        target = self.data[idx + self.window_size][3]  # 预测close价格
        return torch.FloatTensor(window), torch.FloatTensor([target])


# 使用MinMaxScaler对特征进行缩放
scaler = MinMaxScaler(feature_range=(0, 1))
start = "2020-1-1"
end = "2022-12-31"
assets = select_assets(start_time=start, spot=True, m=20)
print(assets)

data = map_and_load_pkl_files(asset_list=assets, start_time=start, end_time=end, level="15min")
# 以BTC数据为例
btc_data = data.xs('BTC-USDT_spot', level='asset').sort_index()
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(btc_data)
dataset = CryptoDataset(pd.DataFrame(scaled_data, columns=btc_data.columns))
train_size = int(len(dataset) * 0.8)
train_set, test_set = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

import torch.nn as nn


class CryptoLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)  # out: (batch, seq_len, hidden_size)
        out = out[:, -1, :]  # 取最后一个时间步输出
        return self.fc(out)


model = CryptoLSTM(input_size=5, hidden_size=64, num_layers=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
best_loss = float('inf')
patience = 7
trigger_times = 0

for epoch in range(100):
    model.train()
    train_loss = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()
        train_loss += loss.item()

    # 验证阶段
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            val_loss += criterion(outputs, y).item()

    # 动态调整学习率
    scheduler.step(val_loss)

    # 早停机制
    if val_loss < best_loss:
        best_loss = val_loss
        trigger_times = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f'Early stopping at epoch {epoch}')
            break


def inverse_transform(predictions, scaler):
    # 确保输入为一维数组
    if predictions.ndim == 2:
        predictions = predictions.squeeze()

    # 创建与原始特征数匹配的占位数组
    dummy = np.zeros((len(predictions), scaler.n_features_in_))
    dummy[:, 3] = predictions  # 填充close预测值
    return scaler.inverse_transform(dummy)[:, 3]


model.load_state_dict(torch.load('best_model.pth'))
model.eval()

predictions, actuals = [], []
with torch.no_grad():
    for X, y in test_loader:
        X = X.to(device)
        pred = model(X).cpu().numpy()[0]
        predictions.append(pred)
        actuals.append(y.numpy()[0])

# 反标准化
pred_prices = inverse_transform(np.array(predictions), scaler)
true_prices = inverse_transform(np.array(actuals), scaler)

plt.figure(figsize=(15,6))
plt.plot(true_prices, label='Actual Price', alpha=0.7)
plt.plot(pred_prices, label='Predicted Price', linestyle='--')
plt.title('BTC-USDT Price Prediction')
plt.legend()
plt.show()


print(f'MAE: {mean_absolute_error(true_prices, pred_prices):.4f}')
print(f'R² Score: {r2_score(true_prices, pred_prices):.4f}')