import sys

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from read_large_files import map_and_load_pkl_files, select_assets
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ========== 1. 生成伪造数据 ==============


# ========== 2. 添加滚动特征 (简化) ==========
def add_features(df, window=10):
    df_ = df.copy()
    df_['mean_close'] = df_.groupby('asset')['close'].rolling(window).mean().values
    df_['std_close'] = df_.groupby('asset')['close'].rolling(window).std().values
    return df_


def factor_ic_analysis(df_factor):
    """
    输入含有 [pred_factor, label] 的 DataFrame
    输出它们的 Pearson 相关系数
    """
    corr_pearson = df_factor['pred_factor'].corr(df_factor['label'], method='pearson')
    print(f"Pearson IC = {corr_pearson:.4f}")

    # 如果想要秩相关系数（Spearman），也可以：
    corr_spearman = df_factor['pred_factor'].corr(df_factor['label'], method='spearman')
    print(f"Spearman IC = {corr_spearman:.4f}")


def collect_factor_and_label(model, dataset, device='cpu'):
    """
    对传入的 dataset 做一次推断，返回 DataFrame:
        index: 样本的顺序（或自行构造 time/asset 索引）
        columns: [pred_factor, label]
    其中 pred_factor 即模型的预测值，可视为“因子”
    label 即真实未来收益率
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            preds = model(x_batch).squeeze(-1)  # (batch,)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    df_factor = pd.DataFrame({
        'pred_factor': all_preds,
        'label': all_labels
    })
    return df_factor


# ========== 3. 构建Dataset ==============
class StocksDataset(Dataset):
    def __init__(self, df, window_size=30):
        self.df = df.copy()
        self.window_size = window_size
        # 排序
        self.df = self.df.sort_index(level=['asset', 'time'])
        self.samples = []
        self._build_samples()

    def _build_samples(self):
        for asset, dfg in self.df.groupby('asset'):
            dfg = dfg.dropna()  # 去除NaN
            values = dfg.values
            if len(values) < self.window_size + 1:
                continue

            for i in range(len(values) - self.window_size - 5):
                x_window = values[i: i + self.window_size]
                y_future = values[i + self.window_size]
                y_next = values[i + self.window_size + 5]
                # 取close列的索引
                close_idx = dfg.columns.get_loc('close')

                close_t = y_future[close_idx]
                close_tp1 = y_next[close_idx]

                # 收益率
                label = (close_tp1 - close_t) / (close_t + 1e-9)

                self.samples.append((x_window, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x_window, label = self.samples[idx]
        x_window = torch.tensor(x_window, dtype=torch.float)  # shape [window_size, num_features]
        label = torch.tensor(label, dtype=torch.float)

        # 转置: [num_features, window_size]
        x_window = x_window.T

        return x_window, label


# ========== 4. 模型结构 ===============
class AlphaNetSimple(nn.Module):
    def __init__(self, num_features=7, window_size=30):
        super().__init__()
        self.conv1 = nn.Conv1d(num_features, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.gap(x)  # [batch, 32, 1]
        x = x.squeeze(-1)  # [batch, 32]
        x = self.fc1(x)  # [batch, 16]
        x = F.relu(x)
        out = self.fc2(x)  # [batch, 1]
        return out


# ========== 5. 训练示例 ==============
def train_model(model, train_loader, lr=1e-3, epochs=5, device='cpu'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(x_batch).squeeze(-1)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x_batch.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch + 1}/{epochs}] Loss: {avg_loss:.6f}")


start = "2020-1-1"
end = "2021-1-30"
#assets = ['BTC-USDT_spot']
assets = select_assets(start_time=start,spot=50)
data = map_and_load_pkl_files(asset_list=assets, start_time=start, end_time=end, level="15min")
# data['future_return'] = data.groupby('asset')['close'].apply(lambda x: x.shift(-10) / x - 1).droplevel(0)


df_feat = add_features(data, window=10)
df_feat = df_feat.dropna()
split_date = '2020-10-01'  # 自定义一个分割日期
df_train = df_feat.loc[df_feat.index.get_level_values('time') < split_date].copy()
df_test = df_feat.loc[df_feat.index.get_level_values('time') >= split_date].copy()

train_dataset = StocksDataset(df_train, window_size=30)
test_dataset = StocksDataset(df_test, window_size=30)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

num_features = df_feat.shape[1]  # columns=[open, high, low, close, volume, mean_close, std_close]
model = AlphaNetSimple(num_features=num_features, window_size=30)

train_model(
    model=model,
    train_loader=train_loader,
    epochs=20
)
torch.save(model.state_dict(), "my_model.pt")
model = AlphaNetSimple(num_features, window_size=30)
model.load_state_dict(torch.load("my_model.pt"))
model.eval()



def evaluate_on_test(model, test_loader, device='cpu'):
    model.eval()
    preds_list = []
    labels_list = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            preds = model(x_batch).squeeze(-1)  # [batch,]
            print(preds)
            preds_list.append(preds.cpu().numpy())
            labels_list.append(y_batch.cpu().numpy())

    preds_array = np.concatenate(preds_list)
    labels_array = np.concatenate(labels_list)

    # 计算 MSE
    mse = np.mean((preds_array - labels_array) ** 2)
    # 计算 MAE
    mae = np.mean(np.abs(preds_array - labels_array))
    # 计算 Pearson 相关系数
    corr_pearson = pd.Series(preds_array).corr(pd.Series(labels_array), method='pearson')
    print(f"Test MSE = {mse:.6f}")
    print(f"Test MAE = {mae:.6f}")
    print(f"Test Pearson Corr = {corr_pearson:.4f}")

    return preds_array, labels_array


# 在训练集结束后:
preds_test, labels_test = evaluate_on_test(model, test_loader, device='cpu')
