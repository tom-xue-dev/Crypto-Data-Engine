import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from read_large_files import load_filtered_data_as_list, map_and_load_pkl_files, select_assets



class TimeSeriesCNN(nn.Module):
    def __init__(self, input_channels=1):
        super(TimeSeriesCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=3, stride=1, padding=1)  # 16个特征
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)  # 32个特征
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 卷积+池化
        x = self.pool(F.relu(self.conv2(x)))
        return x  # 提取的时间序列特征

if __name__ == '__main__':
    # 生成示例数据：100个时间步长
    start = "2020-1-1"
    end = "2022-12-31"
    assets = select_assets(start_time=start, spot=True, n=1)
    data = map_and_load_pkl_files(asset_list=assets, start_time=start, end_time=end, level="15min")
    data['future_return'] = data.groupby('asset')['close'].apply(lambda x: x.shift(-10) / x - 1).droplevel(0)
    future_return = data['future_return'].values
    close_prices = data['close'].values
    close_prices = close_prices.reshape(1, 1, -1)  # 转换为 CNN 输入格式 (batch_size, channels, time_steps)
    close_prices = torch.tensor(close_prices, dtype=torch.float32)
    model = TimeSeriesCNN()
    # 获取 CNN 特征
    features = model(close_prices).detach().numpy()  # 提取特征，形状为 (1, C, T_new)

    # 这里 features.shape[2] 是 CNN 计算后时间维度 T_new
    features = features.reshape(features.shape[1], features.shape[2]).T  # (T_new, C)


    # ic_values = []
    # for i in range(features.shape[1]):  # 遍历每个 CNN 提取的特征通道
    #     valid_idx = min(len(features), len(future_return))  # 确保两个数组长度一致
    #     ic = np.corrcoef(features[:valid_idx, i], future_return[:valid_idx])[0, 1]  # 计算 IC
    #     ic_values.append(ic)
    print(len(features),len(features[0]),len(future_return))