import numpy as np
from read_large_files import map_and_load_pkl_files,select_assets
from matplotlib import pyplot as plt
from feature_generation import alpha4
from IC_calculator import compute_ic
def standard_gaussian_kernel(distance, bandwidth):
    """ 标准高斯核函数（对应论文公式 (10)）
    参数:
      distance: x - Xₜ（可以是数组）
      bandwidth: 带宽参数 h

    返回:
      标准高斯核值，即 1/(h√(2π)) * exp( - (distance/h)² /2 )
    """
    z = distance / bandwidth
    return np.exp(-0.5 * z**2) / (bandwidth * np.sqrt(2 * np.pi))


def smooth_price_series(prices, bandwidth):
    """
    参数:
    prices: 价格序列，数组形式
    bandwidth: 带宽参数
    返回:
    平滑后的价格序列
    """
    n = len(prices)
    # 时间索引作为自变量
    x = np.arange(n)
    smoothed = np.empty(n, dtype=float)

    # 对每个时间点进行平滑
    for i in range(n):
        distances = x - i  # 计算所有时间点与当前点 i 的距离
        weights = standard_gaussian_kernel(distances, bandwidth)
        smoothed[i] = np.sum(weights * prices) / np.sum(weights)
    return smoothed

if __name__ == "__main__":
    start = "2020-12-20"
    end = "2022-12-31"
    #assets = select_assets(start_time=start, spot=True, n=50)
    #print(assets)
    assets = ['ETH-USDT_spot']
    data = map_and_load_pkl_files(asset_list=assets, start_time=start, end_time=end, level="15min")
    prices = data['close'].values
    t = np.linspace(0, len(prices), len(prices))

    plt.plot(t, prices, label="orignal", color="gray", alpha=0.6)
    plt.xlabel("time")
    plt.ylabel("price")
    plt.title("price smoothing")
    plt.legend()
    plt.show()



