import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import numpy as np
from read_large_files import map_and_load_pkl_files, select_assets
import matplotlib.dates as mdates
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report

import concurrent.futures


def process_group_item(args):
    # args 包含三个元素：资产名称、该资产对应的 DataFrame 以及迭代索引 i
    asset, group, i = args
    print(i)
    # 这里假设 group 的行索引足够连续，即使用 iloc 方式获取
    img = generate_candlestick_image(df=group[i*96: i*96 + 96])
    arr = preprocess_image(img)
    label = group.iloc[i *96+ 96]['label']
    return arr, label


#

def generate_candlestick_image(
        df,
        ma_window=20,
        fig_width=6,
        fig_height=4,
        dpi=1000,
        background_color='black'
):
    """
    将OHLC数据绘制成黑白图像，并返回numpy数组 (H, W, 1)。
    :param df: 包含[Date, Open, High, Low, Close, Volume]的DataFrame
    :param ma_window: 移动平均窗口大小
    :param fig_width, fig_height: 画布大小
    :param dpi: 分辨率
    :param background_color: 背景色，默认为黑色
    :return: numpy数组, dtype=uint8, 范围[0, 255]
    """
    # 1. 计算移动平均
    # df['MA'] = df['close'].rolling(ma_window, min_periods=1).mean()

    # 2. 创建画布
    fig, (ax_price, ax_vol) = plt.subplots(
        2, 1,
        figsize=(fig_width, fig_height),
        dpi=dpi,
        gridspec_kw={'height_ratios': [4, 1]},
        sharex=True
    )
    fig.patch.set_facecolor(background_color)  # 整体背景

    # 坐标轴背景
    ax_price.set_facecolor(background_color)
    ax_vol.set_facecolor(background_color)

    # 3. 绘制烛棒
    #   简单方法：逐日绘制垂线和开收横线
    for idx, row in df.iterrows():
        color = 'white'
        x = mdates.date2num(row.name[0])  # 将 Timestamp 转换为数值
        x_spacing = 1 / 96
        # 最高最低线
        ax_price.plot([x, x], [row['low'], row['high']], color=color, linewidth=1, zorder=1)
        # 开盘收盘线
        # 左横线表示Open，右横线表示Close
        ax_price.hlines(row['open'], x - x_spacing / 2, x, color=color, linewidth=1, zorder=2)  # 开盘价横线
        ax_price.hlines(row['close'], x, x + x_spacing / 2, color=color, linewidth=1, zorder=2)  # 收盘价横线

    # print(df.index.get_level_values('time'))
    # 4. 绘制移动平均线 (MA)
    # ax_price.plot(df.index.get_level_values('time'), df['MA'], color='red', linewidth=1)

    # 5. 绘制成交量
    ax_vol.bar(df.index.get_level_values('time'), df['volume'], color='white', width=0.01)

    # 6. 调整外观
    ax_price.set_xlim(mdates.date2num(df.index.get_level_values('time'))[0],
                      mdates.date2num(df.index.get_level_values('time'))[-1])  # 边界留白
    ax_price.set_ylim(df['low'].min() * 0.99, df['high'].max() * 1.01)
    ax_vol.set_ylim(0, df['volume'].max() * 1.1)

    # 隐藏坐标轴刻度和标签
    ax_price.set_xticks([])
    ax_price.set_yticks([])
    ax_vol.set_xticks([])
    ax_vol.set_yticks([])

    # 减少边距
    plt.subplots_adjust(wspace=0, hspace=0)

    # 7. 保存到内存
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, facecolor=background_color)
    plt.close(fig)

    # 8. Pillow打开，转为灰度(“L”模式 -> 0~255)
    buf.seek(0)
    pil_img = Image.open(buf).convert('L')  # 转为灰度
    img_array = np.array(pil_img)
    img_resized = Image.fromarray(img_array).resize((256, 256))
    return img_resized


def preprocess_image(pil_img):
    """
    将 PIL.Image -> (1, 256, 256, 3) 的 Numpy 数组，并进行简单归一化
    """
    # 转换为 numpy 数组 (H, W, C)
    img_array = np.array(pil_img)  # shape: (256, 256, 3)

    # 如果需要的话，可以进行像素值归一化
    img_array = img_array.astype('float32') / 255.0

    # 在第 0 维增加一个 batch 维度，得到 (1, 256, 256, 3)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


class CNNModel(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNModel, self).__init__()
        self.features = nn.Sequential(
            # 第一层：输入 1 个通道，输出 32 个通道
            nn.Conv2d(1, 32, kernel_size=(5, 3), padding=(2, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # 输出 (32, 256, 256) 或者尺寸根据池化而定

            # 第二层：输入 32 个通道（来自第一层），输出 64 个通道
            nn.Conv2d(32, 64, kernel_size=(5, 3), padding=(2, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # 输出 (64, 128, 128)

            # 第三层：输入 64 个通道（来自第二层），输出 128 个通道
            nn.Conv2d(64, 128, kernel_size=(5, 3), padding=(2, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 1))  # 输出 (128, 64, 64) 或根据实际池化结果

        )
        # 如果最后一层卷积输出的特征图形状为 (128, 32, 256)
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 展平成 (batch_size, 128*32*256 = 1048576)
            nn.Linear(128 * 32 * 256, 256),  # 修改这里，使用 1048576 而非 131072
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    start = "2020-1-1"
    end = "2021-1-30"
    assets = ['BTC-USDT_spot', 'ETH-USDT_spot']
    data = map_and_load_pkl_files(asset_list=assets, start_time=start, end_time=end, level="15min")
    data['future_return'] = data.groupby('asset')['close'].apply(lambda x: x.shift(-10) / x - 1).droplevel(0)
    data['label'] = np.where(data['future_return'] < 0, 0, 1)
    data = data.dropna()
    img_arrs = []
    labels = []
    tasks = []
    length = 96 * 2  # 原代码中定义的 length，不过这里循环中只用到 96
    for asset, group in data.groupby('asset'):
        num_iters = len(group)//96
        for i in range(num_iters):
            tasks.append((asset, group, i))

    # 使用 ProcessPoolExecutor 多进程处理任务
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(process_group_item, tasks)
        for arr, label in results:
            img_arrs.append(arr)
            labels.append(label)

    # 假设 img_arrs 是你生成的图像列表，每个元素形状为 (256, 256, 3)
    # 假设 labels 是对应的二分类标签（0 表示下跌，1 表示上涨）
    n = len(img_arrs)  # 样本数
    H, W, C = 256, 256, 3

    # 用真实数据替换随机生成的数据
    x_train_np = np.stack(img_arrs).astype(np.float32)  # (n, 256, 256, 3)
    y_train_np = np.array(labels).astype(np.int64)  # (n,)

    # 如果需要归一化，可根据具体情况处理（例如，除以255）
    x_train_np /= 255.0

    # 转换为 PyTorch 张量，并将通道维度置于前面：(n, 3, 256, 256)
    x_train = torch.from_numpy(x_train_np)  # x_train_np 的形状应为 (n, 1, 256, 256)

    y_train = torch.from_numpy(y_train_np)

    # 创建数据集和 DataLoader
    dataset = TensorDataset(x_train, y_train)
    batch_size = 32
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 模型设置：由于是二分类任务，num_classes 设为2
    input_shape = (256, 256, 3)
    num_classes = 2  # 二分类
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNModel(num_classes=num_classes).to(device)

    # 使用交叉熵损失，适用于多分类（二分类时标签为 0 或 1）
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    # 训练过程
    num_epochs = 500
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        epoch_total_filtered = 0  # 记录整个 epoch 中软max概率 > 0.6 的样本数
        epoch_correct_filtered = 0  # 记录整个 epoch 中被正确预测且概率 > 0.6 的样本数

        for inputs, labels in train_loader:
            inputs = inputs.to(device)  # (batch_size, 3, 256, 256)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)  # 输出形状 (batch_size, num_classes)
            probabilities = F.softmax(outputs, dim=1)  # 计算 softmax 概率

            # 获取每个样本最大概率以及对应的预测类别
            max_probs, preds = torch.max(probabilities, dim=1)
            #print(probabilities)
            # 创建 mask，仅选择 softmax 最大概率大于 0.6 的样本
            mask = max_probs > 0.55

            # 计算当前批次的 loss（这里仍然使用所有样本计算 loss 以保证训练稳定性）
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

            # 针对 mask 内的样本计算准确率
            if mask.sum().item() > 0:
                correct_filtered = (preds[mask] == labels[mask]).sum().item()
                epoch_correct_filtered += correct_filtered
                epoch_total_filtered += mask.sum().item()

                # 检查第一层卷积的梯度
                print("First conv layer gradient:", model.features[0].weight.grad.abs().mean().item())

                # 检查最后一层全连接层的梯度
                print("Last fc layer gradient:", model.classifier[-1].weight.grad.abs().mean().item())
        # 计算整个 epoch 的平均 loss
        epoch_loss = running_loss / len(train_loader.dataset)

        # 如果存在符合条件的样本，则计算这些样本的准确率
        if epoch_total_filtered > 0:
            epoch_filtered_accuracy = epoch_correct_filtered / epoch_total_filtered
        else:
            epoch_filtered_accuracy = 0.0

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, "
              f"Filtered Accuracy (softmax > 0.6): {epoch_filtered_accuracy:.4f} "
              f"(Filtered Samples: {epoch_total_filtered}, Correct: {epoch_correct_filtered})")

    model.eval()  # 切换到评估模式
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)  # 输出形状 (batch_size, num_classes)
            preds = torch.argmax(outputs, dim=1)  # 选择最大概率对应的类别
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    # 可选：输出分类报告
    report = classification_report(all_labels, all_preds, target_names=['Class 0', 'Class 1'])
    print("\nClassification Report:")
    print(report)
