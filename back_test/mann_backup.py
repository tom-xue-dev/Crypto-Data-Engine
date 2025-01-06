import pandas as pd
import re

def calculate_trade_statistics(file_path):
    """
    读取交易记录的 txt 文件，计算胜率和盈亏比。

    参数：
    file_path: str, txt 文件路径

    返回：
    dict: 包含胜率和盈亏比的统计结果
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 存储关仓记录
    close_trades = []

    # 逐行解析交易记录
    for line in lines:
        try:
            # 将 Timestamp 对象替换为普通字符串
            line = re.sub(r"Timestamp\\('(.*?)'\\)", r"'\\1'", line.strip())
            # 提供 eval 的上下文，支持 Timestamp 解析
            trade = eval(line, {"Timestamp": pd.Timestamp})
            if trade['action'] == 'close':
                close_trades.append(trade)
        except Exception as e:
            print(f"Error parsing line: {line}\n{e}")

    # 如果没有关仓记录，直接返回
    if not close_trades:
        return {
            'win_rate': None,
            'profit_loss_ratio': None
        }

    # 转换为 DataFrame
    df = pd.DataFrame(close_trades)

    # 计算胜率和盈亏比
    df['is_profit'] = df['pnl'] > 0
    total_trades = len(df)
    winning_trades = df['is_profit'].sum()
    losing_trades = total_trades - winning_trades

    win_rate = winning_trades / total_trades

    # 盈利与亏损的绝对值
    total_profit = df[df['pnl'] > 0]['pnl'].sum()
    total_loss = abs(df[df['pnl'] <= 0]['pnl'].sum())

    profit_loss_ratio = total_profit / total_loss if total_loss != 0 else None

    return {
        'win_rate': win_rate,
        'profit_loss_ratio': profit_loss_ratio
    }

# 示例用法
file_path = "transactions.txt"
result = calculate_trade_statistics(file_path)
print(result)
