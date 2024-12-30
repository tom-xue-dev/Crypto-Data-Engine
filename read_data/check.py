import os
from pathlib import Path

def check_leaf_directories_for_csv(folder_path):
    """
    递归扫描 'folder_path' 及其子目录，仅报告“最下一级(叶子)文件夹”是否包含 CSV 文件。

    参数:
        folder_path (str): 起始文件夹路径。
    """

    root = Path(folder_path)
    if not root.exists() or not root.is_dir():
        print(f"路径不存在或不是文件夹: {folder_path}")
        return

    def is_leaf_directory(dir_path: Path) -> bool:
        """
        判断 dir_path 是否是“叶子”目录（目录下无子文件夹）。
        """
        for item in dir_path.iterdir():
            if item.is_dir():
                return False
        return True

    def collect_leaf_directories(dir_path: Path):
        """
        递归搜集所有“叶子”目录并返回列表。
        """
        # 找出 dir_path 下所有子文件夹
        subdirs = [d for d in dir_path.iterdir() if d.is_dir()]

        # 如果没有子文件夹，则自己就是“叶子”目录
        if not subdirs:
            return [dir_path]

        # 否则继续深入子文件夹并合并结果
        leaf_list = []
        for subdir in subdirs:
            leaf_list.extend(collect_leaf_directories(subdir))
        return leaf_list

    # 1. 收集所有叶子目录
    leaf_dirs = collect_leaf_directories(root)

    # 2. 对每个叶子目录，检查其中是否存在 CSV 文件，并报告
    for leaf in leaf_dirs:
        csv_files = list(leaf.glob("*.csv"))
        if csv_files:
            continue
        else:
            print(f"叶子文件夹: {leaf}")
            print("  不包含任何 CSV 文件。")
    print("扫描完成。")


data_path = "binance"
check_leaf_directories_for_csv(data_path)