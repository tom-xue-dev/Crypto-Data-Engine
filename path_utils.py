import os

def get_asset_file_path(asset_name):
    files = os.listdir(fr".\data\{asset_name}")
    asset_files = []
    for file in files:
        if asset_name in file:
            asset_files.append(fr".\data\{asset_name}\{file}")
    return asset_files

def get_folder_size(path):
    """计算文件夹的总大小（单位：字节）"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            # 如果是符号链接，则忽略
            if not os.path.islink(filepath):
                total_size += os.path.getsize(filepath)
    return total_size

def get_sorted_assets(root_dir):
    """从小到大获取文件的名称和大小"""
    folder_sizes = []

    # 遍历根目录下的每个子目录
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path):
            size = get_folder_size(folder_path)
            folder_sizes.append((folder, size))

    # 按大小排序，取前N个
    folder_sizes.sort(key=lambda x: x[1])
    return folder_sizes

