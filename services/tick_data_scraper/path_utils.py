import os

def get_asset_file_path(asset_name, data_dir):
    """获取资产的所有 parquet 文件路径"""
    folder = os.path.join(data_dir, asset_name)
    if not os.path.exists(folder):
        return []
    files = os.listdir(folder)
    asset_files = [
        os.path.join(folder, file)
        for file in files
        if file.endswith(".parquet")
    ]
    return asset_files


def get_sorted_assets(root_dir, suffix_filter=None):
    """从小到大获取资产名称和文件总大小"""
    folder_sizes = []

    # 遍历根目录下的每个子目录
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        if suffix_filter and not folder.endswith(suffix_filter):
            continue
        size = get_folder_size(folder_path)
        folder_sizes.append((folder, size))

    folder_sizes.sort(key=lambda x: x[1])
    return folder_sizes


def get_folder_size(folder):
    """计算文件夹大小"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size
