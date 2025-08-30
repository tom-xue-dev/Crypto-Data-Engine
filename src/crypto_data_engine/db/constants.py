from enum import Enum as PyEnum
class TaskStatus(PyEnum):
    """任务状态枚举"""
    PENDING = "pending"  # 未开始
    DOWNLOADING = "downloading"  # 正在下载
    DOWNLOADED = "downloaded"  # 下载完成
    EXTRACTING = "extracting"  # 正在解压
    EXTRACTED = "extracted"  # 解压完成

    PROCESSING = "processing"  # 正在处理/转换
    COMPLETED = "completed"  # 全部完成
    FAILED = "failed"  # 失败
    SKIPPED = "skipped"  # 跳过
    RETRY = "retry"  # 等待重试