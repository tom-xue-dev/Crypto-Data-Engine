from datetime import datetime
from typing import Optional, List,Any

from pydantic import BaseModel

from crypto_data_engine.server.constants.response_code import ResponseCode


class BaseResponse(BaseModel):
    code:int = ResponseCode.SUCCESS
    message:str = "success"
    data:Any = None


class TaskResponse(BaseModel):
    """任务响应模型"""
    id: int
    exchange: str
    symbol: str
    year: int
    month: int
    status: str
    file_name: Optional[str]
    file_size: Optional[int]
    local_path: Optional[str]
    task_start: Optional[datetime]
    task_end: Optional[datetime]


class JobResponse(BaseModel):
    """作业响应模型"""
    job_id: str
    created_tasks: List[TaskResponse]
    skipped_tasks: List[dict]
    total_created: int
    total_skipped: int


class MetricsResponse(BaseModel):
    """指标响应模型"""
    total_tasks: int
    pending_tasks: int
    downloading_tasks: int
    completed_tasks: int
    failed_tasks: int
    skipped_tasks: int
