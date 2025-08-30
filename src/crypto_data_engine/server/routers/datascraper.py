"""
数据抓取API路由 - 基于API文档规范
"""
from datetime import datetime, date
from fastapi import APIRouter, Body, HTTPException, Query, Path, BackgroundTasks
import ccxt
from crypto_data_engine.common.logger.logger import get_logger
from crypto_data_engine.db.repository.download import DownloadTaskRepository
from crypto_data_engine.db.models.download import TaskStatus
from crypto_data_engine.server.constants.request_schema import BatchDownloadRequest
from crypto_data_engine.server.constants.response import JobResponse, TaskResponse, BaseResponse
from crypto_data_engine.server.constants.response_code import ResponseCode
from task_manager.celery_app import celery_app

datascraper_router = APIRouter(prefix="/api/v1/download", tags=["数据抓取"])

logger = get_logger(__name__)
# ==================== 源管理 ====================

@datascraper_router.get("/exchanges", summary="获取支持的数据源")
async def get_sources():
    """获取所有支持的数据源"""
    from crypto_data_engine.common.config.config_settings import settings
    source = settings.downloader_cfg.list_all_exchanges()
    response = BaseResponse(data=source)
    return response

@datascraper_router.get("/{source}/symbols", summary="获取交易所支持的交易对")
async def get_source_symbols(
        source: str = Path(..., description="交易所名称"),
        limit: int = Query(100, ge=1, le=1000, description="返回数量限制")
):
    try:
        # 使用ccxt获取交易所信息
        exchange_class = getattr(ccxt, source.lower())
        exchange = exchange_class()
        markets = exchange.load_markets()
        symbols = list(markets.keys())[:limit]
        return {
            "exchange": source,
            "symbols": symbols,
            "total": len(symbols)
        }
    except AttributeError:
        raise HTTPException(status_code=404, detail=f"Unsupported exchange: {source}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"fail to get the symbol: {str(e)}")

# ==================== 下载任务管理 ====================


@datascraper_router.post("/downloads/jobs", response_model=JobResponse, summary="创建下载作业")
async def create_download_job(request: BatchDownloadRequest):
    """创建批量下载作业并推送给Celery"""
    try:
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        year_month_pairs = [(request.year, month) for month in request.months]
        created_tasks, skipped = DownloadTaskRepository.create_batch_tasks(
            exchange=request.exchange,
            symbols=request.symbols,
            year_month_pairs=year_month_pairs,
        )
        task_responses = []
        for task in created_tasks:
            task_response = TaskResponse(
                id=task.id,
                exchange=task.exchange,
                symbol=task.symbol,
                year=task.year,
                month=task.month,
                status=task.status.value,
                file_name=task.file_name,
                file_size=task.file_size,
                local_path=task.local_path,
                task_start=task.task_start,
                task_end=task.task_end
            )
            task_responses.append(task_response)
    except Exception as e:
        logger.error(f"create download tasks failed: {str(e)}")
        return BaseResponse(code=ResponseCode.DB_ERROR,message="DB_ERROR",data=e)

    try:
        # 推送到 Celery
        celery_task_ids = []
        for i, task in enumerate(created_tasks):
            celery_config = {
                'exchange_name': task.exchange,
                'symbols': [task.symbol],
                'start_date': f"{task.year}-{task.month:02d}",
                'end_date': f"{task.year}-{task.month:02d}",
                'max_threads': 4,
                'task_id': task.id
            }
            celery_result = celery_app.send_task(
                'tick.download',
                args=[celery_config],
                queue='io_intensive'
            )
            celery_task_ids.append({
                'db_task_id': task.id,
                'celery_task_id': celery_result.id,
                'status': celery_result.state
            })
            # 更新任务状态
            DownloadTaskRepository.update_status(task.id, TaskStatus.PENDING)
        return JobResponse(
            job_id=job_id,
            created_tasks=task_responses,  # 使用预构建的响应对象
            skipped_tasks=skipped,
            total_created=len(created_tasks),
            total_skipped=len(skipped)
        )

    except Exception as e:
        logger.error(f"推送任务失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"创建下载作业失败: {str(e)}")


@datascraper_router.get("/celery/status/{celery_task_id}", summary="查询Celery任务状态")
async def get_celery_task_status(celery_task_id: str):
    """查询Celery任务的执行状态"""
    try:
        from celery.result import AsyncResult

        result = AsyncResult(celery_task_id, app=celery_app)

        return {
            'celery_task_id': celery_task_id,
            'status': result.state,
            'result': result.result if result.ready() else None,
            'info': result.info,
            'successful': result.successful(),
            'failed': result.failed()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")

