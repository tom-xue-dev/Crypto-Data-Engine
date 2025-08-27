"""
测试下载服务Repository
"""
from crypto_data_engine.db.session import create_tables, test_connection
from crypto_data_engine.db.repository.download_service import DownloadServiceRepository
from crypto_data_engine.db.models.download import TaskStatus


def test_download_service():
    """测试下载服务功能"""

    print("🧪 测试下载服务Repository...")

    # 1. 创建下载任务
    print("\n📝 1. 创建下载任务")
    task = DownloadServiceRepository.create_download_task(
        exchange="binance",
        symbol="BTCUSDT",
        year=2024,
        month=1,
        file_url="https://data.binance.vision/data/spot/monthly/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-01.zip",
        priority=5
    )
    print(f"   ✅ 创建任务: {task.id} - {task.file_name}")

    # 2. 批量创建任务
    print("\n📦 2. 批量创建任务")
    symbols = ["ETHUSDT", "ADAUSDT", "DOTUSDT"]
    year_months = [(2024, 1), (2024, 2), (2024, 3)]

    created_tasks, skipped = DownloadServiceRepository.create_batch_tasks(
        exchange="binance",
        symbols=symbols,
        year_month_pairs=year_months,
        priority=3
    )
    print(f"   ✅ 批量创建: {len(created_tasks)} 个任务")
    print(f"   ⚠️ 跳过: {len(skipped)} 个任务")

    # 3. 获取可用任务
    print("\n🎯 3. 获取可用任务")
    available_tasks = DownloadServiceRepository.get_available_tasks(
        exchange="binance",
        limit=5,
        worker_id="test-worker-001"
    )
    print(f"   ✅ 可用任务: {len(available_tasks)} 个")

    # 4. 开始执行任务
    print("\n🚀 4. 开始执行任务")
    if available_tasks:
        started_task = DownloadServiceRepository.start_task(
            available_tasks[0].id,
            worker_id="test-worker-001"
        )
        print(f"   ✅ 开始任务: {started_task.id}, 状态: {started_task.status.value}")

        # 5. 更新进度
        print("\n📊 5. 更新任务进度")
        DownloadServiceRepository.update_progress(started_task.id, 50, file_size=1024*1024*5)
        DownloadServiceRepository.update_progress(started_task.id, 100, file_size=1024*1024*10)

        # 6. 完成任务
        print("\n✅ 6. 完成任务")
        completed_task = DownloadServiceRepository.complete_task(
            started_task.id,
            local_path="/data/binance/BTCUSDT/2024-01.parquet",
            file_size=1024*1024*10
        )
        print(f"   ✅ 任务完成: {completed_task.status.value}")

    # 7. 获取任务进度
    print("\n📈 7. 获取任务进度")
    if available_tasks:
        progress = DownloadServiceRepository.get_task_progress(available_tasks[0].id)
        print(f"   📊 任务进度: {progress}")

    # 8. 获取交易所汇总
    print("\n🏢 8. 获取交易所汇总")
    summary = DownloadServiceRepository.get_exchange_summary("binance")
    print(f"   📈 Binance汇总:")
    print(f"      总任务数: {summary['statistics']['total_count']}")
    print(f"      成功率: {summary['statistics']['success_rate']:.1f}%")
    print(f"      最近任务: {len(summary['recent_tasks'])} 个")

    # 9. 服务健康检查
    print("\n🩺 9. 服务健康检查")
    health = DownloadServiceRepository.get_service_health()
    print(f"   🏥 服务状态: {health['status']}")
    print(f"   📊 待处理任务: {health['metrics']['pending_tasks']}")
    print(f"   📊 成功率: {health['metrics']['success_rate_percent']}%")

    # 10. Worker性能分析
    print("\n👷 10. Worker性能分析")
    worker_perf = DownloadServiceRepository.get_worker_performance("test-worker-001")
    print(f"   👨‍💼 Worker: {worker_perf['worker_id']}")
    print(f"   📊 处理任务: {worker_perf.get('total_tasks', 0)} 个")
    print(f"   ✅ 成功率: {worker_perf.get('success_rate', 0):.1f}%")

    print("\n🎉 下载服务测试完成!")


def test_service_management():
    """测试服务管理功能"""
    print("\n🔧 测试服务管理功能...")

    # 重置卡住的任务
    print("\n🔄 1. 重置卡住的任务")
    reset_count = DownloadServiceRepository.reset_stuck_tasks(timeout_hours=2)
    print(f"   ✅ 重置了 {reset_count} 个卡住的任务")

    # 数据清理
    print("\n🗑️ 2. 数据清理")
    cleanup_result = DownloadServiceRepository.cleanup_service_data(
        keep_days=30,
        keep_failed_days=7
    )
    print(f"   🗑️ 清理结果: {cleanup_result}")

    print("\n✅ 服务管理测试完成!")


if __name__ == "__main__":
    if test_connection():
        create_tables()
        test_download_service()
        test_service_management()
    else:
        print("❌ 数据库连接失败")