# """
# Bar 生成服务 - 将 Tick 数据转换为各种类型的 Bar 数据
# """
# import os
# import pandas as pd
# from pathlib import Path
# from datetime import datetime, timezone
# from typing import List, Dict, Any, Optional
# from tqdm import tqdm
# import logging
#
# from crypto_data_engine.services.tick_data_scraper.app.bar_constructor import BarConstructor
#
# logger = logging.getLogger(__name__)
# class BarProcessorContext:
#     """Bar 处理上下文"""
#     def __init__(self, config: Dict[str, Any]):
#         self.raw_data_dir = config['raw_data_dir']  # tick 数据输入目录
#         self.output_dir = config['output_dir']  # bar 数据输出目录
#         self.bar_type = config.get('bar_type', 'volume_bar')
#         self.threshold = config.get('threshold', 10000000)
#         self.process_num_limit = config.get('process_num_limit', 4)
#         self.suffix_filter = config.get('suffix_filter', None)
#         self.adaptive = config.get('adaptive', False)
#         self.sample_days = config.get('sample_days', 1)
#         self.target_bars = config.get('target_bars', 300)
#         self.batch_size = config.get('batch_size', 10)
#
#
# class BarProcessor:
#     """Bar 数据处理器"""
#
#     def __init__(self, context: BarProcessorContext):
#         self.context = context
#         self.processed_files = set()
#
#     def run_bar_generation_pipeline(self, config: Dict[str, Any]):
#         """运行 Bar 生成流水线"""
#         try:
#             logger.info(f"🚀 开始 Bar 生成流水线")
#             logger.info(f"📊 Bar 类型: {self.context.bar_type}")
#             logger.info(f"📂 输入目录: {self.context.raw_data_dir}")
#             logger.info(f"📂 输出目录: {self.context.output_dir}")
#
#             # 获取待处理的文件
#             symbols = self._get_symbols_to_process(config)
#
#             if not symbols:
#                 logger.warning("⚠️  没有找到待处理的数据文件")
#                 return {"status": "completed", "processed": 0, "message": "No files to process"}
#
#             # 创建输出目录
#             os.makedirs(self.context.output_dir, exist_ok=True)
#
#             # 处理每个交易对
#             total_processed = 0
#             results = []
#
#             for symbol in tqdm(symbols, desc="Processing symbols"):
#                 try:
#                     result = self._process_symbol(symbol)
#                     if result:
#                         results.append(result)
#                         total_processed += 1
#                         logger.info(f"✅ {symbol} 处理完成: {result['bars_count']} bars")
#                 except Exception as e:
#                     logger.error(f"❌ {symbol} 处理失败: {str(e)}")
#
#             logger.info(f"🎉 Bar 生成完成！共处理 {total_processed} 个交易对")
#
#             return {
#                 "status": "completed",
#                 "processed": total_processed,
#                 "total_symbols": len(symbols),
#                 "results": results,
#                 "bar_type": self.context.bar_type,
#                 "threshold": self.context.threshold
#             }
#
#         except Exception as e:
#             logger.error(f"❌ Bar 生成流水线失败: {str(e)}")
#             raise
#
#     def _get_symbols_to_process(self, config: Dict[str, Any]) -> List[str]:
#         """获取需要处理的交易对列表"""
#         raw_data_path = Path(self.context.raw_data_dir)
#
#         if not raw_data_path.exists():
#             logger.error(f"❌ 原始数据目录不存在: {raw_data_path}")
#             return []
#
#         # 获取指定的交易对或自动发现
#         if 'symbols' in config and config['symbols']:
#             symbols = config['symbols']
#         else:
#             # 自动发现目录下的交易对
#             symbols = [d.name for d in raw_data_path.iterdir()
#                        if d.is_dir() and not d.name.startswith('.')]
#
#         # 应用后缀过滤器
#         if self.context.suffix_filter:
#             symbols = [s for s in symbols if s.endswith(self.context.suffix_filter)]
#
#         logger.info(f"📋 找到 {len(symbols)} 个待处理交易对: {symbols[:10]}...")
#         return symbols
#
#     def _process_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
#         """处理单个交易对的数据"""
#         symbol_path = Path(self.context.raw_data_dir) / symbol
#
#         if not symbol_path.exists() or not symbol_path.is_dir():
#             logger.warning(f"⚠️  交易对目录不存在: {symbol_path}")
#             return None
#
#         # 获取该交易对的所有 parquet 文件
#         parquet_files = list(symbol_path.glob("*.parquet"))
#
#         if not parquet_files:
#             logger.warning(f"⚠️  {symbol} 目录下没有找到 parquet 文件")
#             return None
#
#         # 排序文件以确保时间顺序
#         parquet_files.sort()
#
#         logger.info(f"📊 {symbol}: 找到 {len(parquet_files)} 个数据文件")
#
#         # 使用 BarConstructor 处理数据
#         constructor = BarConstructor(
#             folder_path=parquet_files,
#             threshold=self.context.threshold,
#             bar_type=self.context.bar_type
#         )
#
#         # 生成 Bar 数据
#         bars_df = constructor.process_asset_data()
#
#         if bars_df is None or bars_df.empty:
#             logger.warning(f"⚠️  {symbol} 生成的 Bar 数据为空")
#             return None
#
#         # 保存结果
#         output_file = self._save_bars(symbol, bars_df)
#
#         return {
#             "symbol": symbol,
#             "bars_count": len(bars_df),
#             "output_file": str(output_file),
#             "bar_type": self.context.bar_type,
#             "threshold": self.context.threshold,
#             "processed_at": datetime.now().isoformat()
#         }
#
#     def _save_bars(self, symbol: str, bars_df: pd.DataFrame) -> Path:
#         """保存 Bar 数据"""
#         # 创建交易对专用输出目录
#         symbol_output_dir = Path(self.context.output_dir) / symbol
#         symbol_output_dir.mkdir(parents=True, exist_ok=True)
#
#         # 生成输出文件名
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         output_file = symbol_output_dir / f"{symbol}_{self.context.bar_type}_{self.context.threshold}_{timestamp}.parquet"
#
#         # 保存为 parquet 格式
#         bars_df.to_parquet(output_file, compression="brotli", index=False)
#
#         logger.info(f"💾 {symbol} Bar 数据已保存: {output_file}")
#         return output_file
#
#     def get_processing_stats(self) -> Dict[str, Any]:
#         """获取处理统计信息"""
#         return {
#             "processed_files": len(self.processed_files),
#             "context": {
#                 "bar_type": self.context.bar_type,
#                 "threshold": self.context.threshold,
#                 "raw_data_dir": self.context.raw_data_dir,
#                 "output_dir": self.context.output_dir
#             }
#         }
#
#
# def run_simple_bar_generation(
#         exchange_name: str = "binance",
#         symbols: Optional[List[str]] = None,
#         bar_type: str = "volume_bar",
#         threshold: int = 10000000,
#         raw_data_dir: str = "./data/tick_data",
#         output_dir: str = "./data/bar_data"
# ) -> Dict[str, Any]:
#     """
#     简化的 Bar 生成函数 - 类似于 run_simple_download
#     """
#
#     logger.info(f"\n🚀 开始 {exchange_name.upper()} Bar 生成")
#     logger.info(f"📊 Bar 类型: {bar_type}")
#     logger.info(f"🎯 阈值: {threshold}")
#
#     # 构建配置
#     config = {
#         'raw_data_dir': f"{raw_data_dir}/{exchange_name}",
#         'output_dir': f"{output_dir}/{exchange_name}",
#         'bar_type': bar_type,
#         'threshold': threshold,
#         'symbols': symbols,
#         'process_num_limit': 4,
#         'batch_size': 10
#     }
#
#     try:
#         # 创建处理上下文和处理器
#         context = BarProcessorContext(config)
#         backend_processor = BarProcessor(context)
#
#         # 运行 Bar 生成流水线
#         result = backend_processor.run_bar_generation_pipeline(config)
#
#         logger.info(f"\n🎉 {exchange_name.upper()} Bar 生成完成！")
#         logger.info(f"📊 处理结果: {result}")
#
#         return result
#
#     except Exception as e:
#         logger.error(f"❌ Bar 生成失败: {e}")
#         raise