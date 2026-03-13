"""
Factor evaluator demo — end-to-end pipeline test.

Usage:
    python -m crypto_data_engine.services.factor_evaluator.demo
"""
from pathlib import Path

from crypto_data_engine.common.config.factor_config import (
    AnalysisConfig,
    FactorConfig,
    FactorType,
)
from crypto_data_engine.services.factor_evaluator.analyzer import FactorAnalyzer
from crypto_data_engine.services.factor_evaluator.backtester import BacktestConfig
from crypto_data_engine.services.factor_evaluator.pipeline import FactorPipeline

OUTPUT_DIR = Path(r"D:\github\quant\Crypto-Data-Engine\data\factor_reports")


def main():
    configs = [
        FactorConfig(
            name="amihud_168",
            factor_type=FactorType.CUSTOM,
            func=lambda df: (
                df["amihud"].rolling(168, min_periods=6).mean()
            ),
        ),
    ]

    pipeline = FactorPipeline(
        analyzer=FactorAnalyzer(AnalysisConfig(
            periods=(1,),
            quantiles=5,
            rebalance_freq=168,
        )),
    )

    batch_results, summary = pipeline.run(
        start_date="2025-06",
        end_date="2026-01",
        factor_configs=configs,
        symbols=None,
        warmup_periods=168,
        bar_type="time",
        interval="1h",
        columns=["amihud"],
        workers=12,
        output_dir=OUTPUT_DIR,
        charts=True,
        backtest=BacktestConfig(cost_bps=10),
    )

    print(f"\nResults: {len(batch_results)} factors")
    if not summary.empty:
        print("\n=== Factor Summary ===")
        print(summary.to_string())
    print(f"\nOutput -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
