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

OUTPUT_DIR = Path("./data/factor_reports")


def main():
    configs = [
        FactorConfig(
            name="LM_T24_consensus_extreme_168",
            factor_type=FactorType.CUSTOM,
            func=lambda df: (df["buy_volume"] / df["volume"] - 0.5).rolling(168, min_periods=1).mean(),
        ),
    ]

    pipeline = FactorPipeline(
        analyzer=FactorAnalyzer(AnalysisConfig(
            periods=(1,3,5),
            quantiles=5,
            rebalance_freq=168,  # weekly rebalancing with 1h bars
        )),
    )

    batch_results, summary = pipeline.run(
        start_date="2020-01",
        end_date="2025-06",
        factor_configs=configs,
        symbols=None,
        warmup_periods=72,
        bar_type="time",
        interval="1h",
        columns=["buy_volume", "volume"],
        workers=12,
        output_dir=OUTPUT_DIR,
        charts=True,
        backtest=BacktestConfig(cost_bps=0),
    )

    print(f"\nResults: {len(batch_results)} factors")
    if not summary.empty:
        print("\n=== Factor Summary ===")
        print(summary.to_string())
    print(f"\nOutput -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
