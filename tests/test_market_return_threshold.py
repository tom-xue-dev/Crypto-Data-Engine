import os
from pathlib import Path
from datetime import datetime
import pytest

from crypto_data_engine.services.back_test.data_loader import MultiAssetDataLoader, DataLoaderConfig
from crypto_data_engine.services.back_test.engine.cross_sectional import CrossSectionalEngine
from crypto_data_engine.services.back_test.config import BacktestConfig, BacktestMode, RiskConfigModel, CostConfigModel, AssetPoolConfig
from crypto_data_engine.services.back_test.strategies.base_strategies import MomentumStrategy


@pytest.mark.slow
def test_market_annual_return_over_50_percent():
    data_root = Path("E:/data")
    if not data_root.exists():
        pytest.skip("E:/data not available in this environment")

    start = datetime(2024, 1, 1)
    end = datetime(2024, 12, 31)

    loader = MultiAssetDataLoader(DataLoaderConfig(data_dir=str(data_root), bar_data_dir=str(data_root), resample_freq="1h", quote_currency="USDT"))
    assets = loader.discover_assets(quote_currency="USDT", start_date=start, end_date=end)
    if not assets:
        pytest.skip("No assets discovered under E:/data")

    assets = assets[:150]
    data = loader.load_bars(assets=assets, start_date=start, end_date=end, bar_type="tick", auto_discover=False, max_assets=150, verbose=False)
    data = loader.load_features(data)

    strategy = MomentumStrategy(lookback_period=20, long_count=20, short_count=0)
    cfg = BacktestConfig(
        mode=BacktestMode.CROSS_SECTIONAL,
        initial_capital=1_000_000.0,
        rebalance_frequency="W-MON",
        price_col="close",
        start_date=start,
        end_date=end,
        risk_config=RiskConfigModel(max_position_size=0.1, stop_loss_enabled=False),
        cost_config=CostConfigModel(commission_rate=0.0005, taker_rate=0.0005, maker_rate=0.0002, slippage_rate=0.0008),
        asset_pool_config=AssetPoolConfig(enabled=True, reselect_frequency="1M", lookback_period="30D", selection_criteria=["dollar_volume"], top_n=150),
        allow_short=False,
    )

    engine = CrossSectionalEngine(cfg, strategy)
    result = engine.run(data)

    assert result.annual_return > 0.5

