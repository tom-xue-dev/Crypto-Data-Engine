import pandas as pd
from datetime import datetime

from crypto_data_engine.services.back_test.engine.cross_sectional import CrossSectionalEngine
from crypto_data_engine.services.back_test.config import BacktestConfig, BacktestMode, CostConfigModel
from crypto_data_engine.core.base import BaseStrategy


class AlwaysLongOne(BaseStrategy):
    def __init__(self):
        super().__init__(name="AlwaysLongOne")

    def generate_weights(self, cross_section: pd.DataFrame, current_weights=None):
        # Assume single-asset cross-section
        assets = list(cross_section.index)
        if not assets:
            return {}
        return {assets[0]: 1.0}


def make_data():
    # Three daily timestamps, single asset AAA
    idx = pd.MultiIndex.from_product(
        [
            pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"]),
            ["AAA"],
        ],
        names=["timestamp", "asset"],
    )
    df = pd.DataFrame(index=idx, data={
        "close": [100.0, 110.0, 120.0],
        "volume": [1000, 1000, 1000],
    })
    return df


def test_cross_sectional_rebalance_executes_next_bar():
    data = make_data()
    strategy = AlwaysLongOne()
    cost = CostConfigModel(commission_rate=0.0, taker_rate=0.0, maker_rate=0.0, slippage_rate=0.0)
    from crypto_data_engine.services.back_test.config import RiskConfigModel
    cfg = BacktestConfig(
        mode=BacktestMode.CROSS_SECTIONAL,
        initial_capital=1000.0,
        rebalance_frequency="D",
        price_col="close",
        start_date=datetime(2021, 1, 1),
        end_date=datetime(2021, 1, 3),
        cost_config=cost,
        risk_config=RiskConfigModel(max_position_size=1.0),
    )

    engine = CrossSectionalEngine(cfg, strategy)
    result = engine.run(data)

    # With delayed execution, we buy on 2021-01-02 at price 110 and close at 120
    # Expected final NAV about 1000 * (120/110) = ~1090.9 (no position cap)
    assert 1089 < result.final_capital < 1092
