"""
Unit tests for backtester — all with hand-verifiable mock data.
"""
import numpy as np
import pandas as pd
import pytest

from crypto_data_engine.services.factor_evaluator.backtester import (
    BacktestConfig,
    _holding_return,
    backtest_long_short,
)


# ---------------------------------------------------------------
# Helpers to build mock data
# ---------------------------------------------------------------

def make_price_df(data: dict, dates, tz="UTC"):
    """Wide price DataFrame: index=timestamps, columns=symbols."""
    idx = pd.DatetimeIndex(dates, tz=tz)
    return pd.DataFrame(data, index=idx)


def make_factor_data(records: list):
    """Build alphalens-style factor_data.

    records: list of (date_str, asset, factor_val, quantile)
    Returns DataFrame with MultiIndex (date, asset).
    """
    rows = []
    for date_str, asset, fval, q in records:
        rows.append({
            "date": pd.Timestamp(date_str),  # tz-naive (alphalens)
            "asset": asset,
            "factor": fval,
            "factor_quantile": q,
        })
    df = pd.DataFrame(rows).set_index(["date", "asset"])
    return df


# ---------------------------------------------------------------
# Tests for _holding_return
# ---------------------------------------------------------------

class TestHoldingReturn:

    def test_basic_return(self):
        """A goes 100→120 (+20%), B goes 100→90 (-10%). Mean = +5%."""
        prices = make_price_df(
            {"A": [100, 120], "B": [100, 90]},
            ["2024-01-01", "2024-01-08"],
        )
        ret = _holding_return(prices, ["A", "B"], prices.index[0], prices.index[1])
        assert ret == pytest.approx(0.05, abs=1e-10)

    def test_single_asset(self):
        """Single asset: 100→150 = +50%."""
        prices = make_price_df(
            {"X": [100, 150]},
            ["2024-01-01", "2024-01-08"],
        )
        ret = _holding_return(prices, ["X"], prices.index[0], prices.index[1])
        assert ret == pytest.approx(0.50, abs=1e-10)

    def test_empty_assets(self):
        prices = make_price_df({"X": [100, 150]}, ["2024-01-01", "2024-01-08"])
        ret = _holding_return(prices, [], prices.index[0], prices.index[1])
        assert ret == 0.0

    def test_missing_asset_ignored(self):
        prices = make_price_df({"A": [100, 120]}, ["2024-01-01", "2024-01-08"])
        ret = _holding_return(prices, ["A", "MISSING"], prices.index[0], prices.index[1])
        assert ret == pytest.approx(0.20, abs=1e-10)

    def test_all_missing(self):
        prices = make_price_df({"A": [100, 120]}, ["2024-01-01", "2024-01-08"])
        ret = _holding_return(prices, ["X", "Y"], prices.index[0], prices.index[1])
        assert ret == 0.0

    def test_negative_return(self):
        prices = make_price_df({"A": [200, 100]}, ["2024-01-01", "2024-01-08"])
        ret = _holding_return(prices, ["A"], prices.index[0], prices.index[1])
        assert ret == pytest.approx(-0.50, abs=1e-10)

    def test_nan_price_dropped(self):
        prices = make_price_df(
            {"A": [100, 120], "B": [100, np.nan]},
            ["2024-01-01", "2024-01-08"],
        )
        ret = _holding_return(prices, ["A", "B"], prices.index[0], prices.index[1])
        assert ret == pytest.approx(0.20, abs=1e-10)

    def test_equal_prices(self):
        prices = make_price_df(
            {"A": [100, 100], "B": [50, 50]},
            ["2024-01-01", "2024-01-08"],
        )
        ret = _holding_return(prices, ["A", "B"], prices.index[0], prices.index[1])
        assert ret == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------
# Tests for backtest_long_short
# ---------------------------------------------------------------

class TestBacktestLongShort:
    """Note: backtester dates come from factor_data.
    N+1 dates in factor_data → N holding periods.
    NAV series has N+1 entries (initial 1.0 + N post-period values).
    """

    def _simple_setup(self):
        """3 assets, 3 dates, 2 holding periods.

        Prices:
            t0=100  t1=+10%/-10%  t2=+10%/-10%
        A   100     110     121     (always long, Q2)
        B   100     100     100     (always short, Q1)
        C   100      90      81     (always short, Q1)

        Period 0 (t0→t1): long=+10%, short=mean(0%,-10%)=-5%, port=+15%
        Period 1 (t1→t2): same → +15%
        NAV: 1.0 → 1.15 → 1.3225
        """
        dates = ["2024-01-01", "2024-01-08", "2024-01-15"]
        prices = make_price_df(
            {"A": [100, 110, 121], "B": [100, 100, 100], "C": [100, 90, 81]},
            dates,
        )
        records = []
        for d in dates:
            records.extend([
                (d, "A", 3.0, 2),
                (d, "B", 1.0, 1),
                (d, "C", 2.0, 1),
            ])
        factor_data = make_factor_data(records)
        return prices, factor_data

    def test_returns_no_cost(self):
        prices, factor_data = self._simple_setup()
        result = backtest_long_short(factor_data, prices, BacktestConfig(cost_bps=0))
        returns = result["returns"]

        assert len(returns) == 2
        for i in range(2):
            assert returns.iloc[i]["long_return"] == pytest.approx(0.10, abs=1e-10)
            assert returns.iloc[i]["short_return"] == pytest.approx(-0.05, abs=1e-10)
            assert returns.iloc[i]["gross_return"] == pytest.approx(0.15, abs=1e-10)

    def test_nav_no_cost(self):
        """NAV: [1.0, 1.15, 1.3225] — includes initial point."""
        prices, factor_data = self._simple_setup()
        result = backtest_long_short(factor_data, prices, BacktestConfig(cost_bps=0))
        nav = result["nav"]

        assert len(nav) == 3  # initial + 2 periods
        assert nav.iloc[0] == pytest.approx(1.0, abs=1e-10)
        assert nav.iloc[1] == pytest.approx(1.15, abs=1e-10)
        assert nav.iloc[2] == pytest.approx(1.3225, abs=1e-10)

    def test_transaction_cost(self):
        prices, factor_data = self._simple_setup()
        result = backtest_long_short(factor_data, prices, BacktestConfig(cost_bps=100))
        returns = result["returns"]

        # Period 0: turnover 2.0, cost = 2.0 * 1% = 0.02
        assert returns.iloc[0]["cost"] == pytest.approx(0.02, abs=1e-10)
        # Period 1: same weights → turnover = 0
        assert returns.iloc[1]["cost"] == pytest.approx(0.0, abs=1e-10)

        nav = result["nav"]
        # nav[0]=1.0, nav[1]=1.0*(1+0.15-0.02)=1.13, nav[2]=1.13*(1+0.15)=1.2995
        assert nav.iloc[1] == pytest.approx(1.13, abs=1e-10)
        assert nav.iloc[2] == pytest.approx(1.2995, abs=1e-10)

    def test_turnover_on_portfolio_change(self):
        dates = ["2024-01-01", "2024-01-08", "2024-01-15"]
        prices = make_price_df(
            {"A": [100, 110, 121], "B": [100, 105, 110], "C": [100, 95, 90]},
            dates,
        )
        factor_data = make_factor_data([
            ("2024-01-01", "A", 3.0, 2), ("2024-01-01", "B", 2.0, 1), ("2024-01-01", "C", 1.0, 1),
            ("2024-01-08", "A", 1.0, 1), ("2024-01-08", "B", 3.0, 2), ("2024-01-08", "C", 2.0, 1),
            ("2024-01-15", "A", 1.0, 1), ("2024-01-15", "B", 3.0, 2), ("2024-01-15", "C", 2.0, 1),
        ])
        result = backtest_long_short(factor_data, prices, BacktestConfig(cost_bps=0))
        turnover = result["turnover"]

        assert turnover.iloc[0] == pytest.approx(2.0, abs=1e-10)
        # A: -0.5 - 1.0 → |1.5|, B: 1.0 - (-0.5) → |1.5|, C: unchanged → 0
        assert turnover.iloc[1] == pytest.approx(3.0, abs=1e-10)

    def test_timezone_alignment(self):
        """factor_data tz-naive + price_df UTC."""
        dates = ["2024-01-01", "2024-01-08", "2024-01-15"]
        prices = make_price_df(
            {"A": [100, 120, 130], "B": [100, 80, 70]}, dates, tz="UTC",
        )
        factor_data = make_factor_data([
            ("2024-01-01", "A", 2.0, 2), ("2024-01-01", "B", 1.0, 1),
            ("2024-01-08", "A", 2.0, 2), ("2024-01-08", "B", 1.0, 1),
        ])
        result = backtest_long_short(factor_data, prices, BacktestConfig(cost_bps=0))
        returns = result["returns"]
        assert len(returns) == 1
        # long A: +20%, short B: -20%, port = 40%
        assert returns.iloc[0]["gross_return"] == pytest.approx(0.40, abs=1e-10)

    def test_both_tz_naive(self):
        dates = ["2024-01-01", "2024-01-08", "2024-01-15"]
        prices = make_price_df(
            {"A": [100, 120, 130], "B": [100, 80, 70]}, dates, tz=None,
        )
        factor_data = make_factor_data([
            ("2024-01-01", "A", 2.0, 2), ("2024-01-01", "B", 1.0, 1),
            ("2024-01-08", "A", 2.0, 2), ("2024-01-08", "B", 1.0, 1),
        ])
        result = backtest_long_short(factor_data, prices, BacktestConfig(cost_bps=0))
        assert result["returns"].iloc[0]["gross_return"] == pytest.approx(0.40, abs=1e-10)

    def test_long_short_same_quantile_cancels(self):
        prices = make_price_df(
            {"A": [100, 200, 300]},
            ["2024-01-01", "2024-01-08", "2024-01-15"],
        )
        factor_data = make_factor_data([
            ("2024-01-01", "A", 1.0, 1),
            ("2024-01-08", "A", 1.0, 1),
        ])
        result = backtest_long_short(
            factor_data, prices,
            BacktestConfig(cost_bps=0, long_quantile=1, short_quantile=1),
        )
        assert result["returns"].iloc[0]["gross_return"] == pytest.approx(0.0, abs=1e-10)

    def test_custom_long_short_quantiles(self):
        dates = ["2024-01-01", "2024-01-08", "2024-01-15"]
        prices = make_price_df(
            {"A": [100, 130, 150], "B": [100, 110, 120], "C": [100, 90, 80]},
            dates,
        )
        factor_data = make_factor_data([
            ("2024-01-01", "A", 3.0, 3), ("2024-01-01", "B", 2.0, 2), ("2024-01-01", "C", 1.0, 1),
            ("2024-01-08", "A", 3.0, 3), ("2024-01-08", "B", 2.0, 2), ("2024-01-08", "C", 1.0, 1),
        ])
        # long Q2 (B), short Q3 (A): port = 10% - 30% = -20%
        result = backtest_long_short(
            factor_data, prices,
            BacktestConfig(cost_bps=0, long_quantile=2, short_quantile=3),
        )
        assert result["returns"].iloc[0]["gross_return"] == pytest.approx(-0.20, abs=1e-10)

    def test_multi_period_nav_correctness(self):
        """5 dates → 4 periods. NAV includes initial 1.0 → 5 entries."""
        dates = [f"2024-01-{7*i+1:02d}" for i in range(5)]
        a_prices = [100 * 1.1**i for i in range(5)]
        b_prices = [100] * 5
        prices = make_price_df({"A": a_prices, "B": b_prices}, dates)

        records = []
        for d in dates:
            records.append((d, "A", 2.0, 2))
            records.append((d, "B", 1.0, 1))
        factor_data = make_factor_data(records)

        result = backtest_long_short(factor_data, prices, BacktestConfig(cost_bps=0))
        nav = result["nav"]
        assert len(nav) == 5  # initial + 4 periods

        assert nav.iloc[0] == pytest.approx(1.0, abs=1e-10)
        expected = 1.0
        for i in range(4):
            expected *= 1.10
            assert nav.iloc[i + 1] == pytest.approx(expected, abs=1e-8)

    def test_performance_metrics_populated(self):
        prices, factor_data = self._simple_setup()
        result = backtest_long_short(factor_data, prices, BacktestConfig(cost_bps=5))
        perf = result["performance"]

        for key in [
            "total_return", "annual_return", "sharpe_ratio",
            "max_drawdown", "total_cost", "avg_turnover",
            "n_rebalances", "long_quantile", "short_quantile",
        ]:
            assert key in perf, f"Missing key: {key}"

        assert perf["n_rebalances"] == 2
        assert perf["long_quantile"] == 2
        assert perf["short_quantile"] == 1
        # total_return should be nav[-1]/nav[0] - 1 = nav[-1] - 1 (since nav[0]=1)
        assert perf["total_return"] == pytest.approx(
            result["nav"].iloc[-1] - 1.0, abs=1e-8
        )

    def test_zero_periods(self):
        """1 date in factor_data → 0 periods → NAV has only initial point."""
        prices = make_price_df({"A": [100, 120]}, ["2024-01-01", "2024-01-08"])
        factor_data = make_factor_data([("2024-01-01", "A", 1.0, 1)])
        result = backtest_long_short(factor_data, prices)
        assert len(result["nav"]) == 1  # just the initial 1.0
        assert result["nav"].iloc[0] == 1.0
        assert len(result["returns"]) == 0

    def test_short_leg_blowup(self):
        """Short leg doubles → -95% portfolio return."""
        prices = make_price_df(
            {"A": [100, 105, 110], "B": [100, 200, 210]},
            ["2024-01-01", "2024-01-08", "2024-01-15"],
        )
        factor_data = make_factor_data([
            ("2024-01-01", "A", 2.0, 2), ("2024-01-01", "B", 1.0, 1),
            ("2024-01-08", "A", 2.0, 2), ("2024-01-08", "B", 1.0, 1),
        ])
        result = backtest_long_short(factor_data, prices, BacktestConfig(cost_bps=0))
        assert result["returns"].iloc[0]["gross_return"] == pytest.approx(-0.95, abs=1e-10)
        # nav: [1.0, 0.05]
        assert result["nav"].iloc[1] == pytest.approx(0.05, abs=1e-10)

    def test_port_ret_is_long_minus_short(self):
        """port_ret = long_ret - short_ret (not +)."""
        dates = ["2024-01-01", "2024-01-08", "2024-01-15"]
        prices = make_price_df(
            {"A": [100, 120, 130], "B": [100, 110, 120]}, dates,
        )
        factor_data = make_factor_data([
            ("2024-01-01", "A", 2.0, 2), ("2024-01-01", "B", 1.0, 1),
            ("2024-01-08", "A", 2.0, 2), ("2024-01-08", "B", 1.0, 1),
        ])
        result = backtest_long_short(factor_data, prices, BacktestConfig(cost_bps=0))
        ret = result["returns"].iloc[0]
        # long A: +20%, short B: +10%, port = 20% - 10% = +10%
        assert ret["long_return"] == pytest.approx(0.20, abs=1e-10)
        assert ret["short_return"] == pytest.approx(0.10, abs=1e-10)
        assert ret["gross_return"] == pytest.approx(0.10, abs=1e-10)

    def test_many_assets_equal_weight(self):
        """4 long assets get 25% weight each."""
        dates = ["2024-01-01", "2024-01-08", "2024-01-15"]
        prices = make_price_df(
            {"A": [100, 110, 120], "B": [100, 120, 130],
             "C": [100, 130, 140], "D": [100, 140, 150],
             "E": [100, 80, 70]},
            dates,
        )
        factor_data = make_factor_data([
            ("2024-01-01", "A", 5.0, 2), ("2024-01-01", "B", 4.0, 2),
            ("2024-01-01", "C", 3.0, 2), ("2024-01-01", "D", 2.0, 2),
            ("2024-01-01", "E", 1.0, 1),
            ("2024-01-08", "A", 5.0, 2), ("2024-01-08", "B", 4.0, 2),
            ("2024-01-08", "C", 3.0, 2), ("2024-01-08", "D", 2.0, 2),
            ("2024-01-08", "E", 1.0, 1),
        ])
        result = backtest_long_short(factor_data, prices, BacktestConfig(cost_bps=0))
        ret = result["returns"].iloc[0]
        # long: mean(10%,20%,30%,40%) = 25%, short: -20%
        assert ret["long_return"] == pytest.approx(0.25, abs=1e-10)
        assert ret["short_return"] == pytest.approx(-0.20, abs=1e-10)
        assert ret["gross_return"] == pytest.approx(0.45, abs=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
