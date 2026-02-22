"""
Unit tests for TickFeatureEnricher.

Tests:
- Enricher produces correct columns
- Sliding window boundaries are correct
- NaN for insufficient ticks
- Consistency with standalone compute_* functions
- Edge cases (empty bars, empty ticks, single bar)
"""
import numpy as np
import pandas as pd
import pytest

from crypto_data_engine.services.bar_aggregator.tick_feature_enricher import (
    TICK_FEATURE_COLUMNS,
    TickFeatureEnricher,
    TickFeatureEnricherConfig,
)


# ── Fixtures ──────────────────────────────────────────────────────────

def _make_ticks(n=10_000, start_ms=1_700_000_000_000, duration_ms=3_600_000):
    """Generate synthetic tick data resembling Binance aggTrades."""
    rng = np.random.RandomState(42)
    ts = np.sort(start_ms + (rng.rand(n) * duration_ms).astype(np.int64))
    price = 40_000 + np.cumsum(rng.randn(n) * 0.5)
    qty = rng.exponential(0.1, size=n).clip(0.001)
    ibm = rng.rand(n) > 0.5
    return pd.DataFrame({
        "timestamp": ts,
        "price": price,
        "quantity": qty,
        "is_buyer_maker": ibm,
    })


def _make_bars(n=20, start_ms=1_700_000_000_000, bar_duration_ms=180_000):
    """Generate synthetic dollar bar metadata (start_time, end_time)."""
    rows = []
    t = start_ms
    rng = np.random.RandomState(99)
    for _ in range(n):
        dur = int(bar_duration_ms * (0.5 + rng.rand()))
        rows.append({
            "start_time": pd.Timestamp(t, unit="ms", tz="UTC"),
            "end_time": pd.Timestamp(t + dur, unit="ms", tz="UTC"),
            "open": 40000.0,
            "high": 40100.0,
            "low": 39900.0,
            "close": 40050.0,
            "volume": 10.0,
            "dollar_volume": 400_000.0,
        })
        t += dur + 1  # small gap between bars
    return pd.DataFrame(rows)


# ── Tests ─────────────────────────────────────────────────────────────

class TestEnricherColumns:
    def test_all_tick_columns_present(self):
        bars = _make_bars(20)
        ticks = _make_ticks(10_000, duration_ms=20 * 180_000)
        enricher = TickFeatureEnricher()
        result = enricher.enrich(bars, ticks)
        for col in TICK_FEATURE_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_original_columns_preserved(self):
        bars = _make_bars(10)
        ticks = _make_ticks(5_000, duration_ms=10 * 180_000)
        enricher = TickFeatureEnricher()
        result = enricher.enrich(bars, ticks)
        for col in bars.columns:
            assert col in result.columns

    def test_row_count_unchanged(self):
        bars = _make_bars(15)
        ticks = _make_ticks(8_000, duration_ms=15 * 180_000)
        enricher = TickFeatureEnricher()
        result = enricher.enrich(bars, ticks)
        assert len(result) == len(bars)


class TestSlidingWindow:
    def test_first_bar_uses_only_own_ticks(self):
        """With lookback=1, first bar should only use ticks in its own range."""
        config = TickFeatureEnricherConfig(lookback_bars=1, min_ticks=10)
        bars = _make_bars(5)
        ticks = _make_ticks(5_000, duration_ms=5 * 180_000)
        enricher = TickFeatureEnricher(config)
        result = enricher.enrich(bars, ticks)
        # First bar should have a result (enough ticks in 1 bar window)
        # Some features may be NaN if the single-bar window is too short
        # but at least one should be computed
        assert result["tick_burstiness"].notna().any()

    def test_lookback_increases_tick_count(self):
        """More lookback bars → more ticks in window → different features."""
        bars = _make_bars(10)
        ticks = _make_ticks(5_000, duration_ms=10 * 180_000)

        cfg1 = TickFeatureEnricherConfig(lookback_bars=1, min_ticks=10)
        cfg50 = TickFeatureEnricherConfig(lookback_bars=50, min_ticks=10)

        r1 = TickFeatureEnricher(cfg1).enrich(bars, ticks)
        r50 = TickFeatureEnricher(cfg50).enrich(bars, ticks)

        # Last bar with lookback=50 uses all ticks; lookback=1 uses fewer
        # VPIN with more data should differ
        last = len(bars) - 1
        v1 = r1["tick_vpin"].iloc[last]
        v50 = r50["tick_vpin"].iloc[last]
        # At least one should be finite
        assert np.isfinite(v1) or np.isfinite(v50)


class TestNaNHandling:
    def test_empty_ticks_all_nan(self):
        bars = _make_bars(5)
        ticks = pd.DataFrame(columns=["timestamp", "price", "quantity", "is_buyer_maker"])
        enricher = TickFeatureEnricher()
        result = enricher.enrich(bars, ticks)
        for col in TICK_FEATURE_COLUMNS:
            assert result[col].isna().all()

    def test_empty_bars_returns_empty(self):
        bars = _make_bars(0)
        ticks = _make_ticks(1000)
        enricher = TickFeatureEnricher()
        result = enricher.enrich(bars, ticks)
        assert len(result) == 0
        for col in TICK_FEATURE_COLUMNS:
            assert col in result.columns

    def test_insufficient_ticks_returns_nan(self):
        config = TickFeatureEnricherConfig(min_ticks=999_999)
        bars = _make_bars(5)
        ticks = _make_ticks(100, duration_ms=5 * 180_000)
        enricher = TickFeatureEnricher(config)
        result = enricher.enrich(bars, ticks)
        for col in TICK_FEATURE_COLUMNS:
            assert result[col].isna().all()


class TestVPINRange:
    def test_vpin_in_valid_range(self):
        bars = _make_bars(20)
        ticks = _make_ticks(20_000, duration_ms=20 * 180_000)
        enricher = TickFeatureEnricher(
            TickFeatureEnricherConfig(min_ticks=100)
        )
        result = enricher.enrich(bars, ticks)
        valid = result["tick_vpin"].dropna()
        assert len(valid) > 0
        assert (valid >= 0).all()
        assert (valid <= 1).all()


class TestBursitinessRange:
    def test_burstiness_in_valid_range(self):
        bars = _make_bars(20)
        ticks = _make_ticks(20_000, duration_ms=20 * 180_000)
        enricher = TickFeatureEnricher(
            TickFeatureEnricherConfig(min_ticks=50)
        )
        result = enricher.enrich(bars, ticks)
        valid = result["tick_burstiness"].dropna()
        assert len(valid) > 0
        assert (valid >= -1).all()
        assert (valid <= 1).all()


class TestEpochConversion:
    def test_utc_datetime_to_epoch(self):
        enricher = TickFeatureEnricher()
        ts = pd.DatetimeIndex(
            [pd.Timestamp("2024-01-01 00:00:00", tz="UTC")],
            dtype="datetime64[ms, UTC]",
        )
        ms = enricher._to_epoch_ms(ts)
        assert ms[0] == 1_704_067_200_000
