"""
Unit tests for tick microstructure factor extraction.
Tests verify correctness of VPIN, Toxicity, Kyle Lambda, Burstiness,
Jump Detection, Whale Activity, and the daily extraction wrapper.
"""
import numpy as np
import pandas as pd
import pytest

from crypto_data_engine.services.feature.tick_microstructure_factors import (
    compute_vpin,
    compute_toxicity,
    compute_kyle_lambda,
    compute_burstiness,
    compute_jump_ratio,
    compute_whale_metrics,
    extract_daily_features,
)


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════
def make_tick_df(n=1000, price_start=100.0, seed=42, all_buy=False,
                 all_sell=False, alternating=False, interval_ms=100):
    """Create synthetic tick DataFrame for testing."""
    rng = np.random.RandomState(seed)
    timestamps = np.arange(n, dtype=np.int64) * interval_ms + 1_700_000_000_000
    prices = price_start + np.cumsum(rng.randn(n) * 0.01)
    prices = np.maximum(prices, 1.0)  # ensure positive
    quantities = np.abs(rng.randn(n) * 0.1) + 0.01

    if all_buy:
        is_buyer_maker = np.zeros(n, dtype=bool)
    elif all_sell:
        is_buyer_maker = np.ones(n, dtype=bool)
    elif alternating:
        is_buyer_maker = np.array([i % 2 == 0 for i in range(n)])
    else:
        is_buyer_maker = rng.choice([True, False], n)

    return pd.DataFrame({
        "price": prices,
        "quantity": quantities,
        "transact_time": timestamps,
        "is_buyer_maker": is_buyer_maker,
    })


# ══════════════════════════════════════════════════════════════════
# VPIN TESTS
# ══════════════════════════════════════════════════════════════════
class TestVPIN:
    def test_all_buys_vpin_near_one(self):
        """All buyer-initiated trades → VPIN should be close to 1.0."""
        df = make_tick_df(n=2000, all_buy=True)
        vpin = compute_vpin(df["quantity"].values, df["is_buyer_maker"].values,
                            n_buckets=20)
        assert vpin > 0.9, f"Expected VPIN > 0.9 for all buys, got {vpin}"

    def test_all_sells_vpin_near_one(self):
        """All seller-initiated trades → VPIN should also be close to 1.0."""
        df = make_tick_df(n=2000, all_sell=True)
        vpin = compute_vpin(df["quantity"].values, df["is_buyer_maker"].values,
                            n_buckets=20)
        assert vpin > 0.9, f"Expected VPIN > 0.9 for all sells, got {vpin}"

    def test_alternating_vpin_near_zero(self):
        """Strictly alternating buy/sell with equal sizes → VPIN ≈ 0."""
        n = 2000
        quantities = np.ones(n) * 0.1
        is_buyer_maker = np.array([i % 2 == 0 for i in range(n)])
        vpin = compute_vpin(quantities, is_buyer_maker, n_buckets=20)
        assert vpin < 0.15, f"Expected VPIN < 0.15 for alternating, got {vpin}"

    def test_vpin_range(self):
        """VPIN should always be in [0, 1]."""
        df = make_tick_df(n=2000)
        vpin = compute_vpin(df["quantity"].values, df["is_buyer_maker"].values)
        assert 0 <= vpin <= 1, f"VPIN out of range: {vpin}"

    def test_vpin_insufficient_data(self):
        """Too few trades → NaN."""
        vpin = compute_vpin(np.array([1.0, 2.0]), np.array([True, False]),
                            n_buckets=50)
        assert np.isnan(vpin)


# ══════════════════════════════════════════════════════════════════
# TOXICITY TESTS
# ══════════════════════════════════════════════════════════════════
class TestToxicity:
    def test_alternating_low_toxicity(self):
        """Alternating buy/sell → run_mean=1, toxicity_ratio=0."""
        ibm = np.array([True, False] * 50)
        run_mean, run_max, tox_ratio = compute_toxicity(ibm)
        assert run_mean == 1.0, f"Expected run_mean=1, got {run_mean}"
        assert run_max == 1, f"Expected run_max=1, got {run_max}"
        assert tox_ratio == 0.0, f"Expected toxicity_ratio=0, got {tox_ratio}"

    def test_all_same_direction(self):
        """All same direction → single long run."""
        ibm = np.ones(100, dtype=bool)
        run_mean, run_max, tox_ratio = compute_toxicity(ibm)
        assert run_mean == 100, f"Expected run_mean=100, got {run_mean}"
        assert run_max == 100, f"Expected run_max=100, got {run_max}"
        assert abs(tox_ratio - 1.0) < 0.02

    def test_known_run_pattern(self):
        """Pattern: 3 buy, 2 sell, 5 buy → runs [3, 2, 5]."""
        ibm = np.array(
            [False] * 3 + [True] * 2 + [False] * 5, dtype=bool
        )
        run_mean, run_max, tox_ratio = compute_toxicity(ibm)
        assert abs(run_mean - 10 / 3) < 0.01
        assert run_max == 5
        # 7 continuations out of 9 transitions
        assert abs(tox_ratio - 7 / 9) < 0.02

    def test_insufficient_data(self):
        """Too few trades → NaN."""
        ibm = np.array([True, False])
        result = compute_toxicity(ibm)
        assert np.isnan(result[0])


# ══════════════════════════════════════════════════════════════════
# KYLE LAMBDA TESTS
# ══════════════════════════════════════════════════════════════════
class TestKyleLambda:
    def test_positive_lambda_buy_pressure(self):
        """Construct data where buy pressure raises price → λ > 0."""
        rng = np.random.RandomState(123)
        n_minutes = 30
        n_per_min = 100
        total_n = n_minutes * n_per_min

        prices = np.zeros(total_n)
        quantities = np.zeros(total_n)
        timestamps = np.zeros(total_n, dtype=np.int64)
        is_buyer_maker = np.zeros(total_n, dtype=bool)

        base_price = 100.0
        for m in range(n_minutes):
            # Randomly decide if this minute is buy-heavy or sell-heavy
            is_buy_minute = rng.rand() > 0.5
            flow = rng.uniform(5, 20) * (1 if is_buy_minute else -1)
            dp = 0.01 * flow  # price impact proportional to flow

            for j in range(n_per_min):
                idx = m * n_per_min + j
                timestamps[idx] = 1_700_000_000_000 + m * 60000 + j * 500
                prices[idx] = base_price + dp * (j + 1) / n_per_min
                quantities[idx] = abs(flow) / n_per_min
                is_buyer_maker[idx] = not is_buy_minute

            base_price += dp

        lam = compute_kyle_lambda(prices, quantities, is_buyer_maker,
                                  timestamps, freq_ms=60000)
        assert lam is not None and not np.isnan(lam)
        assert lam > 0, f"Expected positive lambda, got {lam}"

    def test_insufficient_data(self):
        """Too few trades → NaN."""
        lam = compute_kyle_lambda(
            np.array([100.0, 101.0]),
            np.array([1.0, 1.0]),
            np.array([False, True]),
            np.array([1000, 2000], dtype=np.int64),
        )
        assert np.isnan(lam)

    def test_no_price_movement(self):
        """Constant price → λ should be near 0 or NaN."""
        n = 3000
        prices = np.full(n, 100.0)
        quantities = np.random.uniform(0.1, 1.0, n)
        timestamps = np.arange(n, dtype=np.int64) * 1000 + 1_700_000_000_000
        is_buyer_maker = np.random.choice([True, False], n)
        lam = compute_kyle_lambda(prices, quantities, is_buyer_maker,
                                  timestamps, freq_ms=60000)
        # With constant price, ΔP=0 always → λ near 0
        if not np.isnan(lam):
            assert abs(lam) < 1.0, f"Expected near-zero lambda, got {lam}"


# ══════════════════════════════════════════════════════════════════
# BURSTINESS TESTS
# ══════════════════════════════════════════════════════════════════
class TestBurstiness:
    def test_periodic_burstiness_negative(self):
        """Equally spaced timestamps → std≈0 → B ≈ -1."""
        ts = np.arange(1000, dtype=np.int64) * 100 + 1_700_000_000_000
        b = compute_burstiness(ts)
        assert b < -0.95, f"Expected B < -0.95 for periodic, got {b}"

    def test_bursty_positive(self):
        """Clustered arrivals (mostly short, rare long gaps) → B > 0."""
        rng = np.random.RandomState(42)
        # 95% intervals are 1ms, 5% are 100000ms → std >> mean → B > 0
        intervals = np.where(rng.rand(500) > 0.05, 1, 100000).astype(np.int64)
        ts = np.concatenate([[1_700_000_000_000],
                             1_700_000_000_000 + np.cumsum(intervals)])
        b = compute_burstiness(ts)
        assert b > 0.3, f"Expected B > 0.3 for bursty data, got {b}"

    def test_poisson_near_zero(self):
        """Exponential intervals (Poisson process) → B ≈ 0."""
        rng = np.random.RandomState(42)
        intervals = rng.exponential(100, 5000).astype(np.int64) + 1
        ts = np.concatenate([[1_700_000_000_000],
                             1_700_000_000_000 + np.cumsum(intervals)])
        b = compute_burstiness(ts)
        # For exponential distribution, std = mean, so B = 0
        assert -0.15 < b < 0.15, f"Expected B ≈ 0 for Poisson, got {b}"

    def test_insufficient_data(self):
        """Too few timestamps → NaN."""
        ts = np.array([1000, 2000], dtype=np.int64)
        assert np.isnan(compute_burstiness(ts))


# ══════════════════════════════════════════════════════════════════
# JUMP DETECTION TESTS
# ══════════════════════════════════════════════════════════════════
class TestJumpRatio:
    def test_smooth_price_low_jump(self):
        """Smooth Brownian motion → jump_ratio ≈ 0."""
        rng = np.random.RandomState(42)
        n = 50000
        timestamps = np.arange(n, dtype=np.int64) * 100 + 1_700_000_000_000
        log_prices = np.cumsum(rng.randn(n) * 0.0001)
        prices = 100.0 * np.exp(log_prices)

        jr = compute_jump_ratio(prices, timestamps, freq_ms=300000)
        # For pure continuous process, BV ≈ RV → jump_ratio near 0
        assert jr < 0.3, f"Expected low jump_ratio for smooth, got {jr}"

    def test_with_jump_high_ratio(self):
        """Smooth price + one big jump → jump_ratio > 0."""
        rng = np.random.RandomState(42)
        n = 50000
        timestamps = np.arange(n, dtype=np.int64) * 100 + 1_700_000_000_000
        log_prices = np.cumsum(rng.randn(n) * 0.0001)

        # Insert a large jump at the midpoint
        jump_idx = n // 2
        log_prices[jump_idx:] += 0.05  # 5% jump

        prices = 100.0 * np.exp(log_prices)
        jr = compute_jump_ratio(prices, timestamps, freq_ms=300000)
        assert jr > 0.01, f"Expected jump_ratio > 0.01 with jump, got {jr}"

    def test_jump_ratio_range(self):
        """Jump ratio should be in [0, 1]."""
        df = make_tick_df(n=50000, interval_ms=100)
        jr = compute_jump_ratio(df["price"].values, df["transact_time"].values)
        if not np.isnan(jr):
            assert 0 <= jr <= 1, f"Jump ratio out of range: {jr}"

    def test_insufficient_data(self):
        """Too few trades → NaN."""
        jr = compute_jump_ratio(np.array([100.0, 101.0]),
                                np.array([1000, 2000], dtype=np.int64))
        assert np.isnan(jr)


# ══════════════════════════════════════════════════════════════════
# WHALE ACTIVITY TESTS
# ══════════════════════════════════════════════════════════════════
class TestWhaleMetrics:
    def test_all_buy_whales(self):
        """All large trades are buyer-initiated → imbalance ≈ 1."""
        n = 5000
        rng = np.random.RandomState(42)
        quantities = rng.uniform(0.01, 0.1, n)
        # Make top 1% (50 trades) large AND all buyer-initiated
        whale_idx = rng.choice(n, 50, replace=False)
        quantities[whale_idx] = 10.0  # much larger than others
        is_buyer_maker = rng.choice([True, False], n)
        is_buyer_maker[whale_idx] = False  # buyer-initiated

        prices = 100.0 + np.cumsum(rng.randn(n) * 0.01)
        prices = np.maximum(prices, 1.0)
        timestamps = np.arange(n, dtype=np.int64) * 100 + 1_700_000_000_000

        imb, _ = compute_whale_metrics(prices, quantities, is_buyer_maker,
                                       timestamps, percentile=99)
        assert imb > 0.8, f"Expected imbalance > 0.8, got {imb}"

    def test_balanced_whales(self):
        """Equal buy/sell whale volume → imbalance ≈ 0."""
        n = 5000
        rng = np.random.RandomState(42)
        quantities = rng.uniform(0.01, 0.1, n)
        whale_idx = rng.choice(n, 50, replace=False)
        quantities[whale_idx] = 10.0

        is_buyer_maker = np.ones(n, dtype=bool)  # default seller
        # Split whales evenly: 25 buy, 25 sell
        is_buyer_maker[whale_idx[:25]] = False  # buyer-initiated
        is_buyer_maker[whale_idx[25:]] = True   # seller-initiated

        prices = 100.0 + np.cumsum(rng.randn(n) * 0.01)
        prices = np.maximum(prices, 1.0)
        timestamps = np.arange(n, dtype=np.int64) * 100 + 1_700_000_000_000

        imb, _ = compute_whale_metrics(prices, quantities, is_buyer_maker,
                                       timestamps, percentile=99)
        assert abs(imb) < 0.2, f"Expected |imbalance| < 0.2, got {imb}"

    def test_imbalance_range(self):
        """Whale imbalance should be in [-1, 1]."""
        df = make_tick_df(n=5000)
        imb, _ = compute_whale_metrics(
            df["price"].values, df["quantity"].values,
            df["is_buyer_maker"].values, df["transact_time"].values)
        if not np.isnan(imb):
            assert -1 <= imb <= 1, f"Imbalance out of range: {imb}"

    def test_insufficient_data(self):
        """Too few trades → NaN."""
        result = compute_whale_metrics(
            np.array([100.0]), np.array([1.0]),
            np.array([True]), np.array([1000], dtype=np.int64))
        assert np.isnan(result[0])


# ══════════════════════════════════════════════════════════════════
# DAILY EXTRACTION WRAPPER TESTS
# ══════════════════════════════════════════════════════════════════
class TestExtractDailyFeatures:
    def test_returns_all_expected_keys(self):
        """Output dict must contain all 9 feature keys."""
        df = make_tick_df(n=5000, interval_ms=100)
        feats = extract_daily_features(df)
        expected_keys = {
            "vpin", "toxicity_run_mean", "toxicity_run_max", "toxicity_ratio",
            "kyle_lambda", "burstiness", "jump_ratio",
            "whale_imbalance", "whale_perm_impact",
        }
        assert set(feats.keys()) == expected_keys

    def test_values_are_numeric(self):
        """All values should be float or int (or NaN)."""
        df = make_tick_df(n=5000, interval_ms=100)
        feats = extract_daily_features(df)
        for key, val in feats.items():
            assert isinstance(val, (int, float, np.integer, np.floating)), \
                f"{key} has non-numeric value: {type(val)}"

    def test_vpin_burstiness_present(self):
        """With enough data, VPIN and burstiness should not be NaN."""
        df = make_tick_df(n=10000, interval_ms=50)
        feats = extract_daily_features(df)
        assert not np.isnan(feats["vpin"]), "VPIN should not be NaN with 10k trades"
        assert not np.isnan(feats["burstiness"]), "Burstiness should not be NaN"

    def test_small_data_returns_nans(self):
        """Very few ticks → most features should be NaN."""
        df = make_tick_df(n=5, interval_ms=1000)
        feats = extract_daily_features(df)
        # With only 5 trades, most features should be NaN
        nan_count = sum(1 for v in feats.values()
                        if isinstance(v, float) and np.isnan(v))
        assert nan_count >= 5, f"Expected ≥5 NaN with 5 trades, got {nan_count}"
