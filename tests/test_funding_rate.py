"""Tests for funding rate downloader and loader."""
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Downloader tests
# ---------------------------------------------------------------------------

class TestFundingRateDownloader:

    def test_parse_api_response(self):
        """Verify API response is correctly parsed to DataFrame."""
        mock_records = [
            {
                "symbol": "BTCUSDT",
                "fundingRate": "0.00010000",
                "fundingTime": 1704067200000,
                "markPrice": "42000.50",
            },
            {
                "symbol": "BTCUSDT",
                "fundingRate": "-0.00005000",
                "fundingTime": 1704096000000,
                "markPrice": "42100.00",
            },
        ]
        df = pd.DataFrame(mock_records)
        df = df.rename(columns={
            "fundingTime": "timestamp",
            "fundingRate": "funding_rate",
            "markPrice": "mark_price",
        })
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["funding_rate"] = pd.to_numeric(df["funding_rate"])
        df["mark_price"] = pd.to_numeric(df["mark_price"])
        df = df[["timestamp", "funding_rate", "mark_price"]]

        assert len(df) == 2
        assert df["funding_rate"].iloc[0] == pytest.approx(0.0001)
        assert df["funding_rate"].iloc[1] == pytest.approx(-0.00005)
        assert df["mark_price"].iloc[0] == pytest.approx(42000.50)

    @patch("crypto_data_engine.services.funding_rate.downloader._rate_limited_get")
    def test_pagination_stops_on_empty(self, mock_get, tmp_path):
        """Verify download returns None when API returns empty list."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = []
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        from crypto_data_engine.services.funding_rate.downloader import (
            FundingRateDownloader,
        )

        dl = FundingRateDownloader(output_dir=tmp_path)
        result = dl.download_symbol_month("TESTUSDT", 2099, 1)
        assert result is None

    @patch("crypto_data_engine.services.funding_rate.downloader._rate_limited_get")
    def test_download_saves_parquet(self, mock_get, tmp_path):
        """Verify successful download produces a parquet file."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {
                "symbol": "BTCUSDT",
                "fundingRate": "0.0001",
                "fundingTime": 1704067200000,
                "markPrice": "42000.0",
            },
            {
                "symbol": "BTCUSDT",
                "fundingRate": "0.0002",
                "fundingTime": 1704096000000,
                "markPrice": "42100.0",
            },
        ]
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        from crypto_data_engine.services.funding_rate.downloader import (
            FundingRateDownloader,
        )

        dl = FundingRateDownloader(output_dir=tmp_path)
        result = dl.download_symbol_month("BTCUSDT", 2024, 1)

        assert result is not None
        assert result.exists()
        df = pd.read_parquet(result)
        assert len(df) == 2
        assert list(df.columns) == ["timestamp", "funding_rate", "mark_price"]

    @patch("crypto_data_engine.services.funding_rate.downloader._rate_limited_get")
    def test_incremental_skip(self, mock_get, tmp_path):
        """Verify existing parquet is skipped (incremental download)."""
        sym_dir = tmp_path / "BTCUSDT"
        sym_dir.mkdir()
        existing = sym_dir / "BTCUSDT-fundingRate-2024-01.parquet"
        existing.write_text("dummy")

        from crypto_data_engine.services.funding_rate.downloader import (
            FundingRateDownloader,
        )

        dl = FundingRateDownloader(output_dir=tmp_path)
        result = dl.download_symbol_month("BTCUSDT", 2024, 1)
        assert result is None
        mock_get.assert_not_called()


# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------

class TestFundingRateLoader:

    def test_daily_aggregation(self, tmp_path):
        """Verify 3 intraday rates sum to correct daily rate."""
        sym_dir = tmp_path / "TESTUSDT"
        sym_dir.mkdir()

        df = pd.DataFrame({
            "timestamp": pd.to_datetime([
                "2024-01-15 00:00:00",
                "2024-01-15 08:00:00",
                "2024-01-15 16:00:00",
            ]),
            "funding_rate": [0.0001, 0.0002, -0.00005],
            "mark_price": [42000.0, 42100.0, 42050.0],
        })
        df.to_parquet(
            sym_dir / "TESTUSDT-fundingRate-2024-01.parquet", index=False
        )

        from crypto_data_engine.services.funding_rate.loader import (
            load_daily_funding_rates,
        )

        result = load_daily_funding_rates(["TESTUSDT"], data_dir=tmp_path)

        assert "TESTUSDT" in result
        d = date(2024, 1, 15)
        assert d in result["TESTUSDT"]
        expected = 0.0001 + 0.0002 + (-0.00005)
        assert result["TESTUSDT"][d] == pytest.approx(expected)

    def test_missing_symbol_returns_empty(self, tmp_path):
        """Missing symbol directory should be silently skipped."""
        from crypto_data_engine.services.funding_rate.loader import (
            load_daily_funding_rates,
        )

        result = load_daily_funding_rates(["NOSUCHUSDT"], data_dir=tmp_path)
        assert "NOSUCHUSDT" not in result

    def test_load_multiple_months(self, tmp_path):
        """Verify data from multiple monthly files is concatenated correctly."""
        sym_dir = tmp_path / "ETHUSDT"
        sym_dir.mkdir()

        for month, ts_str, rate in [
            (1, "2024-01-15 00:00:00", 0.0001),
            (2, "2024-02-10 08:00:00", -0.0002),
        ]:
            df = pd.DataFrame({
                "timestamp": pd.to_datetime([ts_str]),
                "funding_rate": [rate],
                "mark_price": [3000.0],
            })
            df.to_parquet(
                sym_dir / f"ETHUSDT-fundingRate-2024-{month:02d}.parquet",
                index=False,
            )

        from crypto_data_engine.services.funding_rate.loader import (
            load_funding_rates,
        )

        df = load_funding_rates("ETHUSDT", data_dir=tmp_path)
        assert len(df) == 2


# ---------------------------------------------------------------------------
# Funding cost math tests
# ---------------------------------------------------------------------------

class TestFundingCostMath:
    """Verify pnl -= w * fr produces correct sign for all cases."""

    def test_long_positive_rate(self):
        """Long pays when funding rate is positive."""
        w, fr = 0.1, 0.0001
        cost = w * fr
        assert cost > 0  # subtracted from PnL -> loss

    def test_short_positive_rate(self):
        """Short receives when funding rate is positive."""
        w, fr = -0.1, 0.0001
        cost = w * fr
        assert cost < 0  # subtracted from PnL -> gain

    def test_long_negative_rate(self):
        """Long receives when funding rate is negative."""
        w, fr = 0.1, -0.0001
        cost = w * fr
        assert cost < 0  # subtracted from PnL -> gain

    def test_short_negative_rate(self):
        """Short pays when funding rate is negative."""
        w, fr = -0.1, -0.0001
        cost = w * fr
        assert cost > 0  # subtracted from PnL -> loss

    def test_zero_rate_no_impact(self):
        """Zero funding rate has no impact."""
        w, fr = 0.1, 0.0
        cost = w * fr
        assert cost == 0.0
