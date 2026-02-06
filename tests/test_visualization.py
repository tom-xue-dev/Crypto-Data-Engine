"""
Unit tests for backtest visualization module.
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from crypto_data_engine.core.base import BacktestResult
from crypto_data_engine.services.back_test.visualization import (
    ChartConfig,
    plot_equity_curve,
    plot_drawdown,
    plot_returns_distribution,
    plot_monthly_returns,
    create_metrics_table,
    create_report,
    PLOTLY_AVAILABLE,
    MATPLOTLIB_AVAILABLE,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_backtest_result():
    """Create a sample BacktestResult for testing."""
    np.random.seed(42)
    n_periods = 200
    
    # Generate NAV history
    returns = np.random.randn(n_periods) * 0.02
    nav = 100_000 * np.exp(np.cumsum(returns))
    nav_list = nav.tolist()
    
    return BacktestResult(
        start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2024, 6, 30, tzinfo=timezone.utc),
        initial_capital=100_000,
        final_capital=nav_list[-1],
        total_return=(nav_list[-1] - 100_000) / 100_000,
        annual_return=0.15,
        sharpe_ratio=1.5,
        sortino_ratio=2.0,
        max_drawdown=-0.12,
        max_drawdown_duration=15,
        calmar_ratio=1.25,
        win_rate=0.55,
        profit_factor=1.8,
        total_trades=50,
        winning_trades=28,
        losing_trades=22,
        avg_trade_return=0.003,
        avg_win=0.02,
        avg_loss=-0.015,
        largest_win=0.08,
        largest_loss=-0.05,
        avg_holding_period=24.0,
        nav_history=nav_list,
        trades=[],
        portfolio_history=[],
    )


# =============================================================================
# Config Tests
# =============================================================================

class TestChartConfig:
    """Tests for ChartConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = ChartConfig()
        
        assert config.width == 1200
        assert config.height == 600
        assert "primary" in config.colors
        assert "negative" in config.colors

    def test_custom_config(self):
        """Test custom configuration."""
        config = ChartConfig(
            width=800,
            height=400,
            title_font_size=20,
        )
        
        assert config.width == 800
        assert config.height == 400
        assert config.title_font_size == 20


# =============================================================================
# Metrics Tests
# =============================================================================

class TestMetricsTable:
    """Tests for metrics table creation."""

    def test_create_metrics_table(self, sample_backtest_result):
        """Test creating metrics table."""
        table = create_metrics_table(sample_backtest_result)
        
        assert isinstance(table, pd.DataFrame)
        assert "Metric" in table.columns
        assert "Value" in table.columns
        assert len(table) > 0
        
        # Check some expected metrics
        metrics = table.set_index("Metric")["Value"]
        assert "Total Return" in metrics.index
        assert "Sharpe Ratio" in metrics.index
        assert "Max Drawdown" in metrics.index


# =============================================================================
# Plotting Tests
# =============================================================================

@pytest.mark.skipif(
    not (PLOTLY_AVAILABLE or MATPLOTLIB_AVAILABLE),
    reason="No plotting library available"
)
class TestPlotting:
    """Tests for plotting functions."""

    def test_plot_equity_curve(self, sample_backtest_result):
        """Test equity curve plotting."""
        fig = plot_equity_curve(sample_backtest_result)
        
        assert fig is not None

    def test_plot_equity_curve_with_benchmark(self, sample_backtest_result):
        """Test equity curve with benchmark."""
        np.random.seed(43)
        benchmark = pd.Series(100_000 * np.exp(np.cumsum(np.random.randn(200) * 0.015)))
        
        fig = plot_equity_curve(sample_backtest_result, benchmark=benchmark)
        
        assert fig is not None

    def test_plot_drawdown(self, sample_backtest_result):
        """Test drawdown plotting."""
        fig = plot_drawdown(sample_backtest_result)
        
        assert fig is not None

    def test_plot_returns_distribution(self, sample_backtest_result):
        """Test returns distribution plotting."""
        fig = plot_returns_distribution(sample_backtest_result)
        
        assert fig is not None

    def test_plot_monthly_returns(self, sample_backtest_result):
        """Test monthly returns plotting."""
        fig = plot_monthly_returns(sample_backtest_result)
        
        assert fig is not None


# =============================================================================
# Report Tests
# =============================================================================

@pytest.mark.skipif(
    not (PLOTLY_AVAILABLE or MATPLOTLIB_AVAILABLE),
    reason="No plotting library available"
)
class TestReport:
    """Tests for report generation."""

    def test_create_report(self, sample_backtest_result):
        """Test creating comprehensive report."""
        report = create_report(sample_backtest_result)
        
        assert isinstance(report, dict)
        assert "metrics" in report
        assert "equity_curve" in report
        assert "drawdown" in report
        assert "returns_dist" in report

    def test_create_report_with_benchmark(self, sample_backtest_result):
        """Test creating report with benchmark."""
        np.random.seed(43)
        benchmark = pd.Series(100_000 * np.exp(np.cumsum(np.random.randn(200) * 0.015)))
        
        report = create_report(sample_backtest_result, benchmark=benchmark)
        
        assert isinstance(report, dict)
        assert "equity_curve" in report


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_minimal_nav_history(self):
        """Test with minimal NAV history."""
        result = BacktestResult(
            start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2024, 1, 2, tzinfo=timezone.utc),
            initial_capital=100_000,
            final_capital=101_000,
            nav_history=[100_000, 101_000],
            trades=[],
            portfolio_history=[],
        )
        
        table = create_metrics_table(result)
        assert len(table) > 0

    @pytest.mark.skipif(
        not (PLOTLY_AVAILABLE or MATPLOTLIB_AVAILABLE),
        reason="No plotting library available"
    )
    def test_plot_with_short_history(self):
        """Test plotting with short history."""
        result = BacktestResult(
            start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2024, 1, 5, tzinfo=timezone.utc),
            initial_capital=100_000,
            final_capital=102_000,
            nav_history=[100_000, 100_500, 101_000, 101_500, 102_000],
            trades=[],
            portfolio_history=[],
        )
        
        fig = plot_equity_curve(result)
        assert fig is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
