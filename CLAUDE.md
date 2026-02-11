# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Crypto Data Engine is a quantitative cryptocurrency trading system covering tick data download, bar aggregation, backtest execution, and result visualization. The system supports multiple exchanges (Binance, OKX, Bybit) and bar types (TickBar, VolumeBar, DollarBar, TimeBar).

## Commands

### Backend

```bash
# Install dependencies
poetry install

# Start API server (production)
poetry run main serve --host 127.0.0.1 --port 8000

# Start API server (development with auto-reload)
poetry run main dev

# Run all tests
poetry run main test

# Run specific test file
poetry run main test --file test_trading_log.py

# Run tests with coverage
poetry run main test --coverage

# Initialize YAML config templates
poetry run main init

# Download tick data
poetry run main data download --start-date 2025-01 --end-date 2025-06

# Aggregate bars
poetry run main aggregate BTCUSDT --bar-type dollar_bar

# Run backtest
poetry run main backtest --strategy momentum --mode cross_sectional
```

### Frontend

```bash
cd frontend
npm install
npm run dev       # Development server at http://localhost:5173
npm run build     # Production build
```

## Architecture

### Source Layout

```
src/crypto_data_engine/
├── main.py              # CLI entry point (Typer-based)
├── app/                 # CLI command modules (server, data, aggregate, backtest, pipeline)
├── api/                 # FastAPI application
│   ├── main.py          # App factory with routers
│   ├── routers/         # Endpoint handlers
│   └── schemas/         # Pydantic models
├── core/                # Base classes and interfaces
│   ├── base.py          # TradeRecord, PortfolioSnapshot, BacktestResult, BaseStrategy
│   └── interfaces.py    # IBacktestEngine and other protocols
├── services/
│   ├── back_test/       # Backtest engine
│   │   ├── engine/      # BaseBacktestEngine, CrossSectional, TimeSeries engines
│   │   ├── portfolio/   # Portfolio, Position, OrderExecutor
│   │   ├── strategies/  # Strategy implementations
│   │   └── trading_log.py
│   ├── bar_aggregator/  # Tick-to-bar aggregation
│   │   ├── unified.py   # Main entry: aggregate_bars(), build_dollar_bars(), etc.
│   │   ├── fast_aggregator.py  # Numba-accelerated implementation
│   │   ├── bar_types.py # BarType enum and builder classes
│   │   └── tick_normalizer.py
│   ├── tick_data_scraper/  # Exchange data downloaders
│   │   ├── downloader/  # Binance, OKX adapters
│   │   └── tick_worker.py
│   ├── asset_pool/      # Dynamic asset selection
│   ├── feature/         # Factor and feature calculation
│   └── signal_generation/  # Signal generators (factor, rule, ensemble)
└── common/
    ├── config/          # Configuration classes and path definitions
    ├── logger/          # Loguru-based logging
    └── task_manager.py  # Background task management
```

### Key Design Patterns

**Backtest Engine Modes:**
- `CrossSectionalEngine`: Fixed-period rebalancing (daily/weekly/monthly) for factor strategies
- `TimeSeriesEngine`: Per-bar decision making for trend-following strategies
- Both inherit from `BaseBacktestEngine` which handles NAV tracking, trade recording, and performance calculation

**Bar Aggregation Pipeline:**
- `aggregate_bars()` in `unified.py` auto-selects implementation based on data size and Numba availability
- Uses Numba JIT for volume/dollar bars when dataset > 10k rows
- Supports streaming aggregation for large files via `StreamingAggregator`

**Strategy Interface:**
- Extend `BaseStrategy` from `core/base.py`
- Implement `generate_signal()` for time-series or `generate_weights()` for cross-sectional
- Strategies receive bar data and return signals or target weights

### Data Flow

1. **Download**: Exchange adapters fetch tick data via ccxt/REST APIs -> Parquet files in `data/tick_data/`
2. **Aggregate**: `bar_aggregator` converts ticks to bars -> `data/bar_data/{bar_type}/`
3. **Backtest**: Engine loads bars, applies strategy, executes orders via `Portfolio`/`OrderExecutor`
4. **Results**: `BacktestResult` contains NAV history, trades, metrics -> JSON/CSV in `data/backtest_logs/`

### Configuration

- Data paths defined in `common/config/paths.py` (PROJECT_ROOT, DATA_ROOT, FUTURES_DATA_ROOT)
- YAML config templates generated via `main init`
- Runtime settings via `.env` file and `pydantic-settings`

## Testing

Tests are in `tests/` directory. Key test files:
- `test_cross_sectional_engine.py`, `test_time_series_engine.py` - Engine tests
- `test_portfolio.py` - Position and order execution
- `test_bar_aggregator.py` - Aggregation correctness
- `test_e2e_backtest.py` - Integration tests

Use `conftest.py` fixtures like `project_root`, `data_dir`, and `temp_task_manager`.
