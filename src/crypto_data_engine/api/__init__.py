"""
FastAPI backend for crypto data engine backtesting system.

Provides REST API for:
- Running backtests
- Configuring strategies
- Retrieving results and visualizations
"""

from .main import app, create_app

__all__ = ["app", "create_app"]
