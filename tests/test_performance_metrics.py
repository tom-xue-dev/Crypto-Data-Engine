import pandas as pd
from datetime import datetime, timedelta

from crypto_data_engine.core.base import calculate_performance_metrics


def test_annual_return_time_aware():
    # NAV doubles over 2 calendar years -> annual return ~41.4%
    start = datetime(2020, 1, 1)
    end = datetime(2022, 1, 1)
    idx = pd.to_datetime([start, end])
    nav = pd.Series([100.0, 200.0], index=idx)

    m = calculate_performance_metrics(nav)
    # (1+R)^2 = 2 -> R = sqrt(2)-1 ~ 0.4142
    assert abs(m["annual_return"] - (2 ** 0.5 - 1)) < 1e-3

