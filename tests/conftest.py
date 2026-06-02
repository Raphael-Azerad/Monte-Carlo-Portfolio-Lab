from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    index = pd.bdate_range("2024-01-01", periods=120)
    base = np.linspace(100, 125, len(index))
    return pd.DataFrame(
        {
            "SPY": base,
            "QQQ": base * 1.1 + np.sin(np.arange(len(index))),
            "TLT": 95 + np.linspace(0, 4, len(index)),
            "GLD": 180 + np.linspace(0, 8, len(index)),
        },
        index=index,
    )
