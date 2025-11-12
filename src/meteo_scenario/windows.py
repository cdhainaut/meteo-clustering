from __future__ import annotations
from typing import List, Tuple
import numpy as np
import pandas as pd


def build_windows(
    times: pd.DatetimeIndex, window_hours: int, stride_hours: int
) -> List[Tuple[int, int]]:
    if len(times) < 2:
        return []
    dt_hours = float((times[1] - times[0]).total_seconds()) / 3600.0
    win_len = max(1, int(round(window_hours / dt_hours)))
    stride = max(1, int(round(stride_hours / dt_hours)))
    wins = []
    for s in range(0, len(times) - win_len + 1, stride):
        e = s + win_len - 1
        wins.append((s, e))
    return wins
