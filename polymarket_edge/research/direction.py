from __future__ import annotations

import numpy as np
import pandas as pd

DIRECTION_LONG_YES = "LONG_YES"
DIRECTION_LONG_NO = "LONG_NO"
DIRECTION_FLAT = "FLAT"


def direction_from_signal(signal: float, threshold: float = 0.0) -> str:
    value = float(signal)
    cut = float(max(0.0, threshold))
    if value > cut:
        return DIRECTION_LONG_YES
    if value < -cut:
        return DIRECTION_LONG_NO
    return DIRECTION_FLAT


def direction_sign(direction: str) -> int:
    d = str(direction).upper().strip()
    if d == DIRECTION_LONG_YES:
        return 1
    if d == DIRECTION_LONG_NO:
        return -1
    return 0


def direction_sign_from_signal(signal: float, threshold: float = 0.0) -> int:
    return direction_sign(direction_from_signal(signal, threshold=threshold))


def direction_label_series(signal: pd.Series, threshold: float = 0.0) -> pd.Series:
    vals = pd.to_numeric(signal, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    cut = float(max(0.0, threshold))
    out = np.full(len(vals), DIRECTION_FLAT, dtype=object)
    out[vals > cut] = DIRECTION_LONG_YES
    out[vals < -cut] = DIRECTION_LONG_NO
    return pd.Series(out, index=signal.index)


def direction_sign_series(signal: pd.Series, threshold: float = 0.0) -> pd.Series:
    vals = pd.to_numeric(signal, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    cut = float(max(0.0, threshold))
    out = np.zeros(len(vals), dtype=np.int8)
    out[vals > cut] = 1
    out[vals < -cut] = -1
    return pd.Series(out, index=signal.index)


def traded_side_from_direction(direction: str) -> str:
    d = str(direction).upper().strip()
    if d == DIRECTION_LONG_YES:
        return "YES"
    if d == DIRECTION_LONG_NO:
        return "NO"
    return "NONE"


def traded_side_series(direction: pd.Series) -> pd.Series:
    d = direction.astype(str).str.upper().str.strip()
    out = pd.Series("NONE", index=direction.index, dtype=object)
    out[d == DIRECTION_LONG_YES] = "YES"
    out[d == DIRECTION_LONG_NO] = "NO"
    return out
