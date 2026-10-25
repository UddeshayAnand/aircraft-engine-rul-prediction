"""Strict C-MAPSS parsing and causal, per-engine feature construction."""
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["engine_id", "cycle"] + [f"setting_{i}" for i in range(1, 4)] + [f"s{i}" for i in range(1, 22)]
# Matches the supplied notebook; selection was not re-tuned on test outcomes.
FEATURES = ["setting_1", "setting_2"] + [f"s{i}" for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]]


def read_trajectories(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=r"\s+", header=None)
    if frame.shape[1] != 26 or frame.empty:
        raise ValueError("Expected a nonempty C-MAPSS file with 26 columns")
    frame.columns = COLUMNS
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("All measurements must be finite")
    for col in ["engine_id", "cycle"]:
        if (frame[col] < 1).any() or (frame[col] % 1 != 0).any():
            raise ValueError(f"{col} must contain positive integers")
        frame[col] = frame[col].astype(int)
    if frame.duplicated(["engine_id", "cycle"]).any():
        raise ValueError("Duplicate engine/cycle records")
    frame = frame.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
    if not frame.groupby("engine_id")["cycle"].apply(lambda s: np.array_equal(s, np.arange(1, len(s) + 1))).all():
        raise ValueError("Each trajectory must contain contiguous cycles beginning at 1")
    return frame


def training_target(frame: pd.DataFrame) -> pd.Series:
    return frame.groupby("engine_id")["cycle"].transform("max") - frame["cycle"]


def features(frame: pd.DataFrame, window: int = 1) -> pd.DataFrame:
    """Only current/past readings enter features; engine IDs never enter models."""
    if window < 1:
        raise ValueError("window must be positive")
    result = frame[FEATURES].copy()
    if window > 1:
        for col in FEATURES:
            result[f"{col}_mean{window}"] = frame.groupby("engine_id")[col].transform(
                lambda s: s.rolling(window, min_periods=1).mean()
            )
    return result


def test_endpoints(frame: pd.DataFrame, rul_path: Path):
    endpoints = frame.groupby("engine_id", sort=True).tail(1).sort_values("engine_id")
    truth = np.loadtxt(rul_path, ndmin=1)
    if truth.ndim != 1 or len(truth) != len(endpoints) or not np.isfinite(truth).all() or (truth < 0).any():
        raise ValueError("RUL file must have one finite nonnegative value per engine")
    if not np.array_equal(endpoints.engine_id, np.arange(1, len(truth) + 1)):
        raise ValueError("RUL row alignment requires contiguous engine IDs starting at 1")
    return endpoints.index.to_numpy(), truth


def validation_endpoints(frame: pd.DataFrame, seed: int) -> np.ndarray:
    """One deterministic pseudo-censored observation per held-out engine."""
    rng = np.random.default_rng(seed)
    return np.array([rng.choice(group.index[max(0, int(len(group) * .2)):len(group) - 1])
                     for _, group in frame.groupby("engine_id", sort=True)])
