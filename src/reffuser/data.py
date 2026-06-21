"""Small, framework-independent utilities for multivariate time-series preparation."""

from typing import Tuple

import numpy as np


class StandardScaler:
    """Feature-wise standardization fitted on a training time-series split."""

    def __init__(self) -> None:
        self.mean_: np.ndarray = None
        self.scale_: np.ndarray = None

    def fit(self, values: np.ndarray) -> "StandardScaler":
        values = _as_time_feature_array(values)
        self.mean_ = values.mean(axis=0, keepdims=True)
        self.scale_ = values.std(axis=0, keepdims=True)
        self.scale_ = np.where(self.scale_ < 1e-8, 1.0, self.scale_)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return (_as_time_feature_array(values) - self.mean_) / self.scale_

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return _as_time_feature_array(values) * self.scale_ + self.mean_

    def _check_fitted(self) -> None:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Call fit() before transforming values.")


class SlidingWindowDataset:
    """Expose fixed historical windows and their following forecast windows.

    Input values follow the shape ``[time, features]``. Indexing returns a tuple of
    ``(history, future)`` arrays with shapes ``[history_length, features]`` and
    ``[horizon, features]`` respectively.
    """

    def __init__(self, values: np.ndarray, history_length: int, horizon: int, stride: int = 1) -> None:
        self.values = _as_time_feature_array(values)
        if history_length <= 0 or horizon <= 0 or stride <= 0:
            raise ValueError("history_length, horizon, and stride must be positive.")
        self.history_length = history_length
        self.horizon = horizon
        self.stride = stride
        available = len(self.values) - history_length - horizon
        self.length = 0 if available < 0 else available // stride + 1

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        if index < 0:
            index += self.length
        if index < 0 or index >= self.length:
            raise IndexError("window index out of range")
        start = index * self.stride
        split = start + self.history_length
        end = split + self.horizon
        return self.values[start:split], self.values[split:end]


def _as_time_feature_array(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("values must have shape [time, features].")
    return values
