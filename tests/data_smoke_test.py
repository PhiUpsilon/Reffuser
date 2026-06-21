"""Executable check for the public time-series data utilities."""

import numpy as np

from reffuser import SlidingWindowDataset, StandardScaler


def main() -> None:
    values = np.arange(60, dtype=np.float32).reshape(20, 3)
    scaler = StandardScaler().fit(values[:12])
    normalized = scaler.transform(values)
    np.testing.assert_allclose(scaler.inverse_transform(normalized), values)

    dataset = SlidingWindowDataset(normalized, history_length=4, horizon=3, stride=2)
    history, future = dataset[1]
    assert len(dataset) == 7
    assert history.shape == (4, 3)
    assert future.shape == (3, 3)
    print("Public data utilities smoke test passed.")


if __name__ == "__main__":
    main()
