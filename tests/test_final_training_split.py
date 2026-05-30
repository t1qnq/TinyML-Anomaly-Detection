from types import SimpleNamespace
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from final_training import split_train_calibration_test


def test_split_train_calibration_test_uses_independent_65_15_20_partitions():
    x_raw = np.arange(100 * 19, dtype=np.float32).reshape(100, 19)
    args = SimpleNamespace(seed=42, train_size=0.65, calibration_size=0.15, test_size=0.20)

    train_raw, calibration_raw, test_raw = split_train_calibration_test(x_raw, "GENTLE", args)

    assert len(train_raw) == 65
    assert len(calibration_raw) == 15
    assert len(test_raw) == 20

    train_ids = set(train_raw[:, 0].astype(int))
    calibration_ids = set(calibration_raw[:, 0].astype(int))
    test_ids = set(test_raw[:, 0].astype(int))

    assert train_ids.isdisjoint(calibration_ids)
    assert train_ids.isdisjoint(test_ids)
    assert calibration_ids.isdisjoint(test_ids)
    assert train_ids | calibration_ids | test_ids == set(x_raw[:, 0].astype(int))
