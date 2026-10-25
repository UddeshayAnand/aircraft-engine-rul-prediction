from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from aircraft_rul.data import COLUMNS, features, read_trajectories, test_endpoints as endpoints, training_target
from aircraft_rul.metrics import scores


def sample():
    frame = pd.DataFrame(np.ones((8, 26)), columns=COLUMNS)
    frame["engine_id"] = [1]*4 + [2]*4
    frame["cycle"] = [1, 2, 3, 4]*2
    frame["s2"] = [1, 2, 3, 4, 100, 200, 300, 400]
    return frame


def test_target_and_causal_features():
    frame = sample()
    assert training_target(frame).tolist() == [3, 2, 1, 0]*2
    x = features(frame, 3)
    assert x.loc[4, "s2_mean3"] == 100  # No carry-over from another engine.
    assert x.loc[2, "s2_mean3"] == 2
    changed = frame.copy()
    changed.loc[3, "s2"] = 99999
    pd.testing.assert_frame_equal(features(changed, 3).iloc[:3], x.iloc[:3])
    assert "engine_id" not in x and "cycle" not in x


def test_metric_direction_and_scale():
    late = scores([20, 30], [30, 40])
    early = scores([20, 30], [10, 20])
    assert late["rmse"] == 10
    assert late["asymmetric_score_sum"] == pytest.approx(2*np.expm1(1))
    assert late["asymmetric_score_sum"] > early["asymmetric_score_sum"]


def test_truth_alignment(tmp_path):
    path = tmp_path / "rul.txt"
    path.write_text("10\n20\n")
    indices, truth = endpoints(sample(), path)
    assert indices.tolist() == [3, 7]
    assert truth.tolist() == [10, 20]
    path.write_text("10\n")
    with pytest.raises(ValueError):
        endpoints(sample(), path)


@pytest.mark.parametrize("defect", ["duplicate", "missing", "nan", "width"])
def test_invalid_data_rejected(tmp_path, defect):
    frame = sample()
    if defect == "duplicate":
        frame.loc[1, "cycle"] = 1
    elif defect == "missing":
        frame = frame.drop(index=1)
    elif defect == "nan":
        frame["s2"] = frame["s2"].astype(float)
        frame.loc[1, "s2"] = np.inf
    else:
        frame = frame.iloc[:, :-1]
    path = tmp_path / "bad.txt"
    frame.to_csv(path, sep=" ", index=False, header=False)
    with pytest.raises(ValueError):
        read_trajectories(path)


def test_supplied_fd001_contract():
    root = Path(__file__).resolve().parents[1] / "data/raw"
    train = read_trajectories(root / "train_FD001.txt")
    test = read_trajectories(root / "test_FD001.txt")
    assert train.shape == (20631, 26)
    assert test.shape == (13096, 26)
    assert train.engine_id.nunique() == test.engine_id.nunique() == 100
    assert len(endpoints(test, root / "RUL_FD001.txt")[1]) == 100
    for a, b in GroupKFold(5).split(train, groups=train.engine_id):
        assert set(train.iloc[a].engine_id).isdisjoint(train.iloc[b].engine_id)
