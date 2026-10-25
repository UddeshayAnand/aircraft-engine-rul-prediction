import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def scores(y, prediction):
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    if y.ndim != 1 or y.shape != prediction.shape or not len(y) or not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError("Expected aligned, nonempty, finite prediction and truth vectors")
    error = prediction - y
    # Positive errors overestimate remaining life and receive the larger penalty.
    penalty = np.expm1(np.where(error < 0, -error / 13.0, error / 10.0))
    return {"rmse": float(np.sqrt(mean_squared_error(y, prediction))),
            "mae": float(mean_absolute_error(y, prediction)),
            "r2": float(r2_score(y, prediction)),
            "asymmetric_score_sum": float(penalty.sum())}
