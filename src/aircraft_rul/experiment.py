"""Fixed candidate comparison, grouped validation, and final test evaluation."""
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from .data import features, read_trajectories, test_endpoints, training_target, validation_endpoints
from .metrics import scores

CANDIDATES = {
    "mean_baseline": {"kind": "mean", "window": 1, "cap": None},
    "ridge_raw": {"kind": "ridge", "window": 1, "cap": None},
    "forest_raw": {"kind": "forest", "window": 1, "cap": None},
    "forest_cap125": {"kind": "forest", "window": 1, "cap": 125},
    "boosting_cap125": {"kind": "boosting", "window": 1, "cap": 125},
    "boosting_rolling_cap125": {"kind": "boosting", "window": 15, "cap": 125},
}


def estimator(kind, seed):
    if kind == "mean":
        return DummyRegressor()
    if kind == "ridge":
        return make_pipeline(RobustScaler(), Ridge(alpha=10))
    if kind == "forest":
        return RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=5,
                                     max_features=.7, random_state=seed, n_jobs=2)
    if kind == "boosting":
        # Disable the default row-wise early stopping split inside this estimator.
        return HistGradientBoostingRegressor(max_iter=200, max_leaf_nodes=15,
                                            learning_rate=.05, l2_regularization=1,
                                            early_stopping=False, random_state=seed)
    raise ValueError(kind)


def predict(model, x):
    return np.maximum(0, model.predict(x))


def run(data_dir: Path, output: Path, artifact: Path, seed: int = 42):
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    train = read_trajectories(data_dir / "train_FD001.txt")
    y = training_target(train)
    group_folds = list(GroupKFold(n_splits=5).split(train, y, train.engine_id))
    cv_rows, fold_records = [], []
    all_features = {w: features(train, w) for w in [1, 15]}
    for fold, (fit_idx, valid_idx) in enumerate(group_folds):
        evaluation = validation_endpoints(train.iloc[valid_idx], seed + fold)
        fold_records.append({"fold": fold, "train_engines": sorted(train.iloc[fit_idx].engine_id.unique().tolist()),
                             "validation_engines": sorted(train.iloc[valid_idx].engine_id.unique().tolist()),
                             "evaluation_rows": evaluation.tolist()})
        for name, spec in CANDIDATES.items():
            model = estimator(spec["kind"], seed)
            target = y if spec["cap"] is None else y.clip(upper=spec["cap"])
            x = all_features[spec["window"]]
            model.fit(x.iloc[fit_idx], target.iloc[fit_idx])
            cv_rows.append({"model": name, "fold": fold, **scores(y.loc[evaluation], predict(model, x.loc[evaluation]))})
        print(f"Completed validation fold {fold + 1}/5", flush=True)
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(output / "validation_folds.csv", index=False)
    summary = cv.groupby("model").agg(cv_rmse_mean=("rmse", "mean"), cv_rmse_std=("rmse", "std"), cv_mae_mean=("mae", "mean")).sort_values("cv_rmse_mean")
    winner = summary.index[0]
    # Test data and labels are loaded only after the candidate is selected.
    test = read_trajectories(data_dir / "test_FD001.txt")
    indices, truth = test_endpoints(test, data_dir / "RUL_FD001.txt")
    predictions = pd.DataFrame({"engine_id": test.loc[indices, "engine_id"].to_numpy(), "actual_rul": truth})
    test_rows, winner_model = [], None
    for name, spec in CANDIDATES.items():
        model = estimator(spec["kind"], seed)
        target = y if spec["cap"] is None else y.clip(upper=spec["cap"])
        model.fit(all_features[spec["window"]], target)
        estimate = predict(model, features(test, spec["window"]).loc[indices])
        predictions[name] = estimate
        test_rows.append({"model": name, **scores(truth, estimate)})
        if name == winner:
            winner_model = model
    results = summary.join(pd.DataFrame(test_rows).set_index("model"))
    results.to_csv(output / "metrics.csv")
    predictions.to_csv(output / "test_predictions.csv", index=False)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": winner_model, "spec": CANDIDATES[winner], "name": winner}, artifact)
    metadata = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "seed": seed,
                "selected_model": winner, "selection_rule": "minimum mean engine-separated validation RMSE",
                "python": platform.python_version(), "platform": platform.platform(),
                "versions": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "matplotlib", "joblib"]},
                "data_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(data_dir.glob("*.txt"))},
                "candidates": CANDIDATES, "model_parameters": {n: repr(estimator(s["kind"], seed)) for n,s in CANDIDATES.items()},
                "train_rows": len(train), "test_rows": len(test), "test_engines": len(truth), "folds": fold_records}
    (output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    plot_results(results, predictions, winner, output / "figures")
    print(results.round(3).to_string(), flush=True)
    print(f"Selected by validation: {winner}", flush=True)
    return metadata


def plot_results(results, predictions, winner, directory):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    names = list(results.index)
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    p = np.arange(len(names))
    ax.barh(p - .18, results.cv_rmse_mean, .36, label="Validation (5-fold mean)", color="#426D90")
    ax.barh(p + .18, results.rmse, .36, label="Official test endpoints", color="#D78A47")
    ax.set(yticks=p, yticklabels=names, xlabel="RMSE (cycles; lower is better)", title="FD001: fixed candidate comparison")
    ax.legend()
    fig.savefig(directory / "model_comparison.png", dpi=160)
    plt.close(fig)
    actual, estimated = predictions.actual_rul, predictions[winner]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    axes[0].scatter(actual, estimated, color="#426D90", alpha=.8)
    limit = max(actual.max(), estimated.max()) + 5
    axes[0].plot([0, limit], [0, limit], "--", color="#555555")
    axes[0].set(xlabel="Actual RUL (cycles)", ylabel="Predicted RUL (cycles)", title="100 unseen test engines")
    axes[1].scatter(actual, estimated - actual, color="#D78A47", alpha=.8)
    axes[1].axhline(0, linestyle="--", color="#555555")
    axes[1].set(xlabel="Actual RUL (cycles)", ylabel="Prediction minus actual (cycles)", title="Positive error overestimates remaining life")
    fig.suptitle(f"Validation-selected model: {winner}")
    fig.savefig(directory / "prediction_diagnostics.png", dpi=160)
    plt.close(fig)
