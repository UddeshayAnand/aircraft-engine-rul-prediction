import argparse
from pathlib import Path

import joblib
import pandas as pd

from .data import features, read_trajectories
from .experiment import predict, run


def main():
    parser = argparse.ArgumentParser(description="C-MAPSS FD001 experiment and inference")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    train.add_argument("--output", type=Path, default=Path("reports/rerun"))
    train.add_argument("--artifact", type=Path, default=Path("artifacts/model.joblib"))
    train.add_argument("--seed", type=int, default=42)
    inference = sub.add_parser("predict")
    inference.add_argument("--input", type=Path, required=True)
    inference.add_argument("--artifact", type=Path, default=Path("artifacts/model.joblib"))
    inference.add_argument("--output", type=Path, default=Path("artifacts/predictions.csv"))
    args = parser.parse_args()
    if args.command == "train":
        run(args.data_dir, args.output, args.artifact, args.seed)
    else:
        bundle = joblib.load(args.artifact)  # Use only locally trained/trusted artifacts.
        frame = read_trajectories(args.input)
        last = frame.groupby("engine_id").tail(1)
        values = predict(bundle["model"], features(frame, bundle["spec"]["window"]).loc[last.index])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"engine_id": last.engine_id.to_numpy(), "predicted_rul": values}).to_csv(args.output, index=False)
        print(f"Wrote {len(last)} engine predictions to {args.output}")


if __name__ == "__main__":
    main()
