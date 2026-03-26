from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_DATASETS = ["FD001", "FD002", "FD003", "FD004"]
DEFAULT_MODELS = ["rnn", "gru", "lstm", "attention", "transformer"]
DEFAULT_SEQ_LENS = [20, 30, 50, 80]


def build_train_command(args, dataset: str, model: str, seq_len: int) -> list[str]:
    return [
        args.python_executable,
        "-m",
        "src.train",
        "--data_dir",
        args.data_dir,
        "--dataset",
        dataset,
        "--model",
        model,
        "--artifact_dir",
        args.artifact_dir,
        "--seq_len",
        str(seq_len),
        "--batch_size",
        str(args.batch_size),
        "--hidden_size",
        str(args.hidden_size),
        "--num_layers",
        str(args.num_layers),
        "--dropout",
        str(args.dropout),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--val_ratio",
        str(args.val_ratio),
        "--max_rul",
        str(args.max_rul),
        "--seed",
        str(args.seed),
        "--early_stopping_patience",
        str(args.early_stopping_patience),
        "--lr_scheduler_patience",
        str(args.lr_scheduler_patience),
        "--lr_scheduler_factor",
        str(args.lr_scheduler_factor),
        "--min_lr",
        str(args.min_lr),
        "--transformer_heads",
        str(args.transformer_heads),
        "--transformer_ff_dim",
        str(args.transformer_ff_dim),
    ]


def summary_path(artifact_dir: str, dataset: str, model: str, seq_len: int) -> Path:
    return Path(artifact_dir) / f"{dataset}_{model}_seq{seq_len}_summary.json"


def checkpoint_path(artifact_dir: str, dataset: str, model: str, seq_len: int) -> Path:
    return Path(artifact_dir) / f"{dataset}_{model}_seq{seq_len}_best.pt"


def format_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def main(args):
    run_matrix = [
        (dataset, model, seq_len)
        for dataset in args.datasets
        for model in args.models
        for seq_len in args.seq_lens
    ]

    print(f"Planned runs: {len(run_matrix)}")

    executed_runs = 0
    skipped_runs = 0

    for dataset, model, seq_len in run_matrix:
        expected_summary = summary_path(args.artifact_dir, dataset, model, seq_len)
        expected_checkpoint = checkpoint_path(args.artifact_dir, dataset, model, seq_len)

        if args.skip_existing and expected_summary.exists() and expected_checkpoint.exists():
            skipped_runs += 1
            print(f"[skip] {dataset} | {model} | seq_len={seq_len} -> existing artifacts found")
            continue

        command = build_train_command(args, dataset, model, seq_len)
        print(f"[run] {dataset} | {model} | seq_len={seq_len}")
        print(format_command(command))

        if args.dry_run:
            continue

        subprocess.run(command, check=True)
        executed_runs += 1

    print(
        f"Finished experiment sweep. executed={executed_runs} skipped={skipped_runs} dry_run={args.dry_run}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a grid of predictive-maintenance training experiments."
    )
    parser.add_argument("--data_dir", type=str, default="data/CMAPSSData")
    parser.add_argument("--artifact_dir", type=str, default="artifacts")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--seq_lens", nargs="+", type=int, default=DEFAULT_SEQ_LENS)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--max_rul", type=int, default=125)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--lr_scheduler_patience", type=int, default=2)
    parser.add_argument("--lr_scheduler_factor", type=float, default=0.5)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--transformer_heads", type=int, default=4)
    parser.add_argument("--transformer_ff_dim", type=int, default=128)
    parser.add_argument(
        "--python_executable",
        type=str,
        default=sys.executable,
        help="Python executable used to launch each training run.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip runs whose expected checkpoint and summary already exist.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the planned commands without executing them.",
    )

    main(parser.parse_args())
