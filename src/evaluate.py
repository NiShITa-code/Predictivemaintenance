from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import prepare_cmapss_datasets
from .models import build_model
from .train import evaluate
from .utils import ensure_dir, save_json


def main(args):
    ensure_dir(args.artifact_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.model_path, map_location=device)
    run_stem = Path(args.model_path).stem.replace("_best", "")
    train_args = checkpoint["args"]

    dataset = train_args["dataset"]
    model_name = train_args["model"]
    seq_len = train_args["seq_len"]
    hidden_size = train_args["hidden_size"]
    num_layers = train_args["num_layers"]
    dropout = train_args["dropout"]
    val_ratio = train_args["val_ratio"]
    max_rul = train_args["max_rul"]
    seed = train_args["seed"]
    transformer_heads = train_args.get("transformer_heads", 4)
    transformer_ff_dim = train_args.get("transformer_ff_dim", 128)

    # Prefer scaler_path saved in checkpoint.
    # Fallback: assume scaler is stored next to the checkpoint.
    if "scaler_path" in checkpoint:
        scaler_path = checkpoint["scaler_path"]
    else:
        checkpoint_dir = Path(args.model_path).parent
        scaler_path = str(checkpoint_dir / f"{dataset}_scaler.joblib")

    bundle = prepare_cmapss_datasets(
        data_dir=args.data_dir,
        dataset=dataset,
        seq_len=seq_len,
        val_ratio=val_ratio,
        max_rul=max_rul,
        scaler_path=scaler_path,
        seed=seed,
        fit_scaler_on_train=False,
    )

    test_loader = DataLoader(bundle.test_dataset, batch_size=args.batch_size, shuffle=False)

    model = build_model(
        model_name,
        input_size=len(bundle.feature_columns),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        transformer_heads=transformer_heads,
        transformer_ff_dim=transformer_ff_dim,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    criterion = nn.MSELoss()
    metrics, y_pred, y_true, attn, metadata = evaluate(model, test_loader, criterion, device)

    save_json(metrics, str(Path(args.artifact_dir) / f"{run_stem}_eval.json"))
    np.save(Path(args.artifact_dir) / f"{run_stem}_eval_y_true.npy", y_true)
    np.save(Path(args.artifact_dir) / f"{run_stem}_eval_y_pred.npy", y_pred)

    if attn is not None:
        np.save(Path(args.artifact_dir) / f"{run_stem}_eval_attention.npy", attn)

    print(metrics)

    plt.figure(figsize=(10, 4))
    n = min(300, len(y_true))
    plt.plot(y_true[:n], label="actual")
    plt.plot(y_pred[:n], label="predicted")
    plt.legend()
    plt.title("Evaluation predictions")
    plt.tight_layout()
    plt.savefig(Path(args.artifact_dir) / f"{run_stem}_eval_plot.png", dpi=160)
    plt.close()

    if attn is not None:
        plt.figure(figsize=(10, 5))
        rows = min(20, attn.shape[0])
        plt.imshow(attn[:rows], aspect="auto", cmap="magma")
        plt.colorbar()
        plt.title("Attention heatmap")
        plt.tight_layout()
        plt.savefig(Path(args.artifact_dir) / f"{run_stem}_eval_attention.png", dpi=160)
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--artifact_dir", type=str, default="artifacts")
    parser.add_argument("--batch_size", type=int, default=64)

    main(parser.parse_args())
