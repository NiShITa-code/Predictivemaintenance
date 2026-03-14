from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import prepare_cmapss_datasets
from .models import build_model
from .utils import ensure_dir, mae, rmse, save_json, set_seed


def run_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for X, y, _ in dataloader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        preds, _ = model(X)
        loss = criterion(preds, y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(dataloader))


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds_all, y_all = [], []
    attn_store = []
    metadata_all = []

    for X, y, metadata in dataloader:
        X = X.to(device)
        y = y.to(device)

        preds, attn = model(X)
        loss = criterion(preds, y)
        total_loss += loss.item()

        preds_all.append(preds.cpu().numpy())
        y_all.append(y.cpu().numpy())

        if attn is not None:
            attn_store.append(attn.cpu().numpy())

        if isinstance(metadata, dict):
            batch_size = len(next(iter(metadata.values())))
            for i in range(batch_size):
                metadata_all.append({k: metadata[k][i] for k in metadata})
        else:
            metadata_all.extend(metadata)

    if len(preds_all) == 0:
        raise ValueError("Evaluation dataloader produced no batches.")

    preds_all = np.vstack(preds_all).reshape(-1)
    y_all = np.vstack(y_all).reshape(-1)
    attn_store = np.vstack(attn_store) if len(attn_store) > 0 else None

    metrics = {
        "loss": total_loss / max(1, len(dataloader)),
        "rmse": rmse(y_all, preds_all),
        "mae": mae(y_all, preds_all),
    }

    return metrics, preds_all, y_all, attn_store, metadata_all


def plot_curves(train_losses, val_losses, out_path):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.title("Training curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_predictions(y_true, y_pred, out_path, n=300):
    plt.figure(figsize=(10, 4))
    n = min(n, len(y_true))
    plt.plot(y_true[:n], label="actual")
    plt.plot(y_pred[:n], label="predicted")
    plt.xlabel("sample")
    plt.ylabel("RUL")
    plt.title("Predicted vs actual RUL")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_attention(attn_weights, out_path, max_rows=20):
    if attn_weights is None:
        return

    rows = min(max_rows, attn_weights.shape[0])
    plt.figure(figsize=(10, 5))
    plt.imshow(attn_weights[:rows], aspect="auto", cmap="viridis")
    plt.colorbar(label="attention weight")
    plt.xlabel("time step in window")
    plt.ylabel("sample")
    plt.title("Temporal attention heatmap")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main(args):
    set_seed(args.seed)
    ensure_dir(args.artifact_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler_path = str(Path(args.artifact_dir) / f"{args.dataset}_scaler.joblib")

    bundle = prepare_cmapss_datasets(
        data_dir=args.data_dir,
        dataset=args.dataset,
        seq_len=args.seq_len,
        val_ratio=args.val_ratio,
        max_rul=args.max_rul,
        scaler_path=scaler_path,
        seed=args.seed,
        fit_scaler_on_train=True,
    )

    train_loader = DataLoader(bundle.train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(bundle.val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(bundle.test_dataset, batch_size=args.batch_size, shuffle=False)

    model = build_model(
        args.model,
        input_size=len(bundle.feature_columns),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_path = Path(args.artifact_dir) / f"{args.dataset}_{args.model}_best.pt"

    train_losses, val_losses = [], []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics, _, _, _, _ = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_metrics["loss"])

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_rmse={val_metrics['rmse']:.4f} | "
            f"val_mae={val_metrics['mae']:.4f}"
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "feature_columns": bundle.feature_columns,
                    "args": vars(args),
                    "scaler_path": bundle.scaler_path,
                },
                best_path,
            )

    elapsed = time.time() - start
    print(f"Training finished in {elapsed:.1f}s")

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_metrics_final, val_pred, val_true, _, _ = evaluate(model, val_loader, criterion, device)
    test_metrics, y_pred, y_true, attn, metadata = evaluate(model, test_loader, criterion, device)

    anomaly_threshold = np.quantile(np.abs(val_true - val_pred), 0.95)
    anomaly_rate = float(np.mean(np.abs(y_true - y_pred) >= anomaly_threshold))

    summary = {
        "dataset": args.dataset,
        "model": args.model,
        "seq_len": args.seq_len,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "epochs": args.epochs,
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "test_loss": test_metrics["loss"],
        "anomaly_threshold_abs_error_p95": float(anomaly_threshold),
        "anomaly_rate": anomaly_rate,
        "best_model_path": str(best_path),
        "train_time_seconds": elapsed,
    }

    save_json(summary, str(Path(args.artifact_dir) / f"{args.dataset}_{args.model}_summary.json"))

    np.save(Path(args.artifact_dir) / f"{args.dataset}_{args.model}_y_true.npy", y_true)
    np.save(Path(args.artifact_dir) / f"{args.dataset}_{args.model}_y_pred.npy", y_pred)
    np.save(Path(args.artifact_dir) / f"{args.dataset}_{args.model}_train_losses.npy", np.array(train_losses))
    np.save(Path(args.artifact_dir) / f"{args.dataset}_{args.model}_val_losses.npy", np.array(val_losses))

    if attn is not None:
        np.save(Path(args.artifact_dir) / f"{args.dataset}_{args.model}_attention.npy", attn)

    plot_curves(train_losses, val_losses, str(Path(args.artifact_dir) / f"{args.dataset}_{args.model}_loss.png"))
    plot_predictions(y_true, y_pred, str(Path(args.artifact_dir) / f"{args.dataset}_{args.model}_predictions.png"))
    plot_attention(attn, str(Path(args.artifact_dir) / f"{args.dataset}_{args.model}_attention.png"))

    print("Test metrics:", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="FD001")
    parser.add_argument("--model", type=str, default="lstm", choices=["rnn", "gru", "lstm", "attention"])
    parser.add_argument("--artifact_dir", type=str, default="artifacts")
    parser.add_argument("--seq_len", type=int, default=50)
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

    main(parser.parse_args())