from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

CMAPSS_COLUMNS = [
    "unit_nr",
    "time_cycles",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
    *[f"sensor_{i}" for i in range(1, 22)],
]

DEFAULT_FEATURE_COLUMNS = [
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
    *[f"sensor_{i}" for i in range(1, 22)],
]


@dataclass
class DatasetBundle:
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset
    feature_columns: List[str]
    scaler_path: str


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, metadata: List[dict] | None = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
        self.metadata = metadata or [{} for _ in range(len(X))]

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.metadata[idx]


def _read_cmapss_file(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.dropna(axis=1, how="all")

    if df.shape[1] != len(CMAPSS_COLUMNS):
        raise ValueError(
            f"Unexpected column count in {path}: got {df.shape[1]}, expected {len(CMAPSS_COLUMNS)}"
        )

    df.columns = CMAPSS_COLUMNS
    return df


def add_rul_train(df: pd.DataFrame, max_rul: int | None = 125) -> pd.DataFrame:
    max_cycle = df.groupby("unit_nr")["time_cycles"].max().rename("max_cycle")
    out = df.merge(max_cycle, on="unit_nr")
    out["RUL"] = out["max_cycle"] - out["time_cycles"]

    if max_rul is not None:
        out["RUL"] = out["RUL"].clip(upper=max_rul)

    return out.drop(columns=["max_cycle"])


def add_rul_test(df_test: pd.DataFrame, rul_path: str | Path, max_rul: int | None = 125) -> pd.DataFrame:
    rul_df = pd.read_csv(rul_path, sep=r"\s+", header=None)
    rul_df = rul_df.dropna(axis=1, how="all")
    rul_df.columns = ["RUL"]
    rul_df["unit_nr"] = np.arange(1, len(rul_df) + 1)

    max_cycle = df_test.groupby("unit_nr")["time_cycles"].max().rename("max_cycle")

    out = df_test.merge(max_cycle, on="unit_nr").merge(rul_df, on="unit_nr")
    out["RUL"] = out["RUL"] + (out["max_cycle"] - out["time_cycles"])

    if max_rul is not None:
        out["RUL"] = out["RUL"].clip(upper=max_rul)

    return out.drop(columns=["max_cycle"])


def fit_scaler(train_df: pd.DataFrame, feature_columns: List[str], scaler_path: str) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_df[feature_columns].values)

    scaler_path = Path(scaler_path)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler, scaler_path)
    return scaler


def load_scaler(scaler_path: str) -> StandardScaler:
    scaler_path = Path(scaler_path)
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    return joblib.load(scaler_path)


def transform_with_scaler(df: pd.DataFrame, feature_columns: List[str], scaler: StandardScaler) -> pd.DataFrame:
    out = df.copy()
    scaled_values = scaler.transform(out[feature_columns].astype(np.float32).values)
    out[feature_columns] = scaled_values.astype(np.float32)
    return out


def make_sequences(
    df: pd.DataFrame,
    feature_columns: List[str],
    seq_len: int,
    last_only: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    X, y, metadata = [], [], []

    for unit_nr, unit_df in df.groupby("unit_nr"):
        unit_df = unit_df.sort_values("time_cycles").reset_index(drop=True)

        feats = unit_df[feature_columns].values.astype(np.float32)
        labels = unit_df["RUL"].values.astype(np.float32)
        cycles = unit_df["time_cycles"].values.astype(int)

        if len(unit_df) < seq_len:
            continue

        if last_only:
            i = len(unit_df) - seq_len
            X.append(feats[i : i + seq_len])
            y.append(labels[i + seq_len - 1])
            metadata.append(
                {
                    "unit_nr": int(unit_nr),
                    "end_cycle": int(cycles[i + seq_len - 1]),
                }
            )
        else:
            for i in range(len(unit_df) - seq_len + 1):
                X.append(feats[i : i + seq_len])
                y.append(labels[i + seq_len - 1])
                metadata.append(
                    {
                        "unit_nr": int(unit_nr),
                        "end_cycle": int(cycles[i + seq_len - 1]),
                    }
                )

    return np.array(X), np.array(y), metadata


def train_val_split_by_unit(
    df: pd.DataFrame,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    units = np.array(sorted(df["unit_nr"].unique()))
    rng.shuffle(units)

    if len(units) < 2:
        raise ValueError("Need at least 2 units to create train/validation split.")

    split_idx = int(len(units) * (1 - val_ratio))
    split_idx = min(max(1, split_idx), len(units) - 1)

    train_units = set(units[:split_idx])
    val_units = set(units[split_idx:])

    train_df = df[df["unit_nr"].isin(train_units)].copy()
    val_df = df[df["unit_nr"].isin(val_units)].copy()

    if train_df.empty or val_df.empty:
        raise ValueError("Train/validation split produced an empty split.")

    return train_df, val_df


def prepare_cmapss_datasets(
    data_dir: str,
    dataset: str = "FD001",
    seq_len: int = 50,
    val_ratio: float = 0.2,
    max_rul: int | None = 125,
    scaler_path: str = "artifacts/scaler.joblib",
    seed: int = 42,
    fit_scaler_on_train: bool = True,
) -> DatasetBundle:
    data_dir = Path(data_dir)

    train_path = data_dir / f"train_{dataset}.txt"
    test_path = data_dir / f"test_{dataset}.txt"
    rul_path = data_dir / f"RUL_{dataset}.txt"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing train file: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test file: {test_path}")
    if not rul_path.exists():
        raise FileNotFoundError(f"Missing RUL file: {rul_path}")

    train_raw = _read_cmapss_file(train_path)
    test_raw = _read_cmapss_file(test_path)

    train_df = add_rul_train(train_raw, max_rul=max_rul)
    test_df = add_rul_test(test_raw, rul_path, max_rul=max_rul)

    feature_columns = DEFAULT_FEATURE_COLUMNS.copy()

    train_df_split, val_df_split = train_val_split_by_unit(
        train_df,
        val_ratio=val_ratio,
        seed=seed,
    )

    if fit_scaler_on_train:
        scaler = fit_scaler(train_df_split, feature_columns, scaler_path)
    else:
        scaler = load_scaler(scaler_path)

    train_df_split = transform_with_scaler(train_df_split, feature_columns, scaler)
    val_df_split = transform_with_scaler(val_df_split, feature_columns, scaler)
    test_df = transform_with_scaler(test_df, feature_columns, scaler)

    X_train, y_train, md_train = make_sequences(
        train_df_split,
        feature_columns,
        seq_len,
        last_only=False,
    )
    X_val, y_val, md_val = make_sequences(
        val_df_split,
        feature_columns,
        seq_len,
        last_only=False,
    )
    X_test, y_test, md_test = make_sequences(
        test_df,
        feature_columns,
        seq_len,
        last_only=True,
    )

    return DatasetBundle(
        train_dataset=SequenceDataset(X_train, y_train, md_train),
        val_dataset=SequenceDataset(X_val, y_val, md_val),
        test_dataset=SequenceDataset(X_test, y_test, md_test),
        feature_columns=feature_columns,
        scaler_path=str(scaler_path),
    )