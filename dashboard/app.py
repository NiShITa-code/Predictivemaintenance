from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")
st.title("Predictive Maintenance Experiment Dashboard")
st.caption("Compare RNN, GRU, LSTM, and Attention-LSTM runs on NASA CMAPSS")

artifact_dir = Path("artifacts")

if not artifact_dir.exists():
    st.warning("No artifacts directory found. Train a model first.")
    st.stop()

summary_files = sorted(artifact_dir.glob("*_summary.json"))
if not summary_files:
    st.warning("No training summaries found. Run training first.")
    st.stop()


def load_summaries(files):
    rows = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fp:
            summary = json.load(fp)
        summary["run_name"] = f.name.replace("_summary.json", "")
        rows.append(summary)
    return pd.DataFrame(rows)


df = load_summaries(summary_files)

# Sidebar filters
st.sidebar.header("Filters")

available_models = sorted(df["model"].unique()) if "model" in df.columns else []
selected_models = st.sidebar.multiselect(
    "Select models",
    available_models,
    default=available_models
)

available_datasets = sorted(df["dataset"].unique()) if "dataset" in df.columns else []
selected_datasets = st.sidebar.multiselect(
    "Select datasets",
    available_datasets,
    default=available_datasets
)

seq_lens = sorted(df["seq_len"].unique()) if "seq_len" in df.columns else []
selected_seq_lens = st.sidebar.multiselect(
    "Select sequence lengths",
    seq_lens,
    default=seq_lens
)

filtered_df = df[
    df["model"].isin(selected_models) &
    df["dataset"].isin(selected_datasets) &
    df["seq_len"].isin(selected_seq_lens)
].copy()

if filtered_df.empty:
    st.warning("No runs match the selected filters.")
    st.stop()

# Show best run
best_row = filtered_df.sort_values("test_rmse").iloc[0]

st.subheader("Best Run")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Best Model", str(best_row["model"]).upper())
c2.metric("Best Test RMSE", f"{best_row['test_rmse']:.2f}")
c3.metric("Best Test MAE", f"{best_row['test_mae']:.2f}")
c4.metric("Best Seq Len", str(best_row["seq_len"]))

st.markdown(
    f"""
**Interpretation:**  
The current best run is **{best_row['model'].upper()}** on **{best_row['dataset']}**
with **sequence length = {best_row['seq_len']}**.
This run achieved the lowest test RMSE among the selected experiments.
"""
)

# Summary table
st.subheader("Experiment Summary Table")
display_cols = [
    "run_name", "dataset", "model", "seq_len",
    "hidden_size", "epochs", "test_rmse", "test_mae",
    "test_loss", "train_time_seconds", "anomaly_rate"
]
display_cols = [c for c in display_cols if c in filtered_df.columns]
st.dataframe(filtered_df[display_cols].sort_values("test_rmse"), use_container_width=True)

# RMSE comparison bar chart
st.subheader("Model Comparison: Test RMSE")
fig_rmse, ax_rmse = plt.subplots(figsize=(10, 4))
plot_df = filtered_df.sort_values("test_rmse")
ax_rmse.bar(plot_df["run_name"], plot_df["test_rmse"])
ax_rmse.set_ylabel("Test RMSE")
ax_rmse.set_xlabel("Run")
ax_rmse.set_title("Lower is better")
ax_rmse.tick_params(axis="x", rotation=45)
plt.tight_layout()
st.pyplot(fig_rmse)
plt.close(fig_rmse)

# Sequence length sensitivity plot
if len(filtered_df["seq_len"].unique()) > 1:
    st.subheader("Sequence Length Sensitivity")
    fig_seq, ax_seq = plt.subplots(figsize=(10, 4))
    for model_name, group in filtered_df.groupby("model"):
        group = group.sort_values("seq_len")
        ax_seq.plot(group["seq_len"], group["test_rmse"], marker="o", label=model_name.upper())
    ax_seq.set_xlabel("Sequence Length")
    ax_seq.set_ylabel("Test RMSE")
    ax_seq.set_title("How performance changes with temporal context")
    ax_seq.legend()
    plt.tight_layout()
    st.pyplot(fig_seq)
    plt.close(fig_seq)

    st.markdown(
        """
**How to read this plot:**  
If RMSE improves as sequence length increases, the model is benefiting from longer-term information.
If RMSE stays flat or worsens, the model may be failing to use long-range dependencies effectively.
"""
    )

# Run detail viewer
st.subheader("Inspect a Specific Run")
run_options = filtered_df["run_name"].tolist()
selected_run = st.selectbox("Choose a run to inspect", run_options)

selected_summary = filtered_df[filtered_df["run_name"] == selected_run].iloc[0]

col1, col2, col3 = st.columns(3)
col1.metric("Test RMSE", f"{selected_summary['test_rmse']:.2f}")
col2.metric("Test MAE", f"{selected_summary['test_mae']:.2f}")
col3.metric("Anomaly Rate", f"{selected_summary['anomaly_rate']:.3f}")

with st.expander("Full Summary JSON"):
    st.json(selected_summary.to_dict())

stem = selected_run
y_true_path = artifact_dir / f"{stem}_y_true.npy"
y_pred_path = artifact_dir / f"{stem}_y_pred.npy"
attn_path = artifact_dir / f"{stem}_attention.npy"

if y_true_path.exists() and y_pred_path.exists():
    y_true = np.load(y_true_path)
    y_pred = np.load(y_pred_path)

    n = st.slider("Number of prediction samples to plot", min_value=50, max_value=min(500, len(y_true)), value=min(300, len(y_true)), step=25)

    fig_pred, ax_pred = plt.subplots(figsize=(12, 4))
    ax_pred.plot(y_true[:n], label="Actual")
    ax_pred.plot(y_pred[:n], label="Predicted")
    ax_pred.set_title(f"Predicted vs Actual RUL — {selected_run}")
    ax_pred.set_xlabel("Sample")
    ax_pred.set_ylabel("RUL")
    ax_pred.legend()
    plt.tight_layout()
    st.pyplot(fig_pred)
    plt.close(fig_pred)

    abs_err = np.abs(y_true - y_pred)
    threshold = selected_summary.get("anomaly_threshold_abs_error_p95", float(np.quantile(abs_err, 0.95)))

    fig_err, ax_err = plt.subplots(figsize=(12, 4))
    ax_err.plot(abs_err[:n], label="Absolute Error")
    ax_err.axhline(threshold, linestyle="--", label="Anomaly Threshold (p95)")
    ax_err.set_title("Prediction Error / Anomaly View")
    ax_err.set_xlabel("Sample")
    ax_err.set_ylabel("Absolute Error")
    ax_err.legend()
    plt.tight_layout()
    st.pyplot(fig_err)
    plt.close(fig_err)

    st.markdown(
        """
**Interpretation:**  
Large spikes in absolute error may indicate samples where the model struggles.  
In an industrial setting, these could be treated as anomalous engine behavior or unexpected degradation patterns.
"""
    )

# Attention heatmap if present
if attn_path.exists():
    attn = np.load(attn_path)
    rows = min(20, attn.shape[0])

    st.subheader("Attention Heatmap")
    fig_attn, ax_attn = plt.subplots(figsize=(12, 5))
    im = ax_attn.imshow(attn[:rows], aspect="auto", cmap="viridis")
    ax_attn.set_title(f"Temporal Attention Heatmap — {selected_run}")
    ax_attn.set_xlabel("Time Step in Sequence")
    ax_attn.set_ylabel("Sample")
    fig_attn.colorbar(im, ax=ax_attn)
    plt.tight_layout()
    st.pyplot(fig_attn)
    plt.close(fig_attn)

    st.markdown(
        """
**Interpretation:**  
Brighter regions indicate timesteps the model considered more important for predicting remaining useful life.
This helps explain *where* the model is focusing in the degradation sequence.
"""
    )

#
