# Predictive Maintenance with RNN, GRU, LSTM, LSTM+Attention, and Transformer

A PyTorch project for comparing recurrent models on Remaining Useful Life (RUL) prediction using the NASA CMAPSS turbofan engine dataset.

## What this repo does
- Loads NASA CMAPSS text files (FD001–FD004 format)
- Builds sliding-window sequences from multivariate sensor streams
- Trains and evaluates:
  - Vanilla RNN
  - GRU
  - LSTM
  - LSTM + Temporal Attention
  - Transformer Encoder
- Reports RMSE and MAE
- Saves plots and artifacts
- Includes a Streamlit dashboard for interactive inspection

## Dataset
Download the CMAPSS data files and place them in:

`data/CMAPSSData/`

Expected files:
- `train_FD001.txt`
- `test_FD001.txt`
- `RUL_FD001.txt`

You can repeat the same pattern for FD002, FD003, FD004.

## Install
```bash
pip install -r requirements.txt
```

## Train
```bash
python -m src.train --data_dir data/CMAPSSData --dataset FD001 --model rnn
python -m src.train --data_dir data/CMAPSSData --dataset FD001 --model gru
python -m src.train --data_dir data/CMAPSSData --dataset FD001 --model lstm
python -m src.train --data_dir data/CMAPSSData --dataset FD001 --model attention
python -m src.train --data_dir data/CMAPSSData --dataset FD001 --model transformer
python -m src.train --data_dir data/CMAPSSData --dataset FD001 --model gru --early_stopping_patience 5 --lr_scheduler_patience 2
python -m src.train --data_dir data/CMAPSSData --dataset FD001 --model transformer --transformer_heads 4 --transformer_ff_dim 128
```

Artifacts are saved with the sequence length in the filename, for example:
`artifacts/FD001_lstm_seq50_best.pt`
`artifacts/FD001_lstm_seq50_summary.json`

## Run a sweep
```bash
python -m src.experiments --dry_run
python -m src.experiments --skip_existing
python -m src.experiments --datasets FD001 FD002 --models gru lstm --seq_lens 30 50 80 --epochs 30
python -m src.experiments --datasets FD001 FD002 --models gru lstm --seq_lens 30 50 80 --epochs 30 --early_stopping_patience 5
```

## Evaluate
```bash
python -m src.evaluate --data_dir data/CMAPSSData --model_path artifacts/FD001_lstm_seq50_best.pt
```

## Run dashboard
```bash
streamlit run dashboard/app.py
```

## Recommended experiment plan
Use the same settings for all models:
- same window size
- same hidden size
- same batch size
- same train/validation split

Then compare:
- RMSE
- MAE
- training time
- prediction curves
- attention heatmaps for the attention model

## Repo structure
```text
rnn-gru-lstm-predictive-maintenance/
├── artifacts/
├── configs/
├── dashboard/
│   └── app.py
├── data/
│   └── CMAPSSData/
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── requirements.txt
└── README.md
```

## Extended Project Narrative

Predictive maintenance is one of the most valuable applications of machine learning in industrial systems. Instead of waiting for machines to fail, companies can predict failures before they happen and schedule maintenance proactively.

In this project we build a deep learning system that predicts Remaining Useful Life (RUL) from sensor time-series data, comparing:
- RNN
- GRU
- LSTM
- Attention-based LSTM
- Transformer encoder

### Problem Statement

Industrial machines degrade gradually over time. Sensors capture this degradation, and the model must learn the relationship between sensor patterns and machine health.

Given a sequence of sensor readings, the model predicts the remaining cycles before failure (RUL).

### Dataset

We use NASA CMAPSS datasets (FD001–FD004). Each row is one cycle for one engine, including operating settings and 21 sensors.

### Pipeline

1. Read raw files
2. Compute RUL (train: final_cycle - cycle, test: with RUL labels)
3. Scale features
4. Create sliding sequences
5. Train recurrent models
6. Evaluate RMSE/MAE and anomaly thresholding

### Results and Insights

Example model performance on FD001:
- RNN: higher RMSE
- GRU: improved
- LSTM: best in many cases
- Attention LSTM: interpretable attention weights

Sequence length experiments show LSTM benefits from longer history.

### How to run

Install:
`pip install -r requirements.txt`

Train:
`python -m src.train --data_dir data/CMAPSSData --dataset FD001 --model rnn`
`python -m src.train --data_dir data/CMAPSSData --dataset FD001 --model gru`
`python -m src.train --data_dir data/CMAPSSData --dataset FD001 --model lstm`
`python -m src.train --data_dir data/CMAPSSData --dataset FD001 --model attention`
`python -m src.train --data_dir data/CMAPSSData --dataset FD001 --model transformer`

Sweep:
`python -m src.experiments --skip_existing`

Evaluate:
`python -m src.evaluate --data_dir data/CMAPSSData --model_path artifacts/FD001_lstm_seq50_best.pt`

Dashboard:
`streamlit run dashboard/app.py`

### Takeaways

- Temporal models are critical.
- Gated architectures help significantly.
- Longer windows help LSTM.
- Anomaly detection from prediction error is practical.

---
