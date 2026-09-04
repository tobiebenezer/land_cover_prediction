# Satellite Land Cover & NDVI Time-Series Forecasting

A deep learning benchmark comparing modern time-series transformers (**PatchTST**, **Temporal Fusion Transformers**) against recurrent architectures (**LSTM**, **GRU**, **SRNN**) for spatial-temporal satellite imagery and NDVI (Normalized Difference Vegetation Index) forecasting.

---

## Overview

Forecasting vegetation dynamics and land cover shifts from remote sensing feeds requires capturing both spatial patch relationships and temporal dependencies across seasons. This repository benchmarks multiple deep learning architectures on multi-temporal satellite data.

### Architectures Evaluated
- **PatchTST**: Segmenting multivariate time series into patches to preserve local semantics while drastically reducing attention complexity.
- **Temporal Fusion Transformer (TFT)**: Attention-based architecture designed for multi-horizon forecasting with interpretable feature dynamics.
- **Recurrent Baselines**: Standard LSTM, Gated Recurrent Units (GRU), and Spatial Recurrent Neural Networks (SRNN) used as comparative baselines.

---

## Repository Structure

```
├── data/                  # Satellite tiles and spatial imagery
├── data.py                # Dataset loaders and NDVI spatial transforms
├── process_patch_data.py  # Spatial patch extraction and preprocessing
├── train_tokenizer.py     # Tokenizer / codebook training for patch representations
├── train_TFT.py           # Temporal Fusion Transformer training pipeline
├── train_lstm.py          # LSTM baseline training
├── train_gru.py           # GRU baseline training
├── train_rnn.py           # Recurrent baseline training
├── evaluation.py          # Prediction loops, recursive multi-step forecasting, R2 & MSE
├── plots/                 # Visualizations of forecasts vs ground truth
├── metrics/               # Recorded validation metrics across models
└── requirements.txt       # Environment dependencies
```

---

## Prerequisites & Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/tobiebenezer/land_cover_prediction.git
   cd land_cover_prediction
   ```

2. **Create a virtual environment and install dependencies**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

---

## Workflow

### 1. Data Preparation
Preprocess raw satellite inputs into patched spatial tensors:
```bash
python process_patch_data.py
```

### 2. Training Models
Train individual architectures using their dedicated pipelines:

- **Temporal Fusion Transformer**:
  ```bash
  python train_TFT.py
  ```

- **LSTM / GRU Baselines**:
  ```bash
  python train_lstm.py
  python train_gru.py
  ```

### 3. Evaluation & Metrics
Run multi-step recursive forecasting and compute performance metrics ($R^2$, Mean Squared Error):
```bash
python evaluation.py
```

---

## Evaluation Metrics

Models are evaluated on out-of-sample temporal horizons:
- **$R^2$ Score**: Proportion of variance explained in vegetation index trajectories.
- **MSE / RMSE**: Reconstruction and prediction loss against ground truth satellite observations.
- **Inference Latency**: Batch prediction speeds across CUDA / MPS / CPU devices.

---

## License & Attribution

Developed by [Tobi Ebenezer](https://github.com/tobiebenezer). Feel free to use and adapt this code for academic and practical remote sensing research.
