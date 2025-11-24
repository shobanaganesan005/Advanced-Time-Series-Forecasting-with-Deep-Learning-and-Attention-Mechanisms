
# Advanced Time Series Forecasting with Deep Learning and Attention Mechanisms

## Dataset

The dataset is a synthetic, multivariate time series generated with `generate_data.py`.
It contains 1,500 time steps and three features. Each feature exhibits:

- A global upward linear trend.
- Multiple seasonalities (daily- and weekly-like cycles).
- Feature-specific noise and small phase shifts.

The data are saved as `data.csv` with a DateTime index and columns `feat_1`, `feat_2`, `feat_3`.

Before modelling, the series is split into train/validation/test segments in chronological
order (70/15/15). Features are standardized using the training-set mean and standard deviation.

## Model Architectures

### Baseline: LSTM Forecasting Model

The baseline is a stacked LSTM implemented in PyTorch:

- Input: past 72 time steps (`input_window`) of all features.
- Output: the next 24 time steps (`horizon`) for all features.
- Architecture:
  - 2 LSTM layers with hidden size 64.
  - The final hidden state is repeated across the forecast horizon and passed through
    a linear projection layer to produce multi-step forecasts.

This model represents a strong sequence model without explicit attention.

### Attention-based Seq2Seq Model

The advanced model is a sequence-to-sequence architecture with Bahdanau attention:

- Encoder:
  - Single-layer LSTM that processes the input window and returns all hidden states.
- Decoder:
  - LSTMCell that operates autoregressively over the prediction horizon.
  - At each step, Bahdanau attention computes a context vector as a weighted sum
    of encoder outputs.
  - The decoder receives the concatenation of the previous target value and the
    context vector, and outputs the forecast for the current step.

Teacher forcing is used during training: the ground-truth future values are provided
to the decoder as inputs.

## Training Procedure

Both models are trained with:

- Mean Squared Error (MSE) loss.
- Adam optimizer with learning rate 1e-3.
- Mini-batches of size 64.
- 10 training epochs by default (configurable via command line).

The training loop records per-epoch training and validation losses, which are stored
in `results.json` for later visualization. The code is organized in `train.py` to keep
data handling, model training, and evaluation in a clear structure.

## Evaluation Metrics

On the held-out test set, the following time-series metrics are computed:

- **RMSE (Root Mean Squared Error)**: measures overall prediction error magnitude.
- **MAPE (Mean Absolute Percentage Error)**: shows error as a percentage of the true values.
- **MASE (Mean Absolute Scaled Error)**: compares model performance against a
  naive seasonal forecast baseline.

These metrics are reported separately for the baseline LSTM and the attention-based model.

Example JSON structure in `results.json`:

```json
{
  "baseline": {
    "history": {"train_loss": [...], "val_loss": [...]},
    "metrics": {"RMSE": 0.45, "MAPE": 7.1, "MASE": 0.62}
  },
  "attention": {
    "history": {"train_loss": [...], "val_loss": [...]},
    "metrics": {"RMSE": 0.38, "MAPE": 5.9, "MASE": 0.51}
  }
}
```

(The numbers above are illustrative; they will depend on your actual run.)

## Interpretation and Comparison

In typical runs, the attention-based seq2seq model should achieve lower RMSE, MAPE,
and MASE than the baseline LSTM, reflecting:

- Better utilization of long-range dependencies by attending to the most relevant
  encoder time steps.
- Improved multi-step forecasting accuracy, especially near regime changes or
  high-variance regions.

The gap in metrics between the two models quantifies the performance gain obtained
from incorporating attention mechanisms on top of a recurrent architecture.

## How to Run

1. Generate the dataset:

   ```bash
   python generate_data.py
   ```

2. Train and evaluate both models:

   ```bash
   python train.py --epochs 10 --batch_size 64
   ```

3. Inspect `results.json` to view metrics and loss curves. You can easily plot
   the training and validation loss history using a short Python script or notebook.

This project demonstrates how modern attention-based architectures improve upon
standard LSTMs for multi-step time series forecasting on complex, seasonal data.
