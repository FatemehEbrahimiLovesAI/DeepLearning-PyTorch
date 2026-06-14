# Linear Regression with PyTorch

A simple implementation of linear regression using PyTorch to predict salary based on years of experience.

## Overview

This project trains a single-layer linear model (`nn.Linear`) on a salary dataset. The full pipeline covers data loading, preprocessing, model training with mini-batch SGD, and evaluation on a held-out test set.

## Dataset

The dataset is a CSV file (`Salary Data.csv`) with at least two columns:

- `Experience Years` — input feature (X)
- `Salary` — target label (y)

The dataset is split 80/20 into training and test sets (40 total samples: 32 train, 8 test).

## Project Structure

```
.
├── main.ipynb              # Main notebook
├── Salary Data.csv         # Dataset
└── LinearRegressionModel   # Saved model (generated after training)
```

## Pipeline

### 1. Data Handling

- Load the CSV with pandas
- Extract `Experience Years` as X and `Salary` as y
- Split into train/test sets using `train_test_split` (80/20, `random_state=12`)
- Reshape vectors to 2D matrices `(-1, 1)`
- Standardize both X and y using `StandardScaler` (fit on train, transform on test)
- Convert NumPy arrays to `float32` PyTorch tensors
- Wrap in `TensorDataset` and `DataLoader` (train batch size: 5, test batch size: 4)

### 2. Model

A single `nn.Linear(1, 1)` layer, which learns one weight and one bias — equivalent to ordinary least squares linear regression.

### 3. Training

- Loss: `nn.MSELoss`
- Optimizer: `SGD` with `lr=0.001` and `momentum=0.9`
- Epochs: 100
- Per epoch, iterates over all mini-batches, computes predictions, calculates MSE loss, and performs a gradient update

After training, the model is saved to disk with `torch.save`.

### 4. Evaluation

The saved model is loaded with `torch.load` and evaluated on the test `DataLoader` by computing MSE loss per batch.

## Requirements

- Python 3.9+
- pandas
- numpy
- scikit-learn
- torch

Install dependencies:

```bash
pip install pandas numpy scikit-learn torch
```

## Usage

1. Place `Salary Data.csv` in the same directory as the notebook.
2. Open and run `main.ipynb` cell by cell.
3. The trained model will be saved as `LinearRegressionModel` in the working directory.

## Notes

- Both X and y are standardized independently. Predictions from the model are in the scaled space; inverse-transform with `y_scaler.inverse_transform()` to get original salary units.
- The model is saved as a full model object (not just state dict), so `weights_only=False` is required when loading with newer versions of PyTorch.
