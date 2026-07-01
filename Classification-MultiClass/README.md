# Iris Classification with PyTorch

This repository contains a complete implementation of a multiclass classification model for the Iris dataset using PyTorch. The model uses a simple linear neural network with cross-entropy loss and SGD optimizer.

## Overview

The project demonstrates a complete machine learning pipeline including:
- Data loading and preprocessing
- Train/validation/test split
- Feature scaling
- PyTorch DataLoader implementation
- Model training with validation
- Model persistence and evaluation

## Requirements

```
torch
pandas
numpy
scikit-learn
```

## Dataset

The Iris dataset is used, which contains 150 samples with 4 features (sepal length, sepal width, petal length, petal width) and 3 target classes (setosa, versicolor, virginica).

## Project Structure

```
├── main.ipynb          # Main Jupyter notebook with complete implementation
├── best_model.pt       # Saved model with best validation performance
└── README.md           # This file
```

## Implementation Details

### Data Preprocessing

1. **Data Splitting**: 
   - Train/Test split: 80/20 with stratification
   - Train/Validation split: 90/10 from training data

2. **Feature Scaling**:
   - StandardScaler is used to normalize features
   - Fitted on training data and transformed on validation and test sets

3. **Data Conversion**:
   - Features converted to torch.float32
   - Labels converted to torch.int64

### DataLoaders

- Train loader: batch_size=10, shuffled
- Validation loader: batch_size=5, shuffled
- Test loader: batch_size=200, shuffled

### Model Architecture

The model uses a single linear layer:
```python
nn.Linear(in_features=4, out_features=3)
```

### Training Configuration

- Loss Function: CrossEntropyLoss
- Optimizer: SGD with momentum=0.9
- Learning Rate: 0.1
- Epochs: 100

### Training Loop

The training loop includes:
1. Forward pass through the model
2. Loss calculation
3. Backward pass
4. Optimizer step
5. Gradient zeroing
6. Training metrics calculation (loss and accuracy)
7. Validation metrics calculation
8. Model checkpointing based on best validation loss

### Model Saving

The model is saved whenever a new best validation loss is achieved:
```python
torch.save(model, 'best_model.pt')
```

### Evaluation

Final evaluation on the test set loads the best saved model and computes both loss and accuracy.

## Key Features

- **Proper Validation**: Separate validation set used for monitoring model performance
- **Model Checkpointing**: Best model saved based on validation loss
- **Metrics Tracking**: Loss and accuracy tracked for both training and validation
- **Data Scaling**: Proper feature scaling using StandardScaler
- **Reproducibility**: Random seed fixed for consistent splits

## Results

The model achieves near-perfect performance on the Iris dataset:
- Training Accuracy: ~97%
- Validation Accuracy: 100%
- Test Accuracy: 100%

## Usage

1. Run all cells in the Jupyter notebook sequentially
2. The model will train for 100 epochs with progress printing
3. Best model will be saved as 'best_model.pt'
4. Final test evaluation will be displayed

## Notes

- The model uses CrossEntropyLoss which expects raw logits (no softmax layer in the model)
- The validation set size is approximately 10% of training data
- Stratification ensures balanced class distribution across splits
