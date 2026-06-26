# Stochastic Gradient Descent (SGD) Implementation in PyTorch

This notebook demonstrates the implementation of Stochastic Gradient Descent (SGD) for a simple linear regression problem using PyTorch. The implementation is shown both with and without using PyTorch's DataLoader.

## Overview

The goal is to learn the parameters `w` (weight) and `b` (bias) for a linear function `y = w * x + b` where the true underlying function is `y = 4x + 5`. The model is trained using Mean Squared Error (MSE) loss and optimized using SGD.

## Requirements

- PyTorch
- torch.utils.data.DataLoader
- torch.utils.data.Dataset

## Implementation Details

### Without DataLoader

The first implementation uses a simple nested loop structure without DataLoader:

- Iterates over epochs and samples manually using `zip(X, Y)`
- For each sample, computes forward pass, loss, and performs backpropagation
- Updates parameters `w` and `b` using gradient descent with learning rate 0.1

### With DataLoader

The second implementation uses PyTorch's DataLoader for batch processing:

1. Defines a custom `Data` class that inherits from `Dataset`
2. Implements `__init__`, `__getitem__`, and `__len__` methods
3. Creates a DataLoader instance with batch size 1
4. Training loop iterates over the DataLoader for each epoch

## Code Structure

- **Data Generation**: Creates synthetic data points `X` ranging from -3 to 3 with step 0.1
- **Model Parameters**: Initializes `w = -1` and `b = 3` with gradient tracking enabled
- **Forward Function**: Implements linear regression `y_hat = w * x + b`
- **Loss Function**: Uses Mean Squared Error (MSE)
- **Training Loop**: Performs parameter updates using SGD

## Results

Both implementations produce similar results, with the average loss converging to approximately 2.6781 after 4 epochs of training.

## Key Concepts Demonstrated

- **Stochastic Gradient Descent**: Updates parameters using individual samples rather than the entire dataset
- **Automatic Differentiation**: PyTorch's autograd handles gradient computation via `loss.backward()`
- **Parameter Updates**: Manual update of `w.data` and `b.data` using computed gradients
- **Gradient Zeroing**: Clearing gradients after each update using `zero_()`
- **DataLoader**: Efficient batch processing and data iteration

## Notes

- The learning rate is set to 0.1 for both implementations
- The results are identical for both implementations as batch size = 1
- Gradient accumulation is not used; gradients are zeroed after each update
- The loss values are stored for each step and averaged at the end
