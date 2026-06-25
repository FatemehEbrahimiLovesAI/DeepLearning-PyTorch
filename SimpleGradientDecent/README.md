# Linear Regression with PyTorch and StandardScaler

This notebook implements a simple linear regression model using PyTorch with manual gradient descent optimization and data standardization.

## Overview

The notebook demonstrates:
- Data generation and standardization using `StandardScaler`
- Manual implementation of linear regression with `torch` tensors
- Gradient descent optimization with custom learning rate
- Training loop with loss tracking

## Implementation Details

### Data Preparation

Input data `x` is generated as a range from -10 to 10 with step 0.1. The data is then standardized using `StandardScaler` to have zero mean and unit variance. The target `y` is defined as:

```python
y = 4 * x + 2
```

**Note:** Target values are computed after standardization to maintain the linear relationship with the scaled input features.

### Model Definition

The model follows a simple linear equation:

```
y_hat = w * x + b
```

where:
- `w` is the weight (slope)
- `b` is the bias (intercept)

Both parameters are initialized and trained using gradient descent.

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Learning Rate | 5e-2 |
| Epochs | 500 |
| Loss Function | Mean Squared Error (MSE) |
| Optimizer | Custom Gradient Descent |

### Training Loop

At each epoch:
1. Forward pass: compute predictions `y_hat = forward(x)`
2. Compute loss: MSE between `y` and `y_hat`
3. Backward pass: compute gradients with `loss.backward()`
4. Update weights: `w = w - lr * w.grad`
5. Zero gradients: `w.grad.zero_()` and `b.grad.zero_()`

## Results

The model successfully converges to the true parameters:

```
epoch: 145 | w: 3.9999990463256836 | b: 2.0 | loss: 9.094947017199453e-13
```

The final values closely match the expected parameters `w=4.0` and `b=2.0`.

## Key Takeaways

- **StandardScaler** significantly affects gradient magnitudes and requires careful learning rate selection
- Proper gradient zeroing with `zero_()` is essential to prevent gradient accumulation
- Target values should be aligned with the scaled feature space when generating synthetic data
- Learning rate `5e-2` proved effective for this standardized dataset

## Dependencies

- PyTorch
- scikit-learn (StandardScaler)
