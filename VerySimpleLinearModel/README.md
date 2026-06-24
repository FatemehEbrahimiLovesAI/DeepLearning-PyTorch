# Linear Regression Model with PyTorch

This notebook demonstrates a simple linear regression implementation using PyTorch's `nn.Module`.

## Overview

A custom linear network class is defined with configurable input and output dimensions. The model performs forward propagation to compute predictions based on the input features.

## Implementation Details

### Model Definition

The `linear_network` class inherits from `nn.Module` and consists of a single linear layer. The forward pass applies this layer to the input tensor.

```python
class linear_network(nn.Module):
    def __init__(self, in_features, out_features):
        super(linear_network, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        return self.linear(x)
```

### Data Preparation

Input data is structured as a 2D tensor with shape `[samples, features]`. In this case, 2 samples with 5 features each.

### Weight Configuration

Model weights and bias are manually assigned to specific values for demonstration purposes:

- Weights: `[5.0, 4.0, 9.0, 8.0, 2.0]`
- Bias: `2.0`

### Forward Propagation

The model computes predictions using the linear transformation:

```
y = W · x + b
```

## Results

For the given input samples:

- Sample 1: `1*5 + 2*4 + 3*9 + 4*8 + 5*2 + 2 = 84`
- Sample 2: `3*5 + 4*4 + 5*9 + 6*8 + 7*2 + 2 = 140`

The model produces the expected outputs, confirming correct implementation.
