# 2D Linear Regression with PyTorch

This project demonstrates a simple 2D linear regression implementation using PyTorch with synthetic data.

## Overview

The code implements a linear regression model that learns a mapping from 2D input features to a scalar output. The dataset is generated synthetically with known parameters and small Gaussian noise.

## Project Structure

### Dependencies
- Python 3.9+
- PyTorch
- torch.utils.data (Dataset, DataLoader)

### Components

#### 1. Custom Dataset Class (Data2D)
Generates synthetic 2D data with the following characteristics:
- 20 samples with 2 features each
- Features are normalized to range [-1, 1] using linspace
- True underlying function: y = X * w + b
- Parameters: w = [0.1, 2.0]^T, b = 2.0
- Added Gaussian noise: 0.1 * randn()

```python
class Data2D(Dataset):
    def __init__(self):
        self.X = torch.zeros((20, 2))
        self.X[:, 0] = torch.linspace(-1., 1., 20)
        self.X[:, 1] = torch.linspace(-1., 1., 20)
        self.w = torch.tensor([[0.1], [2.]])
        self.b = 2.
        self.f = torch.mm(self.X, self.w) + self.b
        self.y = self.f + 0.1 * torch.randn((self.X.shape[0], 1))
```

#### 2. Linear Regression Model (LR)
A simple linear layer with configurable input and output dimensions.

```python
class LR(nn.Module):
    def __init__(self, in_features, out_features):
        super(LR, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, X):
        return self.linear(X)
```

## Training Configuration

- Batch size: 2
- Learning rate: 0.01
- Optimizer: SGD (Stochastic Gradient Descent)
- Loss function: MSELoss (Mean Squared Error)
- Epochs: 10

## Training Process

For each epoch:
1. Iterate through batches from the training DataLoader
2. Forward pass: compute predictions
3. Calculate MSE loss between predictions and targets
4. Zero gradients from previous iteration
5. Backward pass: compute gradients
6. Update model parameters using SGD optimizer
7. Print loss for each batch

## Data Characteristics

The dataset has the following properties:
- **Input range**: Features are normalized between -1 and 1
- **Target range**: Approximately 0.9 to 2.9 (with noise)
- **Noise level**: Standard deviation of 0.1
- **Sample count**: 20 training samples

## Results

The training shows the loss decreasing from approximately 2.0 to 0.9 over 100 batch iterations (10 epochs with 10 batches each). The model successfully converges to a reasonable solution, with the loss stabilizing around 0.9.

Sample loss progression:
- First batch: ~2.0
- After 10 batches: ~0.6-0.9
- After 50 batches: ~0.3-0.5
- After 100 batches: ~0.9 (oscillating due to noise)

## Key Features

1. **Normalized Input Data**: Features are scaled to [-1, 1] range, which helps with training stability
2. **Reproducibility**: Uses manual seed (torch.manual_seed(1)) for consistent results
3. **Custom Dataset**: Implements PyTorch Dataset interface for seamless integration with DataLoader
4. **Minimal Architecture**: Simple linear model suitable for demonstration purposes

## Usage

Run the Jupyter notebook cells in sequential order:

1. Import required libraries
2. Set random seed for reproducibility
3. Create dataset and DataLoader
4. Initialize model, loss function, and optimizer
5. Train the model
6. Monitor loss during training

## Output

The script prints the loss value for each batch during training, allowing real-time monitoring of the model's convergence.

Example output format:
```
tensor(1.9932, grad_fn=<MseLossBackward0>)
tensor(0.5856, grad_fn=<MseLossBackward0>)
...
```

## Limitations and Future Improvements

1. **Validation Set**: Currently no validation set is used; adding one would help detect overfitting
2. **Learning Rate Scheduling**: Implementing a learning rate scheduler could improve convergence
3. **Early Stopping**: Adding early stopping based on validation loss would prevent overfitting
4. **Evaluation Metrics**: Additional metrics like R-squared could provide more insight
5. **Visualization**: Adding plots for loss curves and predictions vs actual values

## Notes

- The two features are currently identical (both using linspace from -1 to 1), which creates multicollinearity. For real applications, features should be distinct.
- Batch size of 2 is small but works for demonstration purposes with 20 samples.
- SGD optimizer with momentum could be added for potentially faster convergence.
