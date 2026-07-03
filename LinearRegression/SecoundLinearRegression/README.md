# Linear Regression with PyTorch

This project implements a simple linear regression model using PyTorch to demonstrate training with different learning rates and handling of outliers in the dataset.

## Overview

The code trains a linear regression model on synthetic data with the following characteristics:
- Input range: -3 to 3 with step size 0.1 (60 samples)
- True function: y = -3x + 5
- Added Gaussian noise: 0.1 * randn()
- Training set contains outliers: 5 samples set to 20 (indices 50-55)
- Validation set contains clean data without outliers

## Dependencies

- Python 3.9+
- PyTorch
- torch.utils.data (Dataset, DataLoader)

## Code Structure

### Data Class
Custom Dataset class that generates synthetic data with optional outliers for training.

```python
class Data(Dataset):
    def __init__(self, train=True)
```

### Model Architecture
Simple linear regression model with one input and one output feature.

```python
class LinearRegression(nn.Module)
```

### Training Configuration
- Number of epochs: 100
- Learning rates tested: [0.0001, 0.001, 0.01, 0.1, 1.0]
- Optimizer: SGD with momentum 0.9
- Loss function: MSELoss
- Batch size: 10

## Training Process

For each learning rate:
1. Initialize a new model
2. Train for 100 epochs
3. Record average training and validation loss per epoch
4. Calculate overall average loss across all epochs
5. Store trained model

The training loop:
- Forward pass through the model
- Calculate MSE loss
- Backward pass with gradient computation
- Update weights using SGD optimizer
- Zero gradients for next iteration
- Evaluate on validation set without gradient tracking

## Results

The following training and validation losses were observed:

| Learning Rate | Train Loss | Validation Loss |
|---------------|------------|-----------------|
| 0.0001        | 46.44      | 13.56           |
| 0.001         | 33.01      | 9.33            |
| 0.01          | 31.64      | 9.12            |
| 0.1           | 61.18      | 36.55           |
| 1.0           | NaN        | NaN             |

## Observations

1. **Learning Rate 0.01**: Best performance among tested rates with lowest validation loss
2. **Learning Rate 1.0**: Causes numerical instability (NaN losses) due to gradient explosion
3. **Outlier Impact**: Training loss is consistently higher than validation loss due to outliers in training data
4. **Suboptimal Performance**: Loss values remain relatively high (30-60) indicating the model struggles with outliers
5. **Overfitting Absence**: Similar training and validation losses suggest no overfitting

## Limitations

1. Outliers in training data significantly affect model performance
2. 100 epochs may be insufficient for convergence with SGD optimizer
3. High learning rates (0.1, 1.0) lead to unstable training
4. Simple linear model cannot capture the outlier patterns effectively

## Recommendations for Improvement

1. Increase number of epochs to 500-1000 for better convergence
2. Replace SGD with Adam optimizer for adaptive learning rates
3. Use learning rates in range [0.0001, 0.01] and remove higher values
4. Consider outlier removal or robust loss functions (Huber loss)
5. Implement learning rate scheduling for better convergence
6. Add weight decay regularization to improve generalization

## Usage

Execute the Jupyter notebook cells in order to:
1. Import required libraries
2. Define data generation class
3. Create training and validation datasets
4. Set up data loaders
5. Define model architecture
6. Configure training parameters
7. Train models with different learning rates
8. Display results

## Output

The script prints training and validation losses for each learning rate, allowing comparison of different optimization settings.
