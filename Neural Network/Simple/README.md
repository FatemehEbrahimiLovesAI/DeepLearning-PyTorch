# PyTorch Regression Model

A simple neural network implementation for regression tasks using PyTorch, featuring custom dataset generation and model training with validation.

## Overview

This project demonstrates a basic regression model using a two-layer neural network with sigmoid activation functions. The model is trained on synthetic data generated from a linear function with added noise.

## Project Structure

```
.
├── main.ipynb          # Main Jupyter notebook with complete implementation
└── README.md           # Project documentation
```

## Requirements

- Python 3.9+
- PyTorch
- Jupyter Notebook

## Model Architecture

The neural network consists of:
- **Input Layer**: 1 neuron
- **Hidden Layer**: 2 neurons with sigmoid activation
- **Output Layer**: 1 neuron with sigmoid activation

```python
class Net(nn.Module):
    def __init__(self, D_in, H, D_out):
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)
```

## Data Generation

The dataset is synthetically generated using the following process:

- **Input**: Random values from a normal distribution
- **True Function**: `f(x) = 2x - 3`
- **Observations**: `y = f(x) + noise` where noise ~ N(0, 0.01)

The dataset is split into:
- **Training Set**: 1000 samples
- **Validation Set**: 1000 samples

## Implementation Details

### Data Loading

Custom `Dataset` class implementation for handling the synthetic data:

```python
class Data(Dataset):
    def __init__(self, train=True):
        self.X = torch.randn(1000).view(-1, 1)
        self.f = self.X * 2 - 3
        self.y = self.f + 0.1 * torch.randn(self.X.size())
```

### Training Configuration

- **Optimizer**: SGD with momentum
  - Learning Rate: 0.1
  - Momentum: 0.9
- **Loss Function**: Mean Squared Error (MSE)
- **Batch Size**: 100
- **Epochs**: 100

### Training Loop

The training process includes:
1. Forward pass through the model
2. Loss calculation using MSE
3. Backward pass for gradient computation
4. Parameter updates using SGD optimizer
5. Loss tracking for both training and validation sets

## Usage

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install torch jupyter
   ```
3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook main.ipynb
   ```
4. Run all cells sequentially

## Results

The model achieves the following average losses:
- **Training Loss**: ~0.4966
- **Validation Loss**: ~0.4981

These similar values indicate that the model is not overfitting and generalizes well to unseen data.

## Key Features

- **Custom Dataset**: Implementation of PyTorch's Dataset class
- **DataLoader**: Efficient batch processing
- **Train/Validation Split**: Separate datasets for training and evaluation
- **Loss Tracking**: Monitoring of MSE loss during training
- **Model Evaluation**: Validation after each epoch using `model.eval()` and `torch.no_grad()`

## Learning Insights

This implementation demonstrates several important PyTorch concepts:

1. **Neural Network Definition**: Creating custom models by subclassing `nn.Module`
2. **Data Pipeline**: Using `Dataset` and `DataLoader` for efficient data handling
3. **Training Loop**: Complete training cycle with forward/backward passes
4. **Model Evaluation**: Proper validation with gradient tracking disabled
5. **Training Modes**: Using `model.train()` and `model.eval()` appropriately

## Future Improvements

- Add dropout layers for regularization
- Implement early stopping
- Add learning rate scheduling
- Include visualization of loss curves
- Experiment with different activation functions
- Try alternative optimizers (Adam, RMSprop)

## License

This project is open-source and available for educational purposes.

## Author

Fatemeh Ebrahimi

## Acknowledgments

- PyTorch documentation and community
- Course materials and instructors
