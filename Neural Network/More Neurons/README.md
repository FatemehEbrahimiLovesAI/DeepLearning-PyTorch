# PyTorch Binary Classification Model

A simple binary classification implementation using PyTorch with a custom dataset and a two-layer neural network.

## Overview

This project demonstrates a binary classification task using a neural network with sigmoid activation functions. The model is trained on synthetic data generated with two positive regions and one negative region.

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

## Dataset

The synthetic dataset is generated as follows:

- **Input**: 100 points linearly spaced from -20 to 20
- **Labels**: Binary classification based on input ranges:
  - `y = 1` for inputs in ranges (-10, -5) and (5, 10)
  - `y = 0` for all other inputs

```
Data Distribution:
-20  ------  -10  ---  -5  ---  5  ---  10  ------  20
              |  y=1  |        |  y=1  |
              |_______|        |_______|
```

## Model Architecture

The neural network consists of:
- **Input Layer**: 1 neuron
- **Hidden Layer**: 12 neurons with sigmoid activation
- **Output Layer**: 1 neuron with sigmoid activation

```python
class Net(nn.Module):
    def __init__(self, d_in, h, d_out):
        self.linear1 = nn.Linear(d_in, h)
        self.linear2 = nn.Linear(h, d_out)
```

## Implementation Details

### Data Loading

Custom `Dataset` class implementation:

```python
class Data(Dataset):
    def __init__(self):
        self.X = torch.linspace(-20, 20, 100).view(-1, 1)
        self.y = torch.zeros(self.X.shape[0])
        self.y[(self.X[:,0] > -10) & (self.X[:,0] < -5)] = 1
        self.y[(self.X[:,0] > 5) & (self.X[:,0] < 10)] = 1
```

### Training Configuration

- **Optimizer**: Adam
  - Learning Rate: 0.01
- **Loss Function**: Binary Cross Entropy (BCELoss)
- **Batch Size**: 100
- **Epochs**: 5

### Training Loop

The training process includes:
1. Forward pass through the model
2. Loss calculation using BCELoss
3. Backward pass for gradient computation
4. Parameter updates using Adam optimizer
5. Loss tracking for each epoch

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

The model achieves decreasing training loss over 5 epochs:

```
[0.6045, 0.5999, 0.5964, 0.5936, 0.5914]
```

The decreasing trend indicates that the model is learning the binary classification pattern.

## Key Features

- **Custom Dataset**: Implementation of PyTorch's Dataset class
- **Binary Classification**: Binary classification using sigmoid output and BCELoss
- **DataLoader**: Efficient batch processing
- **Loss Tracking**: Monitoring of loss during training
- **Adam Optimizer**: Adaptive learning rate optimization

## Learning Insights

This implementation demonstrates several important PyTorch concepts:

1. **Custom Data Generation**: Creating synthetic datasets for classification tasks
2. **Binary Classification**: Using sigmoid activation and binary cross entropy loss
3. **Neural Network Design**: Two-layer network with non-linear activation
4. **Training Loop**: Complete training cycle with optimization steps
5. **Loss Monitoring**: Tracking training progress through loss values

## Future Improvements

- Add validation and test sets
- Implement early stopping
- Add learning rate scheduling
- Visualize decision boundaries
- Experiment with different activation functions
- Increase number of epochs for better convergence

## License

This project is open-source and available for educational purposes.

## Author

Fatemeh Ebrahimi

## Acknowledgments

- PyTorch documentation and community
- Course materials and instructors
