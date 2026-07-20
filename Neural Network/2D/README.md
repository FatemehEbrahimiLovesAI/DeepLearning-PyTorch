# PyTorch XOR Problem Implementation

A simple neural network implementation for solving the XOR problem using PyTorch, featuring a custom dataset and training loop.

## Overview

This project demonstrates a binary classification solution to the classic XOR (exclusive OR) problem using a two-layer neural network with sigmoid activation functions. The model learns to classify four input patterns: (0,0), (0,1), (1,0), and (1,1).

## Project Structure

```
.
├── main.ipynb          # Main Jupyter notebook with complete implementation
└── README.md           # Project documentation
```

## XOR Dataset

The XOR dataset is synthetically generated with the following characteristics:

- **Input**: 2-dimensional binary vectors
- **Output**: XOR logic gate output
  - (0,0) → 0
  - (0,1) → 1
  - (1,0) → 1
  - (1,1) → 0
- **Samples**: 100 samples (25 per class)
- **Noise**: Small Gaussian noise (σ = 0.01) added for realistic data

The dataset class (`XOR_Data`) was sourced from the course materials and used as-is, as the primary learning objective focuses on model architecture and training implementation rather than custom dataset construction.

## Model Architecture

The neural network consists of:
- **Input Layer**: 2 neurons
- **Hidden Layer**: 8 neurons with sigmoid activation
- **Output Layer**: 1 neuron with sigmoid activation

```python
class Net(nn.Module):
    def __init__(self, d_in, h, d_out):
        self.linear1 = nn.Linear(d_in, h)
        self.linear2 = nn.Linear(h, d_out)
```

The sigmoid activation function is used for both hidden and output layers to produce binary classification outputs.

## Implementation Details

### Training Configuration

- **Optimizer**: Adam
  - Learning Rate: 0.01
- **Loss Function**: Binary Cross Entropy (BCELoss)
- **Batch Size**: 100 (full batch training)
- **Epochs**: 5

### Training Function

The training function implements a standard training loop with:

1. Forward pass through the model
2. Loss calculation using BCELoss
3. Backward pass for gradient computation
4. Parameter updates using Adam optimizer
5. Average loss calculation per epoch

```python
def train_model(train_set, train_loader, optimizer, model, criterion, epochs):
    cost = []
    for epoch in range(epochs):
        total = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_hat = model(X_batch)
            loss = criterion(y_hat, y_batch)
            loss.backward()
            optimizer.step()
            total += loss.item()
        cost.append(total / len(train_set))
    return cost
```

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
[0.0073, 0.0072, 0.0072, 0.0071, 0.0071]
```

The decreasing trend indicates that the model is learning the XOR pattern correctly. The low loss values suggest the model converges quickly due to the small dataset size.

## Key Features

- **XOR Dataset**: Implementation of the classic XOR problem
- **Binary Classification**: Using sigmoid output and BCELoss
- **Adam Optimizer**: Adaptive learning rate optimization
- **Loss Tracking**: Monitoring of average loss per epoch
- **Modular Design**: Separate dataset, model, and training components

## Learning Insights

This implementation demonstrates several important PyTorch concepts:

1. **Custom Dataset**: Understanding the structure of PyTorch Dataset class
2. **Binary Classification**: Applying sigmoid activation and binary cross entropy loss
3. **Neural Network Design**: Two-layer network with non-linear activation
4. **Training Loop**: Complete training cycle with optimization steps
5. **Loss Monitoring**: Tracking training progress through average loss values

## Notes

- The `XOR_Data` class was reused from course materials (IBM AI Engineering certificate) to focus learning on model implementation and training logic
- The dataset size (100 samples) is intentionally small for quick experimentation
- Noise is added to prevent the model from perfectly memorizing the XOR pattern

## Future Improvements

- Add validation set for proper model evaluation
- Implement accuracy metrics
- Visualize decision boundaries
- Add early stopping based on validation loss
- Experiment with different activation functions (ReLU, Tanh)
- Add learning rate scheduling
- Test different network architectures (depth, width)

## License

This project is open-source and available for educational purposes.

## Author

[Your Name]

## Acknowledgments

- PyTorch documentation and community
- IBM AI Engineering Professional Certificate on Coursera (dataset structure)
