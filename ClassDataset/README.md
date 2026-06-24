# PyTorch Custom Dataset Tutorial

## Overview

This Jupyter notebook demonstrates how to create a custom dataset class in PyTorch by inheriting from `torch.utils.data.Dataset`. It also shows how to apply custom transformations to the data using callable classes and compose them together.

## File Structure

### Required Imports
```python
import torch
from torch.utils.data import Dataset
from torchvision import transforms
```

### Class Definitions

#### toy_set (Custom Dataset)

This class inherits from PyTorch's `Dataset` and creates a simple synthetic dataset for educational purposes.

**Parameters:**
- `length` (int): Number of samples (default: 100)
- `transform` (callable, optional): Transformation to apply to samples

**Data Characteristics:**
- Features (`x`): Tensor of shape `(length, 3)` filled with value 2
- Labels (`y`): Tensor of shape `(length,)` filled with value 1

**Methods:**

| Method | Description |
|--------|-------------|
| `__init__(self, length=100, transform=None)` | Initializes the dataset with synthetic data |
| `__getitem__(self, index)` | Returns the sample at the specified index with transformations applied if any |
| `__len__(self)` | Returns the total number of samples |

#### mult (Transformation Class)

Multiplies all values in a sample by a constant factor.

**Parameters:**
- `muli` (int/float): Multiplication factor (default: 2)

**Behavior:**
- Applies multiplication to both features (`x`) and labels (`y`)

#### add_muli (Transformation Class)

Performs addition on features and multiplication on labels simultaneously.

**Parameters:**
- `add` (int/float): Value to add to features (default: 3)
- `muli` (int/float): Value to multiply labels by (default: 4)

**Behavior:**
- Features: `x = x + add`
- Labels: `y = y * muli`

### Transformation Pipeline

```python
data_transform = transforms.Compose([add_muli(), mult()])
```

**Order of execution:**
1. `add_muli()`: Adds 3 to features, multiplies labels by 4
2. `mult()`: Multiplies both features and labels by 2

### Dataset Creation

```python
dataset_ = toy_set(length=10, transform=data_transform)
```

Creates a dataset with 10 samples, applying the composed transformation pipeline to each sample.

## Mathematical Walkthrough

Given the initial data:
- Features: `[2, 2, 2]`
- Label: `1`

**Step 1 - add_muli (add=3, muli=4):**
- Features: `[2+3, 2+3, 2+3] = [5, 5, 5]`
- Label: `1 * 4 = 4`

**Step 2 - mult (muli=2):**
- Features: `[5*2, 5*2, 5*2] = [10, 10, 10]`
- Label: `4 * 2 = 8`

**Final Output:**
- All 10 samples have features `[10, 10, 10]` and label `8.0`

## Output

When the code is executed, the following output is displayed for each of the 10 samples:

```
sample [index]
	X : tensor([10., 10., 10.])
	y : 8.0
```

All samples are identical because the synthetic data starts with uniform values and the transformations are deterministic.

## Key Concepts Learned

### 1. Custom Dataset Implementation
- Inheriting from `torch.utils.data.Dataset`
- Implementing required methods: `__init__`, `__getitem__`, `__len__`
- Storing and using transformations within the dataset class

### 2. Custom Transformations
- Creating callable classes with `__call__` method
- Applying transformations to samples
- Returning modified samples in the expected format (tuple)

### 3. Composing Transformations
- Using `transforms.Compose` to chain multiple transformations
- Understanding the order of execution in a transformation pipeline
- Passing the composed transformation to the dataset

### 4. Data Augmentation Concepts
- Adding noise or variations to training data
- Scaling and shifting data values
- Preparing data for machine learning models

## Practical Applications

This structure can be extended for:

| Use Case | Description |
|----------|-------------|
| Image Classification | Loading images from disk, applying resizing and normalization |
| Text Processing | Tokenization, padding, and embedding |
| Time Series Analysis | Windowing, scaling, and feature engineering |
| Custom Data Formats | Loading CSV, JSON, or proprietary file formats |
| Data Augmentation | Rotation, flipping, cropping, and color jittering |

## Common Pitfalls and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| `NameError: name 'transform' is not defined` | Using `transform` instead of `self.transform` in `__getitem__` | Always use `self.transform` when accessing the stored transformation |
| Transformations not applied | Missing `if self.transform:` check in `__getitem__` | Check if transformation exists before applying |
| Unexpected output values | Incorrect order in `Compose` or wrong parameter values | Review the transformation order and parameter defaults |
| Dimension mismatch | Inconsistent tensor shapes | Ensure `x` and `y` have appropriate shapes and data types |

## Requirements

- Python 3.6+
- PyTorch
- torchvision (for `transforms.Compose`)

## Usage Example

```python
# Import required libraries
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Define transformations
transform = transforms.Compose([
    add_muli(add=3, muli=4),
    mult(muli=2)
])

# Create dataset
dataset = toy_set(length=100, transform=transform)

# Create dataloader for batch processing
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Iterate through batches
for batch_x, batch_y in dataloader:
    # Train your model here
    print(f"Batch X shape: {batch_x.shape}")
    print(f"Batch Y shape: {batch_y.shape}")
    break
```

## Summary

This notebook provides a foundational understanding of:
1. Creating custom datasets in PyTorch
2. Implementing and composing transformations
3. Applying data preprocessing in a structured manner
4. Building reusable data pipeline components

The code serves as a template that can be adapted for real-world machine learning projects involving custom data formats and preprocessing requirements.
