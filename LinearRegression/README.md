# Linear Regression Project

This project implements linear regression from scratch using PyTorch and compares it with scikit-learn's implementation. The model predicts salary based on years of experience.

## Overview

The notebook demonstrates the complete workflow of a linear regression model:

- Data loading and exploration
- Data preprocessing and scaling
- Model implementation using PyTorch
- Model implementation using scikit-learn
- Custom gradient descent implementation
- Model evaluation and visualization

## Dataset

The dataset contains two features:
- Experience Years (independent variable)
- Salary (dependent variable)

Data is split into training (70%) and testing (30%) sets. Features are standardized using `StandardScaler`.

## Key Components

### 1. Data Preprocessing
- Train-test split with random_state=12
- Feature scaling using StandardScaler
- Conversion of numpy arrays to PyTorch tensors

### 2. Custom Linear Regression Class
A custom class `Linear_regression` is implemented with:
- Randomly initialized weights (w0, w1)
- Forward pass using `__call__` method
- Custom `__repr__` method

### 3. Loss Functions
Two loss functions are implemented:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)

### 4. Gradient Descent
Manual gradient descent implementation includes:
- Gradient calculation for both parameters
- Parameter update rule
- Iterative optimization over 100 epochs

### 5. Model Comparison
The custom implementation is compared with scikit-learn's `LinearRegression` for validation.

## Visualizations

The notebook includes several plots:
- Training vs test data points
- Regression line visualization
- Loss function over different theta values
- Gradient descent convergence tracking

## Results

The final model is saved using PyTorch's save functionality, and the test loss is calculated to evaluate model performance.

## Dependencies

- pandas
- numpy
- scikit-learn
- matplotlib
- torch
