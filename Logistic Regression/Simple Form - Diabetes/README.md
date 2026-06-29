# Diabetes Classification with Logistic Regression

This project implements a logistic regression model using PyTorch to predict diabetes outcomes based on diagnostic measurements. The model is trained on the Pima Indians Diabetes Database.

## Project Overview

The notebook demonstrates a complete machine learning workflow including data preprocessing, model training, validation, and evaluation. The implementation uses PyTorch's neural network modules to build a binary classification model with sigmoid activation.

## Dataset

The dataset contains 768 samples with 8 features and 1 target variable:

### Features
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI (Body Mass Index)
- DiabetesPedigreeFunction
- Age

### Target
- Outcome: Binary classification (0 = No Diabetes, 1 = Diabetes)

## Dependencies

- numpy
- pandas
- torch
- scikit-learn
- matplotlib

## Data Preprocessing

1. Feature-target separation
2. Train-test split (80-20)
3. Train-validation split (70-30 from training data)
4. Feature scaling using StandardScaler
5. Conversion to PyTorch tensors

### Data Split Summary
- Training samples: 429
- Validation samples: 185
- Test samples: 154

## Model Architecture

The model uses a sequential neural network with:
- Linear layer: 8 input features, 1 output
- Sigmoid activation function

### Model Parameters
- Loss function: Binary Cross Entropy (BCELoss)
- Optimizer: SGD with momentum (0.9)
- Learning rate: 0.1
- Batch size: 100 (training), 50 (validation), 154 (test)

## Training Process

The model is trained for 10 epochs with:
- Batch-wise loss calculation
- Gradient descent optimization
- Training and validation metrics tracking

### Training Metrics Tracked
- Training loss per epoch
- Training accuracy per epoch
- Validation loss per epoch
- Validation accuracy per epoch

## Results

The notebook computes and displays:
- Training loss and accuracy for each epoch
- Validation loss and accuracy for each epoch
- Test L1 loss
- Test accuracy

## Usage

1. Ensure the diabetes.csv file is in the same directory as the notebook
2. Run all cells sequentially
3. The model will train and display metrics for each epoch
4. Final test evaluation will be displayed at the end

## File Structure

```
├── 1.IPYNB           # Main Jupyter notebook
├── diabetes.csv       # Dataset file
└── README.md         # Project documentation
```

## Notes

- The model uses a simple logistic regression approach suitable for binary classification problems
- Standard scaling is applied to all features before training
- Random seed is set for reproducibility
- The model achieves reasonable accuracy on the validation set (~74-76%)
