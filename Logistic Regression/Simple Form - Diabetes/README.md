```markdown
# Diabetes Classification with Logistic Regression

This project implements a logistic regression model using PyTorch to predict diabetes outcomes based on diagnostic measurements. The model is trained on the Pima Indians Diabetes Database.

## Project Overview

The notebook demonstrates a complete machine learning workflow including data preprocessing, model training, validation, and evaluation. The implementation uses PyTorch's neural network modules to build a binary classification model with sigmoid activation.

## Dataset

The dataset contains 768 samples with 8 features and 1 target variable.

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

## Data Preprocessing

1. Feature-target separation
2. Train-test split (80-20)
3. Train-validation split (9% from training data)
4. Feature scaling using StandardScaler
5. Conversion to PyTorch tensors

### Data Split Summary

- Training samples: 558
- Validation samples: 56
- Test samples: 154

## Model Architecture

The model uses a single linear layer with sigmoid activation.

### Model Parameters

- Loss function: Binary Cross Entropy Loss (BCELoss)
- Optimizer: SGD with momentum (0.9)
- Learning rate: 0.1
- Batch size: 100 (training), 10 (validation), 200 (test)

## Training Process

The model is trained for 200 epochs with:

- Batch-wise loss calculation
- Gradient descent optimization
- Training and validation metrics tracking
- Best model checkpoint saving

### Training Metrics Tracked

- Training loss per epoch
- Training accuracy per epoch
- Validation loss per epoch
- Validation accuracy per epoch

## Model Saving

The best model based on validation loss is automatically saved as `LogisticRegressionModel.pt` during training.

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
├── main.ipynb          # Main Jupyter notebook
├── diabetes.csv         # Dataset file
├── LogisticRegressionModel.pt  # Saved model (generated after training)
└── README.md           # Project documentation
```

## Notes

- The model uses a simple logistic regression approach suitable for binary classification problems
- Standard scaling is applied to all features before training
- Random seed is set for reproducibility
- Zero values in Glucose, BloodPressure, SkinThickness, Insulin, and BMI are replaced with column means during preprocessing
```
