# Linear Regression
Implementation of Linear Regression using gradient descent, with support for fine-tuning and regularization.

---
## Features
### Core Functionality
- Feature standardization
- Support for both single and multi-variable regression
- Support for batch, mini-batch, and stochastic gradient descent
- Exports test data and predictions to CSV upon completion

### Tuning
- Customizable number of epochs and learning rate
- Regularization: L2 (Ridge), L1 (Lasso), and Elastic Net with mix-ratio, r
- Early stopping when validation set has reached a minimum error, inhibiting variance via overfitting

### Performance Metrics
- Regularly reports training and validation MSE during training and then final MSE on test set
- Measures CPU clock time during model training

---

## Installation
1. Clone main repository:
   ```bash
   git clone https://github.com/robprob/ml-in-c.git
   cd ml-in-c/linear-regression
   ```
2. Build executable (requires a C compiler):
   ```bash
   make
   ```
3. Run using parameters specified in **config.txt**
   ```bash
   ./linreg
   ```

## Usage Examples
### Edit fields in config.txt
```ini
[Dataset]
file_path = ../sample-data/linear_multi_var_10000.csv
standardize = true
test_proportion = 0.2
valid_proportion = 0.1

[Regularization]
l2_alpha = 0.1
l1_alpha = 0.0
mix_ratio = 0.0

[Training]
num_epochs = 1000
learning_rate = 0.01
batch_size = 0
early_stopping = true
```

## Command Log Output
```plaintext
Training Size: 7008
Validation Size: 1027
Test Size: 1965

==========================================
| Epoch |   Train MSE   | Validation MSE |
==========================================
|     0 |    3215.50643 |     3138.19979 |
|   100 |      64.91487 |       65.77240 |
|   200 |       9.87927 |       10.15291 |
|   300 |       8.91455 |        8.93430 |
|   400 |       8.89755 |        8.88098 |
|   500 |       8.89724 |        8.87584 |
|   600 |       8.89723 |        8.87520 |
|   700 |       8.89723 |        8.87511 |
------------------------------------------
Stopping early, validation set has reached a minimum error.

Trained Model Parameters (un-standardized)
Weights: 7.01052, 6.0026, 0.205286, -3.98885
Bias: 3.87475

Test MSE: 8.828893
Training Time: 0.167617 seconds
```


