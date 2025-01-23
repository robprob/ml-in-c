# Linear and Polynomial Regression
Implementation of both linear regression and polynomial regression using gradient descent and regularization techniques.

---
## Features
### Core Functionality
- Feature standardization
- Support for both single and multi-variable regression
- Support for batch, mini-batch, and stochastic gradient descent
- Exports test data and predictions to CSV upon completion

### Tuning
- Customizable number of epochs and learning rate
- Automated polynomial transformation/feature generation
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
polynomial_degree = 1
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
Training Size: 7047
Validation Size: 1025
Test Size: 1928

==========================================
| Epoch |   Train MSE   | Validation MSE |
==========================================
|     0 |    3213.82519 |     3174.79799 |
|   100 |      65.29408 |       65.71564 |
|   200 |       9.93109 |       10.18250 |
|   300 |       8.95499 |        9.08346 |
|   400 |       8.93770 |        9.04767 |
|   500 |       8.93739 |        9.04482 |
|   600 |       8.93738 |        9.04447 |
------------------------------------------
Stopping early, validation set has reached a minimum error.

Weights:
Feature 1:  x^1:  7.015
Feature 2:  x^1:  6.006
Feature 3:  x^1:  0.208
Feature 4:  x^1: -4.004

Bias: 3.898

Test MSE: 8.589657
Training Time: 0.142319 seconds
```
