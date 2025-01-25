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
- Measures CPU clock time during program execution and model training

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
file_path = ../sample-data/5_features_1_degree_1000_samples.csv
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
Training Size: 723
Validation Size: 97
Test Size: 180

==========================================
| Epoch |   Train MSE   | Validation MSE |
==========================================
|     0 |    2426.65269 |     2274.22015 |
|    10 |     298.10426 |      282.20441 |
|    20 |      41.60404 |       40.40776 |
|    30 |      10.55594 |       11.01081 |
|    40 |       6.78016 |        7.45742 |
|    50 |       6.31862 |        7.04114 |
|    60 |       6.26184 |        6.99835 |
|    70 |       6.25478 |        6.99652 |
|    80 |       6.25389 |        6.99768 |
------------------------------------------
Stopping early, validation set has reached a minimum error.

Weights:
    Feature   1:  x^1: -0.810
    Feature   2:  x^1:  3.460
    Feature   3:  x^1:  2.782
    Feature   4:  x^1:  2.500
    Feature   5:  x^1: -2.749
Bias: 25.764

Test MSE: 5.758157
Test RMSE: 2.399616

Training Time: 0.002459 seconds
Total CPU Time: 0.004423 seconds
```
