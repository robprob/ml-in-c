# Generalized Linear Models (GLMs)
WIP implementations of various generalized linear models, including Linear, Poisson, Gamma, and Tweedie regression. Support for multiple optimization strategies and regularization techniques.

---
## Features
### Core Functionality
- Gaussian, Poisson, Gamma, and Tweedie distributions
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
- MSE, R^2, Poisson Deviance, Gamma Deviance, Deviance R^2, and more
- Measures CPU clock time during program execution and model training

---

## Installation
1. Clone main repository:
   ```bash
   git clone https://github.com/robprob/ml-in-c.git
   cd ml-in-c/general-regression
   ```
2. Build executable (requires a C compiler):
   ```bash
   make
   ```
3. Run using parameters specified in **config.txt**
   ```bash
   ./genreg
   ```

## Usage Examples
### Edit fields in config.txt
```ini
[Model Selection]
distribution_type = gaussian
p = 0

[Dataset]
file_path = ../sample-data/gaussian_5_features_1_degree_1000_samples.csv
polynomial_degree = 1
standardize = true
test_proportion = 0.2
valid_proportion = 0.1

[Regularization]
l2_alpha = 0.1
l1_alpha = 0.0
mix_ratio = 0.0

[Training]
num_epochs = 100
learning_rate = 0.05
batch_size = 0
early_stopping = false
tolerance = .001
patience = 2

```

## Command Log Output
```plaintext
Training Size: 663
Validation Size: 122
Test Size: 215

==========================================
| Epoch |  Train Loss  | Validation Loss |
==========================================
|     0 |    596.13229 |       530.48946 |
|    10 |    217.41881 |       191.57115 |
|    20 |     86.88630 |        69.74507 |
|    30 |     42.94914 |        25.85537 |
|    40 |     28.85248 |        10.01481 |
|    50 |     24.78674 |         4.29178 |
|    60 |     23.92742 |         2.22474 |
|    70 |     23.98373 |         1.48011 |
|    80 |     24.22629 |         1.21360 |
|    90 |     24.44900 |         1.11949 |
|   100 |     24.61172 |         1.08711 |
------------------------------------------

Weights:
    Feature   1:  x^1:  1.185
    Feature   2:  x^1: -3.129
    Feature   3:  x^1:  1.245
    Feature   4:  x^1: -4.058
    Feature   5:  x^1: -0.483
Bias: 7.850

Test Loss: 1.26593
Test MSE: 1.26593
Test RMSE: 1.12514

Test R^2: 0.99518

Training Time: 0.003753 seconds
Total CPU Time: 0.005848 seconds
```
