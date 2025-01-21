# Linear Regression
Implementation of Linear Regression using gradient descent, with support for fine-tuning and regularization.

---
## Features
### Core Functionality
- Support for both single and multi-variable regression
- Feature standardization
- Exports test data and predictions to CSV upon completion

### Tuning
- Customizable number of epochs and learning rate
- Regularization: L2 (Ridge), L1 (Lasso), and Elastic Net with mix-ratio, r
- Early stopping when validation set has reached a minimum error, preventing overfitting

### Performance Metrics
- Regularly reports MSE during training and then final MSE on test set
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
```txt
file_path = ../sample-data/linear_multi_var_500.csv
standardize = true
num_epochs = 1000
learning_rate = 0.05
test_proportion = 0.3
early_stopping = true
batch_size = 0
l2_alpha = 0.0
l1_alpha = 0.0
mix_ratio = 0.0
```

## Command Log Output
Below is an example of program output:
```bash
Epoch 0: Train MSE: 2868.285045
Epoch 100: Train MSE: 8.914645
Epoch 200: Train MSE: 8.914643
Stopping early, validation set has reached a minimum error.

Trained Model Parameters (un-standardized)
Weights: 6.99622, 5.95143, 0.213616, -4.09234
Bias: 4.92785

Test MSE: 8.031067
Training Time: 0.002444 seconds
```


