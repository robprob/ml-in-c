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
- Regularization: L2 (Ridge), L1 (Lasso), and Elastic Net

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
2. Build executable:
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
num_epochs = 100
learning_rate = 0.05
test_proportion = 0.3
gradient_descent = batch
batch_size = 0
l2_lambda = 0.1
l1_lambda = 0.05
```

## Command Log Output
Below is an example of program output:
```bash
File Path: ../sample-data/linear_multi_var_500.csv
Number of Epochs: 100
Learning Rate: 0.05
L2 Lambda: 0
L1 Lambda: 0
Epoch 0: Train MSE: 2799.026694
Epoch 10: Train MSE: 338.355341
Epoch 20: Train MSE: 47.654339
Epoch 30: Train MSE: 13.164656
Epoch 40: Train MSE: 9.057297
Epoch 50: Train MSE: 8.566527
Epoch 60: Train MSE: 8.507713
Epoch 70: Train MSE: 8.500646
Epoch 80: Train MSE: 8.499795
Epoch 90: Train MSE: 8.499692
Epoch 100: Train MSE: 8.499679

Test MSE: 8.851673
Training Time : 0.001422 seconds
```


