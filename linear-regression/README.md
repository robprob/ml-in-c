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
- Measures CPU clock time during model training

---
## Usage Examples
### Default Parameters
Run the program with default test settings, or 100 epochs and a learning rate of 0.05:
```bash
./linreg
```
### Linear Regression
Train a linear regression model, specifying number of epochs and learning rate:
```bash
./linreg ../sample-data/linear_multi_var_500.csv 1000 0.01
```
### Ridge Regression
Train a linear regression with L2 regularization (ridge) using a lambda of 0.1:
```base
./linreg ../sample-data/linear_multi_var_500.csv 1000 0.01 --ridge 0.1
```
### Lasso Regression
Train a linear regression model with L1 regularization (lasso) using a lambda of 0.05:
```base
./linreg ../sample-data/linear_multi_var_500.csv 1000 0.01 --lasso 0.05
```
### Elastic Net
Train a linear regression model using a linear combination of L2 and L1 regularization:
```base
./linreg sample-data/data.csv 1000 0.01 --ridge 0.1 --lasso 0.05
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
