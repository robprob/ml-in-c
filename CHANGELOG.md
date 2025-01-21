## v0.1.0 (2025-01-19)
### Added
- This changelog :)
- Elastic net mix-ratio, r, added as option in config.txt
- Option for simple early stopping when validation MSE has reached a minimum, earning yourself a "beautiful free lunch"
- Un-standardized model weights and bias are now printed after training
- Support for stochastic and mini-batch gradient descent, implementing a "shuffle_batch" function in MLper

### Changed
- MLper "unstandardize" function now accepts a specific feature matrix to un-standardize
- Seed pseudo-RNG used for train_test_split with current time
- Various wording changes to improve clarity

### Fixed
- Corrected name of hyperparameter inputs for regularization from "lambda" to "alpha" (alpha is strength of regularization before scaling lambda to size of training set (with ridge))
- MLper "standardize" function now calculates statistics based on training set alone, preventing data leakage
- MLper standardize now gracefully exits if unable to allocate feature statistic arrays

### Removed
- Commented out printing of selected model parameters, keeping terminal clean
- gradient_descent input variant removed from config.txt (can be inferred from batch_size)
