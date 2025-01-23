## v0.2.0 (2025-01-22)
### Added
- Support for **Polynomial Regression**, implementing a "poly_transform" function in MLper and a polynomial_degree option in config.txt
- MLper function **double_pow**, a lightweight alternative to Math.pow for positive-integer powers
### Changed
- **Refactored entire MLper library** into modular units, aligning with more conventional practices of C libraries while improving reusability and extensibility
- Linked MLper to linreg.c as a **static library** instead of recompiling each time
- Moved initializing of memory for train/test splits into a separate MLper function, **initialize_splits**, allowing it to be dynamically allocated AFTER feature transformation (free_dataset still frees all splits)
- Improved printing of model parameters, specifying polynomial degree of each feature
- Improved header of MLper function, export_results, specifying polynomial degree of each feature
- Various wording changes to improve clarity

---

## v0.1.1 (2025-01-21)
### Added
- MLper **validation_split**, allowing further split of training data into training and validation set
- Config option, **valid_proportion**, which specifies proportion of TOTAL data set to split into a validation set
- After pre-processing, prints size of training, validation, and test sets
- Experimental **patience counter** for early stopping, currently set at 2 to require 2 "bad" MSE comparisons in a row before stopping (MSE comparison still performed every 10% of training epochs)
### Changed
- Parameters in config.txt are now **categorized** for visual clarity and ease of use
- Improved "parse_config", removing the slightly more efficient switch statements in exchange for the flexibility of strcmp
- Moved config parameter test_proportion from ML model to Dataset struct
- **Refactored** redundant train_test_split pointer access for readability and performance
- Improved readability and visual appeal of epoch log output
- Various wording changes to improve clarity
### Fixed
- With implementation of validation_split, early stopping now correctly evaluates a subset of the training set, **preventing data leakage**

---

## v0.1.0 (2025-01-20)
### Added
- This **changelog** :)
- Elastic Net **mix-ratio, r**, added as option in config.txt
- Option for simple **early stopping** when validation MSE has reached a minimum, earning yourself a "beautiful free lunch"
- Un-standardized model weights and bias are now printed after training
- Support for **stochastic** and **mini-batch** gradient descent, implementing a "shuffle_batch" function in MLper

### Changed
- MLper "unstandardize" function now **accepts a specific feature matrix** to un-standardize
- Seed pseudo-RNG used for train_test_split with current time
- Various wording changes to improve clarity

### Fixed
- Corrected name of hyperparameter inputs for regularization from "lambda" to "alpha" (alpha is strength of regularization before scaling lambda to size of training set (with ridge))
- MLper "standardize" function now calculates statistics based on training set alone, **preventing data leakage**
- MLper standardize now gracefully exits if unable to allocate feature statistic arrays

### Removed
- Commented out printing of selected model parameters, keeping terminal clean
- gradient_descent input variant removed from config.txt (can be inferred from batch_size)
