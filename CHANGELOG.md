## v0.3.0 (2025-01-31)
### Added
- Support for **Logistic Regression**
- Support for **Generalized Linear Models** (GLMs), including **Gaussian**, **Poisson**, **Gamma**, and **Tweedie** distributions
- Link and inverse **link** functions to MLper metrics.c
- Variety of **loss** functions to MLper metrics.c
- **Linear algebra** functions to MLper metrics.c, starting with dot_prod
- **Derivative residual** functions for each generalized model distribution
- New **scale-independent performance metrics**, such as r_squared, added to MLper metrics.c
- New gradient loss function, **regularization_loss**, to properly display the effects of the regularization gradient, in MLper metrics.c
- Python scripts to **generate Poisson data, Gamma data, Tweedie data**
- Many more new functions and utilities
- Open source **license**
### Changed
- Instead of commenting the block out, programs instead checks if a VERBOSE macro has been defined to determine whether model parameters are printed
- Various wording changes to improve clarity

---

## v0.2.2 (2025-01-26)
### Added
- WIP implementation of **Logistic Regression**
- Python script to generate **logistic data**
- New function, **sigmoid**, in MLper metrics.c
- New function, **log_loss**, in MLper metrics.c
- New function, **average_log_loss**, in MLper metrics.c
- New function, **accuracy**, in MLper metrics.c
- New function, **classification_metrics**, in MLper metrics.c
- Officially implemented configuration options for both **tolerance and patience** with respect to gradient descent
### Changed
- Started using **memcpy** to efficiently copy arrays to/from model
- Various wording changes to improve clarity

---

## v0.2.1 (2025-01-24)
### Added
- After training, test **RMSE** is printed alongside MSE
### Changed
- Changed all instances of **"num_entries" to "num_samples"**
- **Improved sample data generation**, allowing command-line input and allowing randomized polynomial feature generation
- **Improved Python example** by using scikit-learn to properly represent training efficiency
- **Improved CPU time measurements**, including both training and total program CPU time
### Fixed
- **Bias now properly calculated** for polynomial features, using original feature mean/standard deviation before polynomial transformation
### Removed
- **Static seeding** of data generation

---

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
