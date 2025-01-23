# ML in C
**ML in C** is a collection of machine learning algorithms implemented with the help of native C library, [MLper](https://github.com/robprob/ml-in-c/tree/main/mlper). This project was originally created for educational purposes as I learn C, with the goal of making lightweight, fast, and dependency-free models.

## Quick Start
1. Clone this repo:
   ```bash
   git clone https://github.com/robprob/ml-in-c.git
   cd ml-in-c
   ```
2. Review model-specific documentation

---
# Features
## Models
- [Linear and Polynomial Regression](https://github.com/robprob/ml-in-c/tree/main/linear-regression)
## ML Helper Library
- Algorithm-agnostic helper functions
- [MLper](https://github.com/robprob/ml-in-c/tree/main/mlper)
## Data Generation
- Sample CSV data and the Python scripts used to generate them
- [Sample Data](https://github.com/robprob/ml-in-c/tree/main/sample-data)

---
## TODO
- Provide more detailed help for model configuration, e.g. recommending early stopping is disabled for SGD
- Refactor linreg.c into modular functions (As more models are developed, shift additional utilities to Mlper)
- Improve sample data generation and allow command-line interaction
- Add additional pre-processing/analysis utility to MLper library
- Improve readability and visual appeal of terminal output

## Future Models
- Logistic Regression
- GLMs
- K-Means Clustering
