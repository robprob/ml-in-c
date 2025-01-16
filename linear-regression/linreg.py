#%%

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

"""
Linear Function
h(x) = w * x + b

h(x): linear prediction
w: weight vector
x: feature vector
b: bias

Mean squared error (MSE) Function
E = (1/2) * Σ(i=1)^n (yi - ^yi)^2
"""


def main():
    print('Original function (without added noise):\n y = 2.50x + 4.0')

    data = pd.read_csv('../sample-data/linear_single_var_500.csv')

    # Feature variables returned as np array
    X = data["X"].values

    # Target variable returned as np array
    y = data["y"].values

    # Split into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=115)

    # Number of training samples, n
    n = len(X_train)

    # Initialize weight vector and bias at 0 (horizontal line)
    w = 0
    b = 0

    # Maximum iterations (epochs)
    max_iter = 1000

    # Learning rate ("step" size)
    L = 0.005

    # Start clock
    start_time = time.process_time()

    # Iterate epochs, training model
    for _ in range(max_iter):
        # Calculate linear predictions (y-hat, or ^y)
        y_pred = w * X_train + b

        # Partial derivative of MSE, or E, with respect to w and b
        dE_dw = (-2/n) * np.sum(X_train * (y_train - y_pred))
        dE_db = (-2/n) * np.sum(y_train - y_pred)

        # Update weights in opposite direction of highest slope, moving towards a local minima
        w -= L * dE_dw
        b -= L * dE_db

    # End clock
    end_time = time.process_time()

    print(f'Resulting function:\n y = {round(w,2)}x + {round(b,2)}')
    print(f'Training time: {round((end_time - start_time), 4)} seconds')

    # Make test predictions on test set
    y_pred = w * X_test + b

    # Plot test data
    plt.scatter(X_test, y_test)

    # Plot linear predictions
    plt.plot(X_test, y_pred, color='red')
    plt.show()


if __name__ == '__main__':
    main()

# %%
