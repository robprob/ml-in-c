#%%

import numpy as np
import pandas as pd
import plotext as plt
import matplotlib.pyplot as plt


np.random.seed(115)

# Number of data points
m = 10000

# Feature inputs
X1 = 10 * np.random.rand(m, 1)
X2 = 10 * np.random.rand(m, 1)
X3 = 10 * np.random.rand(m, 1)
X4 = 10 * np.random.rand(m, 1)

# Base linear function
y = (7 * X1) + (6 * X2) + (0.2 * X3) + (-4 * X4) + 4

# Add gaussian noise
noise = np.random.normal(0, 3, size=(m, 1))
y += noise

# Round and flatten arrays to 1D
X1 = np.round(X1, 2).ravel()
X2 = np.round(X2, 2).ravel()
X3 = np.round(X3, 2).ravel()
X4 = np.round(X4, 2).ravel()
y = np.round(y, 2).ravel()

# Display subplots for each feature variable
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

axes[0, 0].scatter(X1, y)
axes[0, 0].set_title('X1')

axes[0, 1].scatter(X2, y)
axes[0, 1].set_title('X2')

axes[1, 0].scatter(X3, y)
axes[1, 0].set_title('X3')

axes[1, 1].scatter(X4, y)
axes[1, 1].set_title('X4')

plt.suptitle("Feature Variables vs Target (y)")
plt.show()

data = pd.DataFrame({'X1': X1,
                     'X2': X2,
                     'X3': X3,
                     'X4': X4,
                     'y': y})

data.to_csv(f'linear_multi_var_' + str(m) + '.csv', index=False)

# %%
