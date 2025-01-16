#%%

import numpy as np
import pandas as pd
import plotext as plt
import matplotlib.pyplot as plt


np.random.seed(115)

# Number of data points
m = 10000

# Feature inputs
X = 10 * np.random.rand(m, 1)

# Base linear function
y = 2.5 * X + 4

# Add gaussian noise
noise = np.random.normal(0, 5, size=(m, 1))
y += noise

# Round and flatten arrays to 1D
X = np.round(X, 2).ravel()
y = np.round(y, 2).ravel()

# Float feature variable vs target variable
plt.scatter(X, y)
plt.show()

data = pd.DataFrame({'X': X, 'y': y})
data.to_csv(f'linear_single_var_{m}.csv', index=False)

# %%
