import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

# Read command-line arguments
if len(sys.argv) < 3:
    print("Usage: python generate_linear_data.py <num_features> <num_samples> [<noise_std>]")
    sys.exit(1)

num_features = int(sys.argv[1])
num_samples = int(sys.argv[2])
if len(sys.argv) > 3:
    noise_std = float(sys.argv[3])
else:
    noise_std = 3.0

# Generate random feature inputs
np.random.seed(115)
X = 10 * np.random.rand(num_samples, num_features)

# Generate random coefficients for features
coefficients = np.random.uniform(-10, 10, size=(num_features,))
intercept = np.random.uniform(-5, 5)

# Base linear function
y = np.dot(X, coefficients) + intercept

# Add Gaussian noise
noise = np.random.normal(0, noise_std, size=(num_samples,))
y += noise

# Round data to 2 decimal places
X = np.round(X, 2)
y = np.round(y, 2)

# Generate column header names
columns = [f"X{i+1}" for i in range(num_features)]

# Create df using sample data
data = pd.DataFrame(X, columns=columns)
data['y'] = y

# Save to CSV
data.to_csv(f'linear_multi_var_{num_features}_features_{num_samples}_samples.csv', index=False)

print(f"Dataset saved as 'linear_multi_var_{num_features}_features_{num_samples}_samples.csv'")
