import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

# Read command-line arguments
if len(sys.argv) < 4:
    print("Usage: python generate_polynomial_data.py <num_features> <degree> <num_samples> [<noise_std>]")
    sys.exit(1)

num_features = int(sys.argv[1])
degree = int(sys.argv[2])
num_samples = int(sys.argv[3])
if len(sys.argv) > 4:
    noise_std = float(sys.argv[4])
else:
    noise_std = 3.0

# Generate random feature inputs
np.random.seed(115)
X = 10 * np.random.rand(num_samples, num_features)

# Generate random coefficients for polynomial features
coefficients = np.random.uniform(-5, 5, size=(num_features, degree))
intercept = np.random.uniform(-10, 10)

# Compute the polynomial function
y = np.zeros(num_samples)
for feature_idx in range(num_features):
    for d in range(1, degree + 1):
        y += coefficients[feature_idx, d - 1] * (X[:, feature_idx] ** d)

# Add intercept
y += intercept

# Add Gaussian noise
noise = np.random.normal(0, noise_std, size=(num_samples,))
y += noise

# Round data to 2 decimal places
X = np.round(X, 2)
y = np.round(y, 2)

# Save to CSV
columns = [f"X{i+1}" for i in range(num_features)]

# Create df using sample data
data = pd.DataFrame(X, columns=columns)
data['y'] = y

# Save to CSV
data.to_csv(f'single_poly_{num_features}_features_{degree}_degree_{num_samples}_samples.csv', index=False)

print(f"Dataset saved as 'single_poly_{num_features}_features_{degree}_degree_{num_samples}_samples.csv'")
