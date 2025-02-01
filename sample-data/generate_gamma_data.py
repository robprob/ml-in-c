import numpy as np
import pandas as pd
import sys

# Read command-line arguments
if len(sys.argv) < 5:
    print("Usage: python generate_gamma_data.py <num_features> <max_degree> <shape> <num_samples>")
    sys.exit(1)

num_features = int(sys.argv[1])
max_degree = int(sys.argv[2])
shape = int(sys.argv[3])
num_samples = int(sys.argv[4])

# Generate random feature inputs
X = 2 * np.random.rand(num_samples, num_features)

# Generate random feature coefficients
coefficients = np.random.uniform(0, 1, size=(num_features,))
intercept = np.random.uniform(-2, 2)
# Generate random degrees of each feature
X_degrees = np.random.randint(1, max_degree + 1, num_features)
# Ensure max degree reached
if max_degree not in X_degrees:
    X_degrees[np.random.randint(0, num_features)] = max_degree

# Print generated parameters
print("Generated Parameters")
for i in range(len(coefficients)):
    print(f"    Coefficient {i + 1:3d}: {coefficients[i]:6.3f}")
print(f"    Degrees: {X_degrees}")
print(f"    Intercept: {intercept}\n")

# Calculate linear combinations of input features
linear_combination = np.zeros(num_samples)
# Clip linear combinations to [-10, 10]
linear_combination = np.clip(linear_combination, -10, 10)
for feature in range(num_features):
    # Factor in degree of coefficient
    linear_combination += coefficients[feature] * (X[:, feature] ** X_degrees[feature])

# Add intercept
linear_combination += intercept

# Transform linear combinations to exponential function
mu = np.exp(linear_combination)

# Calculate scale, clipping to prevent it being too small
scale = np.clip(mu / shape, 1e-6, None)

# Randomly generate Gamma outputs
y = np.random.gamma(shape, scale, size=num_samples)

# Round data to 2 decimal places
X = np.round(X, 2)
y = np.round(y, 2)

# Save to CSV
columns = [f"X{i+1}" for i in range(num_features)]

# Create df using sample data
data = pd.DataFrame(X, columns=columns)
data['y'] = y

# Save to CSV
output_filename = f"gamma_{num_features}_features_{max_degree}_degree_{num_samples}_samples.csv"
data.to_csv(output_filename, index=False)

print(f"Dataset saved as '{output_filename}'")
