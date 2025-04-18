import numpy as np
import pandas as pd
import sys

# Read command-line arguments
if len(sys.argv) < 4:
    print("Usage: python generate_gaussian_data.py <num_features> <max_degree> <num_samples> [<noise_std>]")
    sys.exit(1)

num_features = int(sys.argv[1])
max_degree = int(sys.argv[2])
num_samples = int(sys.argv[3])
if len(sys.argv) > 4:
    noise_std = float(sys.argv[4])
else:
    noise_std = 1.0

# Generate random feature inputs
X = 10 * np.random.rand(num_samples, num_features)

# Generate random feature coefficients
coefficients = np.random.uniform(-5, 5, size=(num_features,))
intercept = np.random.uniform(-10, 10)
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

# Calculate function outputs
y = np.zeros(num_samples)
for feature in range(num_features):
    # Factor in degree of coefficient
    y += coefficients[feature] * (X[:, feature] ** X_degrees[feature])

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
output_filename = f"gaussian_{num_features}_features_{max_degree}_degree_{num_samples}_samples.csv"
data.to_csv(output_filename, index=False)

print(f"Dataset saved as '{output_filename}'")
