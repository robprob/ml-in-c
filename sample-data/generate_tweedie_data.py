import numpy as np
import pandas as pd
import sys

# Read command-line arguments
if len(sys.argv) < 5:
    print("Usage: python generate_tweedie_data.py <num_features> <max_degree> <power> <num_samples> [<dispersion>]")
    sys.exit(1)

num_features = int(sys.argv[1])
max_degree = int(sys.argv[2])
power = float(sys.argv[3])
num_samples = int(sys.argv[4])
if len(sys.argv) > 5:
    phi = float(sys.argv[4]) # Dispersion parameter
else:
    phi = 1.0

# Validate power paramater for compound Poisson-Gamma only
if not (1 < power < 2):
    print("Power for compound gamma must be between 1 and 2, exclusive. For other distributions, use the other provided scripts.")
    sys.exit(1)

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
for feature in range(num_features):
    # Factor in degree of coefficient
    linear_combination += coefficients[feature] * (X[:, feature] ** X_degrees[feature])

# Add intercept
linear_combination += intercept

# Transform linear combinations to exponential function
mu = np.exp(linear_combination)

# Generate compound Poisson Gamma distributed target variable
y = np.zeros(num_samples)
for i in range(num_samples):
    lambda_poisson = mu[i] ** (2 - power) / (2 - power) # Adjust for Poisson rate
    num_events = np.random.poisson(lambda_poisson)

    if num_events > 0:
        shape = (2 - power) / (power - 1)
        scale = mu[i] ** (power - 1) / (2 - power)
        y[i] = np.sum(np.random.gamma(shape=shape, scale=scale, size=num_events))
    else:
        y[i] = 0

# Round data to 2 decimal places
X = np.round(X, 2)
y = np.round(y, 2)

# Save to CSV
columns = [f"X{i+1}" for i in range(num_features)]

# Create df using sample data
data = pd.DataFrame(X, columns=columns)
data['y'] = y

# Save to CSV
output_filename = f"tweedie_{num_features}_features_{max_degree}_degree_{num_samples}_samples.csv"
data.to_csv(output_filename, index=False)

print(f"Dataset saved as '{output_filename}'")
