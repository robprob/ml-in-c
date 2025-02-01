import numpy as np
import pandas as pd
import sys

# Read command-line arguments
if len(sys.argv) < 4:
    print("Usage: python generate_logistic_data.py <num_features> <class_balance> <num_samples> [<noise_std>]")
    sys.exit(1)

num_features = int(sys.argv[1])
class_balance = float(sys.argv[2])
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

# Print generated parameters
print("Generated Parameters")
for i in range(len(coefficients)):
    print(f"    Coefficient {i + 1:3d}: {coefficients[i]:6.3f}")
print(f"    Intercept: {intercept}\n")

# Calculate linear combination of input features via matrix multiplication and add intercept
logits = X @ coefficients + intercept

# Add Gaussian noise
logits += np.random.normal(0, noise_std, size=(num_samples,))

# Calculate probabilities via sigmoid function
probabilities = 1 / (1 + np.exp(-logits))

# Generate outcomes based on given class balance
threshold = np.percentile(probabilities, (1 - class_balance) * 100)
# Cast truth value of comparison of probabilities to threshold as integer, resulting in binary output
y = (probabilities >= threshold).astype(int)

# Round features to 2 decimal places
X = np.round(X, 2)

# Save to CSV
columns = [f"X{i+1}" for i in range(num_features)]

# Create df using sample data
data = pd.DataFrame(X, columns=columns)
data['y'] = y

# Save to CSV
output_filename = f"logistic_{num_features}_features_{class_balance}_balance_{num_samples}_samples.csv"
data.to_csv(output_filename, index=False)

print(f"Dataset saved as '{output_filename}'")
