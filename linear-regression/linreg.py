import sys
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

start = time.process_time()

max_degree = 2

# Read command-line arguments, establish default file path
if len(sys.argv) == 1:
    file_path = "../sample-data/5_features_2_degree_1000_samples.csv"
elif len(sys.argv) == 2:
    file_path = sys.argv[1]
else:
    print("Usage: python linreg.py <file_path>")
    sys.exit(1)

# Load in data
data = pd.read_csv(file_path)

# Separate feature and target data
y = data['y']
X = data.drop(columns=['y'])

# Transform polynomial features, if necessary
if max_degree > 1:
    poly = PolynomialFeatures(1)
    poly.fit_transform(X)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=115)

# Fit a scaler to training data
scaler = StandardScaler().fit(X_train)

# Scale training and test data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Instantiate a new linear regression model
linreg = LinearRegression()

# Time model fitting
train_start = time.process_time()
linreg.fit(X_train, y_train)
train_end = time.process_time()

# Make predictions
y_pred = linreg.predict(X_test)

# Print model parameters
print("Weights:")
i = 1
for weight in linreg.coef_:
    print(f"    Feature {i}: {weight:2f}")
    i += 1
print(f"Bias: {linreg.intercept_}\n")

# Calculate MSE
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse}\n")

print(f'Training Time: {(train_end - train_start):3f} seconds')
end = time.process_time()
print(f'Total CPU Time: {(end - start):3f} seconds')
