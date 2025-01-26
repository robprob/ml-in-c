import sys
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

start = time.process_time()

# Read command-line arguments, establish default file path
if len(sys.argv) == 1:
    file_path = "../sample-data/3_features_0.7_balance_1000_samples.csv"
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

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=115)

# Fit a scaler to training data
scaler = StandardScaler().fit(X_train)

# Scale training and test data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Instantiate a new linear regression model
logreg = LogisticRegression(penalty='l2', verbose=1)

# Time model fitting
train_start = time.process_time()
logreg.fit(X_train, y_train)
train_end = time.process_time()

# Make predictions
y_pred = logreg.predict(X_test)

# Print model parameters
print("Weights:")
i = 1
for weight in logreg.coef_:
    print(f"    Feature {i}: {weight}")
    i += 1
print(f"Bias: {logreg.intercept_}\n")

# Calculate accuracy
acc = logreg.score(X_test, y_test)
print(f"Test Accuracy: {acc}\n")

print(f'Training Time: {(train_end - train_start):3f} seconds')
end = time.process_time()
print(f'Total CPU Time: {(end - start):3f} seconds')
