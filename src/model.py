import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Get current file directory
current_dir = os.path.dirname(__file__)

# Build correct path
file_path = os.path.join(current_dir, '..', 'data', 'housing.csv')

# Load dataset
data = pd.read_csv(file_path)

print("Dataset loaded successfully!")

# Features and target
X = data[['area', 'bedrooms', 'age']]
y = data['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Print results
print("Predictions:", predictions)
print("Actual:", y_test.values)

# Plot (area vs price)
plt.scatter(data['area'], data['price'])
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("House Price Distribution")

# Create results folder path
results_path = os.path.join(current_dir, '..', 'results')

# Create folder if not exists
os.makedirs(results_path, exist_ok=True)

# Save file
output_file = os.path.join(results_path, 'output.png')
plt.savefig(output_file)

print("Graph saved in results folder")