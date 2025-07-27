import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
n_samples = 500

square_footage = np.random.randint(500, 4000, n_samples)  # Square footage between 500 and 4000
bedrooms = np.random.randint(1, 6, n_samples)             # Bedrooms 1-5
bathrooms = np.random.randint(1, 4, n_samples)            # Bathrooms 1-3

# Generate price based on features (simple formula + noise)
price = (square_footage * 150) + (bedrooms * 10000) + (bathrooms * 5000) + np.random.randint(-20000, 20000, n_samples)

# Create DataFrame
data = pd.DataFrame({
    'square_footage': square_footage,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'price': price
})

# Display first 5 rows
print(data.head())

import matplotlib.pyplot as plt
import seaborn as sns

# Basic info
print("\nDataset Info:")
print(data.info())

# Summary statistics
print("\nSummary Statistics:")
print(data.describe())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Pairplot to see relationships
sns.pairplot(data)
plt.show()

# Correlation heatmap
plt.figure(figsize=(6,4))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

from sklearn.model_selection import train_test_split

# Define features and target
X = data[['square_footage', 'bedrooms', 'bathrooms']]
y = data['price']

# Split into training and testing sets (80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

from sklearn.linear_model import LinearRegression

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Model coefficients
print("\nModel Coefficients:")
print(f"Square Footage: {model.coef_[0]:.2f}")
print(f"Bedrooms: {model.coef_[1]:.2f}")
print(f"Bathrooms: {model.coef_[2]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Predict on test data
y_pred = model.predict(X_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.4f}")

# Plot Actual vs Predicted
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred, color="blue")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # reference line
plt.show()
