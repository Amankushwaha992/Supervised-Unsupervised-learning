import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("House Price Prediction Dataset.csv")

# Features (X) and Target (y)
X = data[['Area', 'Bedrooms']]
y = data['Price']

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Print predictions
print("Predictions:", y_pred)

# Predict new house price
new_house = [[1000, 3]]  # Area=1000, Bedrooms=3
price = model.predict(new_house)
print("Predicted Price:", price[0])

# Visualization (only Area vs Price)
plt.scatter(data['Area'], data['Price'])
plt.plot(data['Area'], model.predict(data[['Area','Bedrooms']]), linestyle='dashed')
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("House Price Prediction")
plt.show()