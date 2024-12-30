from sklearn.linear_model import LinearRegression
import numpy as np

"""
House Price Prediction using Linear Regressi

This code demonstrates how to use a LinearRegression model from scikit-learn to predict house prices based on 
features such as square footage, number of bedrooms, and house age. It trains the model on sample data and then 
predicts the price for a new house with specified features.
"""

# Sample data
X = np.array([[1500, 3, 8], [1800, 4, 9], [1200, 2, 7], [2000, 4, 10]])  # Features: [square footage, bedrooms, age]
y = np.array([300000, 400000, 200000, 500000])  # Target: house prices

# Initialize the Linear Regressi model
model = LinearRegression()

# Train the model on the dataset
model.fit(X, y)

# Predicting the price for a new house
new_house = np.array([[1700, 3, 8]])  # New house features: [square footage, bedrooms, age]
predicted_price = model.predict(new_house)  # Predict price based on trained model
print("Predicted Price:", predicted_price[0])  # Output the predicted price

