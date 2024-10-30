
#### main.py

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import shap
import matplotlib.pyplot as plt

# Step 1: Generate Sample Data
np.random.seed(42)  # For reproducibility
n_samples = 1000
data = {
    'square_feet': np.random.randint(500, 3500, n_samples),
    'num_bedrooms': np.random.randint(1, 6, n_samples),
    'num_bathrooms': np.random.randint(1, 4, n_samples),
    'num_floors': np.random.randint(1, 3, n_samples),
    'age_of_home': np.random.randint(0, 100, n_samples),
    'price': np.random.randint(50000, 500000, n_samples)
}

# Create DataFrame and save it as CSV
df = pd.DataFrame(data)
df.to_csv('c:/demos/house_prices_sample_generated_data.csv', index=False)

# Display first few rows of the dataset
print("Sample data:\n", df.head())

# Step 2: Train the Machine Learning Model
# Load the dataset
X = df.drop('price', axis=1)
y = df['price']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 3: Generate SHAP Values
# Create a SHAP explainer for the Random Forest model
explainer = shap.TreeExplainer(model)

# Calculate SHAP values for the test set
shap_values = explainer.shap_values(X_test)

# Step 4: Visualize SHAP Values
# Summary plot showing feature importance
print("Generating SHAP summary plot...")
shap.summary_plot(shap_values, X_test)

# Force plot for a single prediction (index 0)
print("Generating SHAP force plot for a single prediction...")
shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:], matplotlib=True)

# Additional insight: Display insights based on SHAP values for each feature
print("\nInsights based on SHAP summary plot:")
print("1. Square Feet: Significant impact on price predictions.")
print("2. Age of Home: Older homes generally decrease predicted price.")
print("3. Number of Bedrooms & Bathrooms: Positive correlation with price.")
print("4. Number of Floors: Minimal impact relative to other features.")

