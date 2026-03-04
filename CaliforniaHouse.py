from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Load the California housing dataset (with feature names for visualizations)
data = fetch_california_housing()
X = data.data
y = data.target
feature_names = list(data.feature_names)
#split the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.2,random_state=42)

#Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data loaded and prepocessed sucessfully")

#Train a linear regression model
model = LinearRegression()
model.fit(X_train_scaled,y_train)

#test the accuracy of the model and how good the model 
print("MODEL PERFORMANCE:")
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test,y_pred)
print(f"MSE is:{mse}")
rmse = root_mean_squared_error(y_test,y_pred)
print(f"RMSE is:{rmse}")
rsquare= r2_score(y_test,y_pred)
print(f"R square is:{rsquare}")

# ============ DATA VISUALIZATION ============

# 1. Scatter plot: Median Income vs House Price
plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], y, alpha=0.3, s=10, c='steelblue', edgecolors='none')
plt.xlabel('Median Income')
plt.ylabel('House Price (in $100,000)')
plt.title('Relationship Between Median Income and House Price')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 2. Histogram: Distribution of Median Income
plt.figure(figsize=(8, 5))
plt.hist(X[:, 0], bins=50, color='steelblue', edgecolor='white', alpha=0.8)
plt.xlabel('Median Income')
plt.ylabel('Frequency')
plt.title('Distribution of Median Income')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# 3. Scatter plot: Actual vs Predicted House Prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, s=30, c='steelblue', edgecolors='none')
max_val = max(y_test.max(), y_pred.max())
min_val = min(y_test.min(), y_pred.min())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual House Price (in $100,000)')
plt.ylabel('Predicted House Price (in $100,000)')
plt.title('Actual vs Predicted House Prices')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 4. Residual plot: Prediction errors (Actual - Predicted)
residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
plt.scatter(y_pred, residuals, alpha=0.5, s=30, c='steelblue', edgecolors='none')
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted House Price (in $100,000)')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residual Plot: Prediction Errors')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 5. Correlation heatmap of all features
df = pd.DataFrame(X, columns=feature_names)
df['MedHouseVal'] = y
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.title('Correlation Heatmap of California Housing Features')
plt.tight_layout()
plt.show()

# ============ INTERACTIVE PREDICTION ============

print(" California House Price Prediction")

while True:
    try:
        MedInc = float(input("Enter Median Income: "))
        HouseAge = float(input("Enter House Age: "))
        AveRooms= float(input("Enter Average Rooms: "))
        AveBedrms= float(input("Enter Average Bedrooms: "))
        Population = float(input("Enter Population: "))
        AveOccup = float(input("Enter AveOccup: "))
        Latitude = float(input("Enter Latitude: "))
        Longitude = float(input("Enter Longitude: "))
        features = [[ MedInc,HouseAge , AveRooms, AveBedrms , Population,AveOccup, Latitude, Longitude]]
        features_scaled = scaler.transform(features)
        predicted_price = model.predict(features_scaled)[0]
        print(f"\nEstimated House Price: {predicted_price:.3f}")

    except ValueError:
        print("Please enter valid numeric values.")


