from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error

#Load the California housing dataset
X,y  = fetch_california_housing(return_X_y=True)
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


