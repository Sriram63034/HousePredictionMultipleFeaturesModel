🏠 California Housing Price Prediction
Multiple Linear Regression (Machine Learning Project)
📌 Project Overview

This project implements a Multiple Linear Regression model to predict house prices using the California Housing dataset.

The model is trained using all available features and allows users to input house details to get a predicted price.

This project demonstrates a complete end-to-end Machine Learning pipeline including:

Data loading

Train-test splitting

Feature scaling

Model training

Model evaluation

Interactive user prediction system

📊 Dataset Used

California Housing Dataset (from Scikit-learn)

Features used:

Median Income (MedInc)

House Age (HouseAge)

Average Rooms (AveRooms)

Average Bedrooms (AveBedrms)

Population (Population)

Average Occupancy (AveOccup)

Latitude (Latitude)

Longitude (Longitude)

Target Variable:

Median House Value (MedHouseVal)

Note: The target value is in units of $100,000.

⚙️ Technologies Used

Python

NumPy

Scikit-learn

StandardScaler

LinearRegression

Git & GitHub

🧠 Machine Learning Workflow

Load dataset

Split into training and testing sets

Apply feature scaling using StandardScaler

Train LinearRegression model

Evaluate using:

Mean Squared Error (MSE)

Root Mean Squared Error (RMSE)

R² Score

Accept user input for real-time prediction

📈 Model Performance

Example Results:

MSE: 0.55

RMSE: 0.74

R² Score: 0.57

The model explains approximately 57% of the variance in house prices.

This serves as a strong baseline model for regression.

💻 How to Run the Project

Clone the repository:

git clone <your-repository-link>

Install dependencies:

pip install numpy scikit-learn

Run the program:

python CaliforniaHouse.py

Enter house feature values when prompted.

🚀 Future Improvements

Add feature engineering

Implement Polynomial Regression

Try advanced models (Random Forest, Gradient Boosting)

Convert to a web application (Flask/Django/Streamlit)

Save and load trained model using joblib

🎯 Learning Outcomes

Through this project, I learned:

How to build a regression model from scratch

Importance of train-test split

Feature scaling and preprocessing

Evaluating model performance

Preventing data leakage

Building an interactive ML prediction system

👨‍💻 Author

Devana Sriram
B.Tech CSE (AI & ML)
Aspiring AI Engineer
