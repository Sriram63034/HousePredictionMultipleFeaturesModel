# 🏠 California Housing Price Prediction

This project implements a **Machine Learning model using Multiple Linear Regression** to predict house prices based on features from the California Housing dataset.

## 📊 Project Overview

The goal of this project is to build a regression model that predicts the **median house value** based on several housing-related features such as income, location, population, and housing characteristics.

The project demonstrates a complete **end-to-end machine learning pipeline** including data preprocessing, model training, evaluation, visualization, and prediction.

---

## ⚙️ Technologies Used

* Python
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Git & GitHub

---

## 📂 Dataset

The dataset used is the **California Housing dataset** available in Scikit-learn.

Features used in the model:

* Median Income (`MedInc`)
* House Age (`HouseAge`)
* Average Rooms (`AveRooms`)
* Average Bedrooms (`AveBedrms`)
* Population (`Population`)
* Average Occupancy (`AveOccup`)
* Latitude (`Latitude`)
* Longitude (`Longitude`)

Target Variable:

* Median House Value (`MedHouseVal`)

Note: House prices are represented in units of **$100,000**.

---

## 🧠 Machine Learning Workflow

1. Load dataset from Scikit-learn
2. Split dataset into training and testing sets
3. Apply feature scaling using **StandardScaler**
4. Train a **Linear Regression** model
5. Evaluate model performance using:

   * Mean Squared Error (MSE)
   * Root Mean Squared Error (RMSE)
   * R² Score
6. Accept user input for predicting house price

---

## 📈 Model Performance

Example results:

* **MSE:** ~0.55
* **RMSE:** ~0.74
* **R² Score:** ~0.57

The model explains approximately **57% of the variance** in housing prices.

---

## 📊 Visualizations

This project also includes visualizations using **Matplotlib and Seaborn**, such as:

* Feature vs House Price scatter plots
* Histogram distributions of features
* Actual vs Predicted price comparison
* Residual plots
* Correlation heatmap

These visualizations help better understand the dataset and model performance.

---

## 🚀 How to Run the Project

Clone the repository:

```bash
git clone https://github.com/Sriram63034/MLLinearRegressionModelMultipleFeatures.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python CaliforniaHouse.py
```

Enter the requested feature values to predict house prices.

---

## 🎯 Learning Outcomes

Through this project, I learned:

* Building a **Machine Learning regression model**
* Data preprocessing and **feature scaling**
* Evaluating models using **MSE, RMSE, and R²**
* Creating **data visualizations**
* Using **Git and GitHub** for project version control

---

## 👨‍💻 Author

**Devana Sriram**
B.Tech CSE (AI & ML)
Aspiring AI Engineer

---


