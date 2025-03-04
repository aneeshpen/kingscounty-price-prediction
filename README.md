# King's County Housing Price Prediction

This repository contains a Jupyter Notebook (`kingcounty.ipynb`) focused on analyzing housing data from King's County. The analysis leverages Python libraries like Pandas, NumPy, Seaborn, and Matplotlib for data processing, visualization, and exploration.

---

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Code Explanation](#code-explanation)
  - [Data Preprocessing](#data-preprocessing)
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
- [How to Run](#how-to-run)
- [Outputs](#outputs)
- [License](#license)

---

## Overview

The goal of this project is to build a machine learning model capable of predicting house prices based on various features such as the number of bedrooms, bathrooms, grade of the house, and other properties. The dataset consists of two files: one for training and another for testing. The code applies multiple regression techniques, evaluates their performance, and provides comparison metrics.

---

## Requirements

To run this project, you will need Python and the following libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `xgboost`
- `sklearn`

You can install the required libraries using the following command:

```bash
pip install numpy pandas matplotlib seaborn xgboost scikit-learn
```

---

## Dataset

The dataset consists of two CSV files:

1. `df_train.csv`: Contains the training data with features such as house price, number of bedrooms, grade, nice view, etc.
2. `df_test.csv`: Contains the test data, which is used for final model evaluation.

Each dataset includes the following columns:
- **price**: The price of the house (target variable).
- **bedrooms**: The number of bedrooms in the house.
- **real_bathrooms**: The number of real bathrooms.
- **has_lavatory**: Whether the house has a lavatory (boolean).
- **grade**: The grade of the house (a scale of quality).
- **nice_view**: Whether the house has a nice view (boolean).
- **date**: The date when the data was recorded.

You can adjust the file paths in the script to match the location of your datasets.

---

## Code Explanation

### Data Preprocessing

1. **Loading the Data**:
   - The training and test datasets are read into pandas DataFrames using `pd.read_csv()`.

2. **Handling Missing Values**:
   - The script checks for missing values in both the training and test datasets using `isnull().sum()` and `duplicated().sum()` to identify any issues with the data.

3. **Feature Engineering**:
   - The **date** columns are converted to datetime objects using `pd.to_datetime()`.
   - A new feature `total_bathrooms` is created by summing the `real_bathrooms` and `has_lavatory` columns.

4. **Categorical Encoding**:
   - The categorical columns are encoded into numerical values using `LabelEncoder` to make them suitable for model training.

5. **Feature Selection**:
   - The `numerical_cols` and `categorical_cols` lists store the column names of numerical and categorical features, respectively.

6. **Data Merging**:
   - The training and test datasets are concatenated into a single DataFrame (`df`) for easier manipulation.

---

### Exploratory Data Analysis (EDA)

1. **Price Distribution**:
   - A histogram with a Kernel Density Estimate (KDE) is plotted to visualize the distribution of house prices.

2. **Average Price by Categories**:
   - The average price of houses is computed for various categorical features, such as:
     - Number of bedrooms
     - Grade of the house
     - Whether the house has a nice view

---

### Model Training

Three regression models are used for predicting house prices:
1. **Linear Regression (LR)**
2. **Random Forest Regressor (RF)**
3. **XGBoost Regressor (XGB)**

The code splits the data into training and validation sets using `train_test_split()`, then trains each model on the training data.

---

### Model Evaluation

The models are evaluated on the validation and test sets using the following metrics:
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**
- **R-squared (R²)**

The results for each model are printed, comparing the performance across different evaluation metrics.

---

## How to Run

1. **Dataset File Paths**:
   Ensure the paths to the `df_train.csv` and `df_test.csv` files are correctly specified in the code.

2. **Run the Script**:
   Once the datasets are set up and dependencies are installed, you can run the script using the following command:

   ```bash
   python house_price_prediction.py
   ```

3. **View Results**:
   The script will print the performance metrics for each model on both the validation and test datasets.

---

## Outputs

The model evaluation for each regression algorithm will display the following metrics:

### Example Output:

```
Random Forest validation prediction
MSE:  120000000.0
RMSE:  10954.0
MAE:  8000.0
R2:  0.85
--------------------------------------------------
Linear Regression validation prediction
MSE:  150000000.0
RMSE:  12247.0
MAE:  9000.0
R2:  0.80
--------------------------------------------------
XGBoost validation prediction
MSE:  115000000.0
RMSE:  10717.0
MAE:  7500.0
R2:  0.87
--------------------------------------------------
```

---

