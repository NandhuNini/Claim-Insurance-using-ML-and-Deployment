# Claim-Insurance-using-ML-and-Deployment

# 🚗 Insurance Claim Prediction using Logistic Regression

This project is a **machine learning classification system** that predicts whether an **insurance claim** will be valid or not.  
The workflow covers data preprocessing, model training, evaluation, and saving the trained model for future use.

---

## 📖 Table of Contents
1. [Project Overview](#project-overview)  
2. [Features](#features)  
3. [Dataset](#dataset)  
4. [Project Workflow](#project-workflow)  
5. [Installation](#installation)  
6. [Usage](#usage)  
7. [Code Walkthrough](#code-walkthrough)  
8. [Results](#results)  
9. [Future Improvements](#future-improvements)  
10. [License](#license)  
11. [Acknowledgments](#acknowledgments)

---

## 📌 Project Overview
Insurance companies need to evaluate claims efficiently. This project uses **Logistic Regression** to classify insurance claims based on claimant details.  

The notebook demonstrates:
- **Data cleaning** (handling missing values, dropping unnecessary columns)  
- **Feature scaling** with `StandardScaler`  
- **Model training & evaluation** using Logistic Regression  
- **Model saving** with `pickle`  

This project can be extended for real-world deployment in insurance systems.

---

## 🌟 Features
- ✅ Reads dataset (`claimants.csv`) using **Pandas**  
- ✅ Handles missing values with **mode** and **median imputation**  
- ✅ Drops irrelevant columns (`CASENUM`)  
- ✅ Splits dataset into **train/test sets**  
- ✅ Applies **StandardScaler** to standardize features  
- ✅ Trains a **Logistic Regression classifier**  
- ✅ Evaluates model accuracy  
- ✅ Saves the trained model into `model.pkl` using Pickle  

---

## 📂 Dataset
- **File used**: `claimants.csv`  

### Example Columns
| Column   | Description                          |
|----------|--------------------------------------|
| CASENUM  | Case number (removed during training) |
| CLMSEX   | Gender of claimant                   |
| CLMINSUR | Whether claimant had insurance       |
| SEATBELT | Seatbelt usage                       |
| CLMAGE   | Age of claimant                      |
| Target   | Claim valid or not (0/1)             |

---

## 🔄 Project Workflow
1. Data Import 
   ```python
   import pandas as pd
   df = pd.read_csv("claimants.csv")

2. Data Cleaning

Dropping irrelevant column:

df.drop(columns=["CASENUM"], inplace=True)


3. Handling missing values:

df.fillna({
    "CLMSEX": df.CLMSEX.mode()[0],
    "CLMINSUR": df.CLMINSUR.mode()[0],
    "SEATBELT": df.SEATBELT.mode()[0],
    "CLMAGE": df.CLMAGE.median()
}, inplace=True)


4. Train-Test Split

from sklearn.model_selection import train_test_split
X = df.drop("ATTORNEY", axis=1)   # Example target column
y = df["ATTORNEY"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


5.  Feature Scaling

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


6. Model Training

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)


7. Model Evaluation

from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


8. Saving Model

import pickle
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# ⚙️ Installation

Clone this repository:

git clone https://github.com/your-username/insurance-claim-prediction.git
cd insurance-claim-prediction


Install required dependencies:

pip install -r requirements.txt

# ▶️ Usage
Running the Notebook

Open Jupyter Notebook and run:

jupyter notebook insurance_claim_prediction.ipynb

Using the Trained Model

Once trained, the model is saved as model.pkl.
You can load it in Python:

import pickle
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Make predictions
sample_input = [[1, 1, 0, 35]]  # Example feature values
prediction = model.predict(sample_input)
print("Prediction:", prediction)

# 📊 Results

The Logistic Regression model achieved an accuracy score of 85%.

Dataset cleaned with missing value imputation.

Standardized features improved model stability.

# 🚀 Future Improvements

Add more ML models (Random Forest, XGBoost, Neural Networks)

Implement cross-validation for robust performance

Build a Streamlit/Flask web app for deployment

Enhance dataset with more real-world features

# 📜 License

This project is licensed under the MIT License.
See the LICENSE
 file for details.

🙌 Acknowledgments

Dataset: Academic dataset (claimants.csv)

Libraries used: NumPy, Pandas, Scikit-learn, Matplotlib
