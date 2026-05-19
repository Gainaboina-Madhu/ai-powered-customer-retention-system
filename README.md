# ai-powered-customer-retention-system

# <div align="center">🚀 AI-Powered Customer Retention Prediction System</div>

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge\&logo=python)

![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-orange?style=for-the-badge\&logo=scikit-learn)
![Logistic Regression](https://img.shields.io/badge/Model-Logistic_Regression-success?style=for-the-badge)


</div>

---

<div align="center">


## Predict customer churn using Machine Learning, SMOTE balancing, Feature Engineering, Hyperparameter Tuning, and Flask Deployment.

</div>


---

# 📸 Project Preview

<div align="center">

![Banner](images/banner.png)

</div>

---

# 📖 Abstract

Customer retention plays a critical role in maintaining business profitability and long-term growth, especially in highly competitive industries such as telecommunications, banking, and e-commerce. Customer churn occurs when customers discontinue a company’s services, resulting in revenue loss and increased customer acquisition costs.

This project presents an advanced **AI-Powered Customer Retention Prediction System** developed using Machine Learning techniques to accurately identify customers who are likely to churn.

The project follows a complete end-to-end Machine Learning pipeline including:

* Data Cleaning
* Missing Value Handling
* Variable Transformation
* Outlier Handling
* Feature Engineering
* Categorical Encoding
* Feature Scaling
* Data Balancing using SMOTE
* Hyperparameter Tuning
* Model Evaluation
* Flask Deployment

Multiple machine learning algorithms were trained and evaluated, including:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Decision Tree
* Naïve Bayes
* Random Forest
* AdaBoost
* Gradient Boosting
* XGBoost
* Support Vector Machine (SVM)

Among all models, **Logistic Regression** achieved the best performance and was selected as the final deployment model.

The final model was integrated into a Flask-based web application that allows users to enter customer details and receive real-time churn predictions with retention probability.

---

# 🎯 Project Objectives

✅ Predict customer churn accurately
✅ Improve customer retention strategies
✅ Reduce revenue loss
✅ Help businesses identify high-risk customers
✅ Build a real-world deployable AI system
✅ Support data-driven business decisions

---

# 🧠 Technologies Used

| Technology       | Purpose                     |
| ---------------- | --------------------------- |
| Python           | Core Programming Language   |
| Flask            | Backend Web Framework       |
| Scikit-learn     | Machine Learning Algorithms |
| Pandas           | Data Analysis               |
| NumPy            | Numerical Operations        |
| Matplotlib       | Data Visualization          |
| Seaborn          | Statistical Visualization   |
| XGBoost          | Boosting Algorithm          |
| HTML/CSS         | Frontend Design             |
| Pickle           | Model Serialization         |
| Imbalanced-learn | SMOTE Balancing             |
| Jupyter Notebook | Model Development           |

---

# 🏗️ System Architecture

<div align="center">

![Architecture](images/architecture.png)

</div>

---

# 📊 Dataset Overview

| Attribute       | Details                      |
| --------------- | ---------------------------- |
| Dataset Name    | Telco Customer Churn Dataset |
| Total Records   | 7043 Customers               |
| Total Features  | 21 Columns                   |
| Target Variable | Churn                        |
| Problem Type    | Binary Classification        |

---

# 📁 Dataset Features

## 👤 Customer Demographics

* Customer ID
* Gender
* Senior Citizen
* Partner
* Dependents

## 📞 Services Subscribed

* Phone Service
* Multiple Lines
* Internet Service
* Online Security
* Online Backup
* Device Protection
* Tech Support
* Streaming TV
* Streaming Movies

## 💳 Account Information

* Tenure
* Contract
* Paperless Billing
* Payment Method
* Monthly Charges
* Total Charges

## 🎯 Target Variable

* Churn (Yes / No)

---

# 🔄 Complete Machine Learning Workflow

<div align="center">

![Workflow](images/workflow.png)

</div>

```text
Dataset Collection
        ↓
Data Cleaning
        ↓
Handling Missing Values
        ↓
Feature Engineering
        ↓
Variable Transformation
        ↓
Outlier Handling
        ↓
Feature Selection
        ↓
Categorical Encoding
        ↓
Data Balancing (SMOTE)
        ↓
Feature Scaling
        ↓
Model Training
        ↓
Hyperparameter Tuning
        ↓
Model Evaluation
        ↓
Flask Deployment
```

---

# 📈 Exploratory Data Analysis (EDA)

The project includes detailed visual analysis to understand customer behavior and churn patterns.

## 📊 Visualizations Included

* Count Plots
* Pie Charts
* Histograms
* Heatmaps
* Box Plots
* Correlation Matrix
* ROC Curves
* Distribution Plots

---

## 📸 EDA Results

<div align="center">

### Customer Churn Distribution

![EDA1](images/churn_distribution.png)

### Correlation Heatmap

![EDA2](images/correlation_heatmap.png)

### ROC Curve

![EDA3](images/roc_curve.png)

</div>

---

# 🛠️ Feature Engineering

Feature engineering was performed to improve data quality and increase model performance.

## 🔹 Missing Value Handling

Multiple imputation techniques were evaluated:

* Mean Imputation
* Median Imputation
* Mode Imputation
* KNN Imputation
* Random Sample Imputation
* Forward Fill
* Backward Fill
* Interpolation

### ✅ Final Selected Method

# Mode Imputation

Reason:

* Preserved original data distribution
* Produced minimum deviation score
* Improved model consistency

---

# 🔄 Variable Transformation

Transformation techniques applied:

| Transformation     | Purpose                         |
| ------------------ | ------------------------------- |
| Log Transformation | Reduce Skewness                 |
| Box-Cox            | Stabilize Variance              |
| Yeo-Johnson        | Handle Positive/Negative Values |
| Reciprocal         | Compress Large Values           |
| Square Root        | Moderate Skew Reduction         |
| Exponential        | Correct Negative Skew           |
| Cube Root          | Mild Skew Handling              |
| Arcsin             | Proportion Data Transformation  |

## ✅ Final Selected Transformations

| Feature         | Transformation |
| --------------- | -------------- |
| Tenure          | Yeo-Johnson    |
| Monthly Charges | Box-Cox        |
| Total Charges   | Yeo-Johnson    |

---

# 📌 Handling Outliers

Outlier handling techniques evaluated:

* IQR Trimming
* IQR Capping
* Mean-Standard Deviation Capping
* Quantile Capping
* Gaussian Winsorization

## ✅ Final Techniques

| Feature         | Technique        |
| --------------- | ---------------- |
| Tenure          | IQR Capping      |
| Monthly Charges | Mean-Std Capping |
| Total Charges   | Mean-Std Capping |

---

# 🧪 Feature Selection

Feature selection techniques used:

## 🔹 Filter Methods

* Constant Feature Removal
* Quasi-Constant Feature Removal

## 🔹 Statistical Methods

* Pearson Correlation
* Hypothesis Testing

### Benefits

✅ Reduced dimensionality
✅ Improved model efficiency
✅ Reduced overfitting
✅ Faster training speed

---

# 🔤 Categorical Encoding

| Encoding Technique | Purpose                   |
| ------------------ | ------------------------- |
| One-Hot Encoding   | Nominal Variables         |
| Ordinal Encoding   | Ordered Categories        |
| Label Encoding     | Target Variable           |
| Target Encoding    | High Cardinality Features |

---

# ⚖️ Data Balancing

The dataset was highly imbalanced.

## 📊 Before Balancing

| Class     | Samples |
| --------- | ------- |
| Non-Churn | 4118    |
| Churn     | 1516    |

---

## 🔹 Balancing Techniques Used

* Random Over Sampling
* Random Under Sampling
* SMOTE

### ✅ Final Selected Technique

# SMOTE (Synthetic Minority Over-sampling Technique)

### Why SMOTE?

✅ Generates synthetic minority samples
✅ Reduces model bias
✅ Improves recall score
✅ Better churn prediction performance

---

# 📏 Feature Scaling

| Scaling Technique | Description           |
| ----------------- | --------------------- |
| StandardScaler    | Z-score normalization |
| MinMaxScaler      | Scale between 0 and 1 |
| RobustScaler      | Handles outliers      |

### ✅ Final Scaling Method

# StandardScaler (Z-Score Normalization)

Reason:

* Faster convergence
* Better stability
* Improved model performance

---

# 🤖 Model Training

Multiple machine learning algorithms were trained and evaluated.

## 📌 Models Used

| Model                  |
| ---------------------- |
| Logistic Regression    |
| K-Nearest Neighbors    |
| Naïve Bayes            |
| Decision Tree          |
| Random Forest          |
| AdaBoost               |
| Gradient Boosting      |
| XGBoost                |
| Support Vector Machine |

---

# 🏆 Best Model — Logistic Regression

After comparing all models using ROC-AUC and classification metrics, **Logistic Regression** achieved the best overall performance.

## ✅ Why Logistic Regression?

* Strong classification performance
* High precision and recall
* Better class separation
* Stable predictions
* Effective binary classification

---

# ⚙️ Hyperparameter Tuning

Hyperparameter tuning was performed using:

# GridSearchCV (10-Fold Cross Validation)

## 🔹 Best Parameters

| Parameter      | Value     |
| -------------- | --------- |
| C              | 100       |
| Penalty        | l1        |
| Solver         | liblinear |
| Max Iterations | 500       |
| Class Weight   | balanced  |

---

# 📊 Model Performance

<div align="center">

![Performance](images/model_performance.png)

</div>

## 📌 Performance Metrics

| Metric        | Score     |
| ------------- | --------- |
| Accuracy      | 76.65%    |
| Precision     | 0.92      |
| Recall        | 0.76      |
| F1-Score      | 0.83      |
| ROC-AUC Score | Excellent |

---

# 📉 Confusion Matrix

<div align="center">

![Confusion Matrix](images/confusion_matrix.png)

</div>

---

# 🌐 Flask Web Application

## 📸 Application Preview

<div align="center">

![Frontend](images/frontend.png)

</div>

---

# 💻 Frontend

The frontend was developed using:

* HTML
* CSS
* Flask Templates

## Features

✅ User-friendly interface
✅ Real-time prediction
✅ Probability display
✅ Interactive forms
✅ Responsive design

---

# ⚡ Backend

The backend was developed using Flask.

## Backend Responsibilities

* Request Handling
* Data Preprocessing
* Feature Encoding
* Feature Scaling
* Model Prediction
* Probability Calculation
* Result Rendering

---

# 📂 Project Folder Structure

```bash
AI-Powered-Customer-Retention-Prediction-System/
│
├── static/
│   ├── css/
│   ├── images/
│
├── templates/
│   ├── index.html
│
├── model/
│   ├── churn_model.pkl
│   ├── scaler.pkl
│   ├── encoder.pkl
│
├── notebook/
│   ├── churn_prediction.ipynb
│
├── dataset/
│   ├── telco_customer_churn.csv
│
├── app.py
├── requirements.txt
├── README.md
```

---

# ⚡ Installation Guide

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/AI-Powered-Customer-Retention-Prediction-System.git
```

---

## 2️⃣ Navigate to Project Folder

```bash
cd AI-Powered-Customer-Retention-Prediction-System
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Run Flask Application

```bash
python app.py
```

---

## 5️⃣ Open Browser

```bash
http://127.0.0.1:5000
```

---

# 📌 Prediction Output

| Prediction            | Result   |
| --------------------- | -------- |
| Churn Risk            | Low Risk |
| Retention Probability | 91.73%   |
| Churn Probability     | 8.27%    |

---

# 🚀 Future Enhancements

* Deep Learning Integration (ANN, CNN, LSTM)
* Real-Time Prediction Pipelines
* Docker Deployment
* Cloud Deployment
* Interactive Dashboard
* Explainable AI (XAI)
* Automated ML Pipeline
* Real-Time Customer Analytics
* Streamlit Deployment

---

# 💼 Business Impact

This project helps organizations:

✅ Reduce customer churn
✅ Improve customer retention
✅ Increase profitability
✅ Enable proactive business strategies
✅ Improve customer satisfaction

---

# 📚 References

1. Kaggle — Telco Customer Churn Dataset
2. Scikit-learn Documentation
3. Pandas Documentation
4. NumPy Documentation
5. Flask Documentation
6. XGBoost Documentation
7. Imbalanced-learn Documentation
8. Machine Learning Mastery
9. Towards Data Science

---

# 👨‍💻 Author

<div align="center">

## Gainaboina Madhu

### Machine Learning & Deep Learning Enthusiast

</div>

---

# ⭐ Final Conclusion

The **AI-Powered Customer Retention Prediction System** successfully demonstrates the practical implementation of Machine Learning for customer churn prediction using a complete end-to-end pipeline.

The project combines:

* Advanced preprocessing techniques
* Feature engineering
* Data balancing
* Hyperparameter tuning
* Model evaluation
* Flask deployment

The final system enables organizations to proactively identify customers at risk of churn and take strategic actions to improve retention, customer satisfaction, and business profitability.

---

