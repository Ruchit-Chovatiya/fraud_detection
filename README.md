# fraud_detection
A Streamlit app to detect Fraud using Machine Learning
# Credit Card Fraud Detection

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)

## Project Overview

Credit card fraud is a major challenge in the financial sector. Fraudulent transactions lead to billions of dollars in losses each year. Detecting these transactions accurately and quickly is essential.

This project builds a **machine learning model to detect fraudulent credit card transactions** using historical transaction data. The model learns patterns from legitimate and fraudulent transactions and predicts whether a new transaction is fraudulent.

---

## Dataset

The dataset used in this project is larger than GitHub’s recommended file size and therefore is not included in the repository.

Download the dataset from the following link:

Dataset Link:
[https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset]

After downloading, place the dataset inside:

```
data/creditcard.csv
```

Dataset characteristics:

* ~6.3 million financial transaction records
* CSV format dataset
* Contains transaction details such as type, amount, sender, and receiver balances
* Target column isFraud indicates fraudulent (1) or legitimate (0) transactions
* Highly imbalanced dataset, making fraud detection challenging
* Suitable for building machine learning classification models

---


## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Streamlit
* Jupyter Notebook

---

## Machine Learning Pipeline

1. Data Loading
2. Exploratory Data Analysis
3. Handling Class Imbalance
4. Feature Scaling
5. Model Training
6. Model Evaluation
7. Fraud Prediction

---

## Model Used

The following models were evaluated:

* Logistic Regression

---

## Evaluation Metrics

Due to **class imbalance**, accuracy alone is not sufficient. Therefore the following metrics are used:

* Confusion Matrix
* F1 Score
* Recall
* Precision

Example results:

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 0.95  |
| Precision | 0.92  |
| Recall    | 0.94  |
| F1 Score  | 0.97  |

---

## Author

**Ruchit Chovatiya**

Email: [chovatiyaruchit@gmail.com](mailto:chovatiyaruchit@gmail.com)
LinkedIn: https://linkedin.com/in/ruchit-chovatiya-7a320428a
GitHub: https://github.com/Ruchit-Chovatiya
