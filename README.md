# 🛡️ Fraud Detection using Machine Learning (Decision Tree)

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)

---

## 📌 Overview

This project implements machine learning models to detect and predict fraudulent credit card transactions.  
Using historical transaction data, the model is trained to identify patterns and anomalies indicative of fraud in real-time.

Model developed:
- 🌲 **Decision Tree Classifier**

The project simulates a real-world business case for a Credit Union to minimize fraud-related losses and maintain customer trust.

---

## 🧠 Key Features

- ✅ End-to-end Machine Learning Pipeline (Preprocessing → Training → Evaluation)
- 📈 Performance Metrics: Accuracy, Precision, Recall, F1 Score
- 📊 Confusion Matrix and Tree Visualizations
- ⚡ Overview of Decision Tree
- 🛡️ Clean GitHub repository (no large datasets included)

---

## 🛠 Technologies Used

- Python 3.12
- Scikit-Learn
- Pandas & NumPy
- Seaborn & Matplotlib
- Imbalanced-Learn (SMOTE)

---

## 📂 Project Structure

fraud-detection-decision-tree/ ├── data/ 
# Raw dataset (Kaggle source) ├── notebooks/ 
# Jupyter notebooks for EDA ├── reports/ 
# Model visualizations (confusion matrices, tree plots) ├── src/ 
# Source code (preprocessing, training, evaluation) ├── main.py 
# Main execution script ├── requirements.txt 
# Project dependencies └── README.md 
# This file


---

## 📈 Results

| Model                  | Accuracy | Precision | Recall | F1 Score |
|:------------------------|:---------|:----------|:-------|:---------|
| Decision Tree           | 96.6%    | 4.2%      | 85.8%  | 8.0%     |

✅ High recall was prioritized to minimize missed frauds — critical in fraud detection contexts.

---

## 📊 Visualizations

- Confusion Matrix for Decision Tree ✅
- Decision Tree Diagram ✅

_All visualizations are saved in the `/reports/` folder._

---

## 📦 Dataset

**Note:**  
The dataset (`creditcard.csv`) is not included in this repository due to GitHub size restrictions.  
You can download it separately from [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

---

## 🚀 How to Run

1. Clone the repo:
    ```bash
    git clone https://github.com/tbirch5/fraud-detection-decision-tree.git
    cd fraud-detection-decision-tree
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate       # macOS/Linux
    .\venv\Scripts\activate         # Windows
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the pipeline:
    ```bash
    python main.py
    ```

---

## 📫 Contact

If you liked this project, feel free to connect with me on [GitHub](https://github.com/tbirch5)!  
Let's collaborate and build more!

---

