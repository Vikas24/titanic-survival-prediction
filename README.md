# 🚢 Titanic Survival Prediction using Machine Learning

## 📌 Project Overview
This project predicts whether a passenger survived the Titanic disaster using
Machine Learning techniques.  
It demonstrates a **complete ML workflow** including data analysis,
preprocessing, class imbalance handling, model training, hyperparameter tuning,
and detailed evaluation.

The project is structured following **industry best practices** for ML projects
using Git, GitHub, and modular Python scripts.

---

## 🧠 Key Concepts Implemented
- Exploratory Data Analysis (EDA)
- Data Cleaning & Feature Engineering
- Logistic Regression
- Handling Class Imbalance using SMOTE
- Hyperparameter Tuning with GridSearchCV
- Cross-Validation using ROC-AUC
- Model Evaluation:
  - Confusion Matrix
  - ROC Curve
  - Precision–Recall Curve
  - Threshold Analysis

---

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- imbalanced-learn (SMOTE)
- Jupyter Notebook
- Git & GitHub

---

## 📂 Dataset
- **Source:** Kaggle – Titanic Dataset
- **Target Variable:** `Survived`
- **Features Used:**
  - Pclass
  - Sex
  - Age
  - Fare
  - SibSp
  - Parch
  - Embarked

---

## 📊 Exploratory Data Analysis (EDA)
EDA was performed to understand:
- Survival distribution
- Gender-wise survival comparison
- Passenger class impact on survival
- Age distribution
- Missing values and correlations

EDA notebooks are available in the `notebooks/` folder.

---

## ⚙️ Data Preprocessing
- Dropped irrelevant columns
- Handled missing values
- Encoded categorical features
- Feature scaling using StandardScaler
- Saved processed dataset for reproducibility

---

## 🤖 Model Training
- Algorithm: **Logistic Regression**
- Train-test split with stratification
- Hyperparameter tuning using **GridSearchCV**
- Evaluation metric: **ROC-AUC**

---

## ⚖️ Handling Class Imbalance
The dataset is imbalanced.
**SMOTE (Synthetic Minority Oversampling Technique)** was applied **only on the training data**
to avoid data leakage and improve recall and F1-score.

---

## 🔁 Cross-Validation
- 5-fold cross-validation
- ROC-AUC used as the scoring metric
- Performance stability verified across folds

---

## 📈 Model Evaluation
- Accuracy, Precision, Recall, F1-score
- Confusion Matrix
- ROC Curve
- Precision–Recall Curve
- Threshold vs Precision–Recall analysis

Evaluation plots are saved in the `results/` folder.

---

## 📁 Project Structure
titanic-survival-prediction/
│
├── data/
│ ├── raw/
│ └── processed/
│
├── notebooks/
│ ├── 01_eda.ipynb
│ ├── 02_preprocessing.ipynb
│ └── 03_model_training.ipynb
│
├── src/
│ ├── preprocessing.py
│ ├── model.py
│ ├── evaluation.py
│ └── train.py
│
├── results/
│
├── README.md
├── requirements.txt
└── .gitignore


---

## ▶️ How to Run the Project
```bash
git clone https://github.com/yourusername/titanic-survival-prediction.git
cd titanic-survival-prediction
pip install -r requirements.txt
python src/train.py

👤 Author

Vikas Kumar
B.Tech Computer Science
Interested in Data Science & Machine Learning
