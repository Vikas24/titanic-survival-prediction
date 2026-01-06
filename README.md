# 🚢 Titanic Survival Prediction

An end-to-end machine learning project that predicts passenger survival on the Titanic dataset using **Logistic Regression**, with a strong focus on **reproducibility**, **evaluation**, and **clean project structure**.

---

## 📂 Project Structure

```text
titanic-survival-prediction/
│
├── data/
│   ├── raw/                # Original dataset
│   └── processed/          # Preprocessed dataset
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model_training.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── evaluation.py
│   └── train.py
│
├── results/                # Evaluation plots
├── requirements.txt
└── README.md

🔍 Exploratory Data Analysis (EDA)

Key insights derived from data exploration:

Female passengers had significantly higher survival rates than males

Passengers in higher classes (Pclass 1) had better survival chances

Age and Fare showed meaningful relationships with survival

Missing values in Age and Embarked required preprocessing

⚙️ Data Preprocessing

Missing Age values filled using median

Missing Embarked values filled using mode

Categorical features encoded using one-hot encoding

Feature scaling applied for Logistic Regression

SMOTE applied on training data to handle class imbalance

🤖 Model Training

Algorithm: Logistic Regression

Hyperparameter tuning using GridSearchCV

Optimal parameters selected based on cross-validated performance

Best Hyperparameters:

C = 0.1
penalty = l2
solver = liblinear

📊 Model Evaluation
Evaluation Metrics

Accuracy

Precision, Recall, F1-score

ROC-AUC

Precision–Recall Curve

Cross-Validation ROC-AUC

Results Summary

Accuracy: ~80%

Mean Cross-Validation ROC-AUC: ~0.85

Precision–Recall analysis provided better insight into minority class performance

SMOTE improved recall without significantly reducing precision

Model performance remained stable across folds

📈 Evaluation Visualizations

The following plots are generated and saved in the results/ directory:

Confusion Matrix

ROC Curve

Precision–Recall Curve

Precision–Recall vs Threshold Curve

🧠 Environment-Independent Execution

This project is designed to run seamlessly across different environments:

Local machine (VS Code)

Google Colab

Any Unix-based system

Key design decisions:

Dynamic project root resolution using __file__

No hard-coded paths

Automatic creation of required directories

This ensures reproducibility and portability.