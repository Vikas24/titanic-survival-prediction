# 🚢 Titanic Survival Prediction

An end-to-end machine learning project to predict passenger survival on the Titanic dataset using **Logistic Regression**, with emphasis on **clean code**, **robust evaluation**, and **environment-independent execution**.
## 🚀 Live Demo

This project is deployed as an interactive web application using **Streamlit Cloud**.

👉 **Live App:** https://vikas24-titanic-survival-prediction.streamlit.app  

Users can enter passenger details (class, age, sex, fare, etc.) and get real-time survival predictions along with probability scores.


---


---
## 📂 Project Structure

```text
titanic-survival-prediction/
│
├── data/
│   ├── raw/                # Original dataset
│   └── processed/          # Preprocessed dataset + scaler
│
├── model/
│   ├── model.pkl           # Trained Logistic Regression model
│   └── scaler.pkl          # Saved StandardScaler for inference
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
├── results/                # Saved evaluation plots
├── app.py                  # Streamlit web application
├── requirements.txt
└── README.md
```

## 🔍 Exploratory Data Analysis (EDA)

**Key insights from data exploration:**

- Female passengers had significantly higher survival rates than males  
- Passengers in higher classes (**Pclass 1**) had better survival chances  
- **Age** and **Fare** showed meaningful relationships with survival  
- Missing values were identified in **Age** and **Embarked**  
- These observations guided preprocessing and feature engineering  

---

## ⚙️ Data Preprocessing

- Missing **Age** values filled using **median**  
- Missing **Embarked** values filled using **mode**  
- Categorical features converted to numeric using **one-hot encoding**  
- Feature scaling applied to improve **Logistic Regression** performance  
- **SMOTE** applied on training data to handle class imbalance  

---

## 🤖 Model Training

- Algorithm used: **Logistic Regression**  
- Hyperparameter tuning performed using **GridSearchCV**  
- Model selected based on **cross-validated performance**  

### Best Hyperparameters
- **C:** 0.1  
- **Penalty:** l2  
- **Solver:** liblinear  

---

## 📊 Model Evaluation

### Evaluation Metrics
- Accuracy  
- Precision, Recall, F1-score  
- ROC-AUC  
- Precision–Recall Curve  
- Cross-Validation ROC-AUC  

### Performance Summary
- **Accuracy:** ~80%  
- **Mean Cross-Validation ROC-AUC:** ~0.85  
- Precision–Recall analysis provided better insight into minority class performance  
- SMOTE improved recall without significantly reducing precision  
- Model performance remained stable across multiple folds  

---

## 📈 Evaluation Visualizations

The following plots are generated and saved in the `results/` directory:

- Confusion Matrix  
- ROC Curve  
- Precision–Recall Curve  
- Precision–Recall vs Threshold Curve  

---

## 🧠 Environment-Independent Execution

The project is designed to run consistently across different environments:

- Local machine (VS Code)  
- Google Colab  
- Any Unix-based system  

**Key design choices:**
- Dynamic project root resolution using `__file__`  
- No hard-coded file paths  
- Automatic creation of required directories  
- Robust handling of missing files and outputs  

This ensures **reproducibility, portability, and reliability**.

## ▶️ Run the Interactive Web App Locally

To run the Streamlit app on your local machine:

```bash
pip install -r requirements.txt
streamlit run app.py
```
## 📦 Deployment

The application is deployed using **Streamlit Cloud** directly from this GitHub repository.

Deployment steps:
1. Push code to GitHub  
2. Connect repository on Streamlit Cloud  
3. Select `app.py` as the main file  
4. Deploy and share the public URL  

This enables easy access to the model through a browser without any local setup.

