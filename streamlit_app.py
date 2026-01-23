import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# Page config
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

# Load model and scaler
@st.cache_resource
def load_artifacts():
    with open("model/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artifacts()

# Sidebar
st.sidebar.title("🚢 About This App")
st.sidebar.markdown("""
This app predicts whether a passenger would survive the Titanic disaster  
using a **Logistic Regression model** trained on the Kaggle Titanic dataset.

**Steps:**
1. Enter passenger details  
2. Click *Predict Survival*  
3. View prediction and probability  

Developed by **Vikas Kumar**
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Info")
st.sidebar.markdown("- Algorithm: Logistic Regression")
st.sidebar.markdown("- Accuracy: ~80%")
st.sidebar.markdown("- ROC-AUC: ~0.85")

# Main Title
st.title("🚢 Titanic Survival Prediction System")
st.write("Fill in the passenger details and click **Predict Survival**.")

st.markdown("---")

# Input section
with st.container():
    st.subheader("🔹 Passenger Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        pclass = st.selectbox("Passenger Class", [1, 2, 3], help="1 = Upper, 2 = Middle, 3 = Lower")
        sex = st.radio("Sex", ["male", "female"])
        age = st.slider("Age", 1, 80, 25)

    with col2:
        sibsp = st.number_input("Siblings / Spouses Aboard", 0, 8, 0)
        parch = st.number_input("Parents / Children Aboard", 0, 6, 0)
        embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

    with col3:
        fare = st.slider("Fare Paid", 0.0, 500.0, 32.0)
        st.markdown("#### Quick Tips")
        st.info("""
        - Females had higher survival  
        - 1st class had better chances  
        - Very young children had higher survival  
        """)

st.markdown("---")

# Predict button
predict_btn = st.button("🚀 Predict Survival", use_container_width=True)

if predict_btn:
    # Build raw input dataframe (same as training before get_dummies)
    input_df = pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [sex],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Embarked": [embarked]
    })

    # Apply one-hot encoding
    input_df = pd.get_dummies(input_df, drop_first=True)

    # Align with training columns
    training_columns = scaler.feature_names_in_
    input_df = input_df.reindex(columns=training_columns, fill_value=0)

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    st.markdown("## 📊 Prediction Result")

    colA, colB = st.columns(2)

    with colA:
        if prediction == 1:
            st.success("🎉 Passenger is likely to **SURVIVE**")
        else:
            st.error("❌ Passenger is likely to **NOT SURVIVE**")

        st.markdown(f"**Survival Probability:** `{prob:.2f}`")

    with colB:
        st.markdown("### 🔎 Input Summary")
        st.write(input_df)

    # Probability bar
    st.markdown("### 📈 Survival Probability Visualization")
    st.progress(int(prob * 100))

    if prob > 0.7:
        st.success("High chance of survival")
    elif prob > 0.4:
        st.warning("Moderate chance of survival")
    else:
        st.error("Low chance of survival")

st.markdown("---")

st.caption("Built with ❤️ using Streamlit and scikit-learn")
