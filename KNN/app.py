import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="KNN Classifier",
    page_icon="📊",
    layout="wide"
)

st.title("🔍 KNN Classification App")
st.write("Upload a CSV file to perform KNN classification")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    st.success("✅ File loaded successfully")

    # Preview
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    # Target column selection
    target_col = st.selectbox("🎯 Select Target Column", df.columns)

    if target_col:
        # Split features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Encode categorical features
        for col in X.select_dtypes(include=["object"]):
            X[col] = LabelEncoder().fit_transform(X[col])

        # Encode target if needed
        if y.dtype == "object":
            y = LabelEncoder().fit_transform(y)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Choose K
        k = st.slider("🔢 Select K value", min_value=1, max_value=15, value=5)

        # Train model
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)

        st.subheader("📊 Model Performance")
        st.success(f"✅ Accuracy: {accuracy:.2f}")
