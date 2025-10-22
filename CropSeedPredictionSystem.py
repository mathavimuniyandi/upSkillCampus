# CropSeedPredictionSystem.py
# Streamlit-based Crop Seed Prediction System
# Author: Mathavi M (Upskill Campus Internship Project)

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Crop Seed Prediction System", page_icon="🌾")

# -----------------------------------------------
# Title and Description
# -----------------------------------------------
st.title("🌾 Crop Seed Prediction System")
st.write("""
This web app predicts the most suitable crop seed based on environmental and soil parameters.
Developed by **Mathavi M** as part of the Upskill Campus Internship in
**Data Science & Machine Learning**.
""")

# -----------------------------------------------
# Sample Dataset Creation (for demo purpose)
# -----------------------------------------------
data = {
    "Temperature": [20, 25, 30, 35, 40, 22, 27, 32, 37, 42],
    "Humidity": [80, 70, 60, 55, 50, 82, 68, 62, 57, 49],
    "pH": [6.5, 7.0, 6.8, 5.5, 5.0, 6.9, 7.1, 6.4, 5.8, 5.2],
    "Rainfall": [200, 180, 160, 120, 100, 210, 170, 150, 130, 90],
    "Crop": ["Rice", "Rice", "Maize", "Maize", "Wheat", "Rice", "Maize", "Wheat", "Wheat", "Bajra"]
}
df = pd.DataFrame(data)

# -----------------------------------------------
# Train Model
# -----------------------------------------------
X = df[["Temperature", "Humidity", "pH", "Rainfall"]]
y = df["Crop"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Accuracy display
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# -----------------------------------------------
# User Input
# -----------------------------------------------
st.header("🔍 Enter Environmental Parameters")

temp = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=65.0)
ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
rain = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=150.0)

if st.button("Predict Best Crop Seed 🌱"):
    input_data = pd.DataFrame([[temp, humidity, ph, rain]], columns=X.columns)
    prediction = model.predict(input_data)[0]
    st.success(f"✅ Recommended Crop Seed: **{prediction}**")

# -----------------------------------------------
# Footer
# -----------------------------------------------
st.write("---")
st.write("Developed by **Mathavi M** | [LinkedIn Profile](https://www.linkedin.com/in/mathavi-m-3315a22a9/)")
st.caption("Upskill Campus Internship — Data Science & Machine Learning")

