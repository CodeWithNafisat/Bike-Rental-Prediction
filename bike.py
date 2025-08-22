import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib

# Load trained model
model = joblib.load("Bike.pkl")

FEATURE_NAMES = [
    'season', 'yr', 'mnth', 'holiday', 'weekday', 
    'workingday', 'weathersit', 'temp', 'atemp', 
    'hum', 'windspeed', 'Day']

st.title('Bike Count Prediction on Daily Basis')
st.image("bike.jpg", caption="Bike Sharing System", use_container_width=True)

with st.sidebar:
    st.write('Fill in your details to see the recommended bike count')

    season = st.number_input("Season (1 = Winter, 2 = Spring, 3 = Summer, 4 = Fall)", 1, 4, 1)
    yr = st.number_input("Year (0 = 2011, 1 = 2012)", 0, 1, 0)
    mnth = st.number_input("Month (1-12)", 1, 12, 1)
    holiday = st.number_input("Holiday (0 = No, 1 = Yes)", 0, 1, 0)
    weekday = st.number_input("Weekday (0 = Sunday ... 6 = Saturday)", 0, 6, 0)
    workingday = st.number_input("Working Day (0 = No, 1 = Yes)", 0, 1, 0)
    weathersit = st.number_input("Weather Situation (1 = Clear, 2 = Mist, 3 = Light Snow/Rain)", 1, 3, 1)
    temp = st.number_input("Normalized Temperature (0 to 1)", 0.0, 1.0, 0.5, step=0.01)
    atemp = st.number_input("Normalized Feeling Temperature (0 to 1)", 0.0, 1.0, 0.5, step=0.01)
    hum = st.number_input("Humidity (0 to 1)", 0.0, 1.0, 0.5, step=0.01)
    windspeed = st.number_input("Windspeed (0 to 1)", 0.0, 1.0, 0.5, step=0.01)
    Day = st.number_input("Day of Month (1-31)", 1, 31, 1)

if st.button('Submit'):
    # Arrange inputs into DataFrame (for SHAP)
    data = pd.DataFrame([[
        season, yr, mnth, holiday, weekday, 
        workingday, weathersit, temp, atemp, 
        hum, windspeed, Day
    ]], columns = FEATURE_NAMES)

    # Prediction
    prediction = model.predict(data)[0]
    st.success(f'Predicted number of bikes to rent: {int(prediction)}')
    
    # Load full dataset (for background reference)
    data2 = pd.read_csv(r"data\sample.csv")
    
    # Background dataset for SHAP (sample for speed)
    background_data = data2.sample(200, random_state=20)
    
    # ---- STEP 2: Create SHAP explainer ----
    explainer = shap.KernelExplainer(model.predict, background_data)
    
    # ---- STEP 3: Compute SHAP values for user input ----
    shap_values = explainer.shap_values(data)
    shap_values_instance = shap_values[0]
    
    # ---- STEP 4: Build SHAP Explanation object ----
    shap_exp = shap.Explanation(
        values        = shap_values_instance,
        base_values   = explainer.expected_value,
        data          = data.iloc[0],         # convert to 1D row
        feature_names = data.columns
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_exp, show=False)
    
    plt.title(
        f"SHAP Waterfall Plot\nPredicted Value: {model.predict(data)[0]:.2f}",
        fontsize=12
    )
    
    plt.figtext(
        0.5, -0.05,
        "This plot shows how each feature contributed to the prediction.\n"
        "Red bars increase the prediction, blue bars decrease it.\n"
        "The model starts from the baseline and adds/subtracts contributions\n"
        "to arrive at the final prediction.",
        ha="center", fontsize=10, wrap=True
    )
    
    plt.tight_layout()
    
    # ---- STEP 6: Render in Streamlit ----
    st.pyplot(fig)

    
