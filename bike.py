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

st.title('**Bike Count Prediction on Daily Basis**')
st.image("bike.jpg", caption="Bike Sharing System", use_container_width=True)

with st.sidebar:
    st.write('**Fill in your details to see the recommended bike count**')

    season_map = {"Winter": 1, "Spring": 2, "Summer": 3, "Fall": 4}
    season_choice = st.selectbox("Season", ["Winter", "Spring", "Summer", "Fall"], index=0)
    season = season_map[season_choice]
    
    yr = st.number_input("Year (0 = 2011, 1 = 2012)", 0, 1, 0)

    mnth = st.number_input("Enter the month as a number (1=Jan, 2=Feb, ... 12=Dec)", min_value=1, max_value=12, value=1)

    holiday = st.number_input("Holiday (0 = No, 1 = Yes)", min_value=0, max_value=1, value=0)
    
    weekday_map = {"Sunday": 0, "Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4, "Friday": 5, "Saturday": 6}
    weekday_choice = st.selectbox("Weekday", ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"], index=0)
    weekday = weekday_map[weekday_choice]

    workingday = st.number_input("Working Day (0 = No, 1 = Yes)", min_value=0, max_value=1, value=0)

    weather_situation_map = {"Clear" : 1, "Mist" : 2, "Light snow/Rain" : 3}
    weather_choice = st.selectbox("Weather", ["Clear", "Mist", "Light Snow/Rain"])
    weathersit = weather_situation_map[weather_choice]

    temp = st.number_input("Normalized Temperature (0 to 1)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    atemp = st.number_input("Normalized Feeling Temperature (0 to 1)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    hum = st.number_input( "Humidity (0 to 1)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    
    windspeed = st.number_input("Windspeed (0 to 1)", min_value=0.0, max_value=1.0, value=0.5, step=0.01,
        help="Enter the normalized wind speed (0=calm, 1=very windy)"
    )

    Day = st.number_input(
        "Day of Month (1-31)", min_value=1, max_value=31, value=1,
        help="Enter the day of the month (1-31)"
    )

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
    
    data2 = pd.read_csv(r"sample.csv")
    background_data = data2.sample(200, random_state=20)
    
    explainer = shap.KernelExplainer(model.predict, background_data)
    
    shap_values = explainer.shap_values(data)
    shap_values_instance = shap_values[0]
    
    shap_exp = shap.Explanation(
        values        = np.round(shap_values_instance).astype(int),
        base_values   = int(round(explainer.expected_value)),
        data          = data.iloc[0],         # convert to 1D row
        feature_names = data.columns
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_exp, show=False)
    
    plt.title(
        f"SHAP Waterfall Plot\nPredicted Value: {int(model.predict(data)[0])}",
        fontsize=12
    )
    
    plt.subplots_adjust(bottom=0.25)

    plt.figtext(
        0.5, -0.2,
        "This plot shows how each feature contributed to the prediction.\n"
        "Red bars increase the prediction, blue bars decrease it.\n"
        "The model starts from the baseline and adds/subtracts contributions\n"
        "to arrive at the final prediction.",
        ha="center", fontsize = 15, wrap = True
    )
    
    plt.tight_layout()
    st.pyplot(fig)
    

    
